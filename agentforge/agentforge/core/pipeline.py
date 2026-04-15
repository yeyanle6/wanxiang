from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from .factory import TeamPlan
from .message import Message, MessageStatus

StepCallback = Callable[[Message], Awaitable[None] | None]
EventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


class WorkflowEngine:
    def __init__(
        self,
        agents: dict[str, Any],
        plan: TeamPlan,
        on_event: EventCallback | None = None,
    ) -> None:
        self.agents = agents
        self.plan = plan
        self.on_event = on_event
        self.logger = logging.getLogger("agentforge.workflow")
        self._validate_plan()

    async def run(self, task: Message, on_step: StepCallback | None = None) -> list[Message]:
        if self.plan.workflow == "pipeline":
            return await self._run_pipeline(task, on_step=on_step)
        if self.plan.workflow == "review_loop":
            return await self._run_review_loop(task, on_step=on_step)
        if self.plan.workflow == "parallel":
            return await self._run_parallel(task, on_step=on_step)
        raise ValueError(f"Unsupported workflow: {self.plan.workflow}")

    async def _run_pipeline(self, task: Message, on_step: StepCallback | None = None) -> list[Message]:
        trace: list[Message] = []
        current = task

        for name in self.plan.execution_order:
            agent = self.agents[name]
            self.logger.info("Pipeline execute: agent=%s input_id=%s", name, current.id)
            await self._emit(
                {
                    "type": "agent_started",
                    "agent": name,
                    "turn": current.turn,
                    "iteration": None,
                }
            )
            started_at = time.perf_counter()
            output = await agent.execute(current)
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            trace.append(output)
            await self._emit(
                {
                    "type": "agent_completed",
                    "agent": name,
                    "status": output.status.value,
                    "content": output.content,
                    "turn": output.turn,
                    "iteration": None,
                    "elapsed_ms": elapsed_ms,
                    "message": output,
                }
            )
            await self._notify_step(on_step, output)

            if output.status == MessageStatus.ERROR:
                self.logger.warning("Pipeline terminated by error from agent=%s", name)
                break
            current = output

        return trace

    async def _run_review_loop(self, task: Message, on_step: StepCallback | None = None) -> list[Message]:
        if len(self.plan.execution_order) < 2:
            raise ValueError("review_loop requires at least two agents in execution_order.")

        producer_name = self.plan.execution_order[0]
        reviewer_name = self.plan.execution_order[1]
        producer = self.agents[producer_name]
        reviewer = self.agents[reviewer_name]

        trace: list[Message] = []
        current_input = task

        for iteration in range(1, self.plan.max_iterations + 1):
            self.logger.info(
                "Review loop iteration=%s producer=%s reviewer=%s",
                iteration,
                producer_name,
                reviewer_name,
            )

            await self._emit(
                {
                    "type": "agent_started",
                    "agent": producer_name,
                    "turn": current_input.turn,
                    "iteration": iteration,
                }
            )
            producer_started_at = time.perf_counter()
            produced = await producer.execute(current_input)
            producer_elapsed_ms = int((time.perf_counter() - producer_started_at) * 1000)
            trace.append(produced)
            await self._emit(
                {
                    "type": "agent_completed",
                    "agent": producer_name,
                    "status": produced.status.value,
                    "content": produced.content,
                    "turn": produced.turn,
                    "iteration": iteration,
                    "elapsed_ms": producer_elapsed_ms,
                    "message": produced,
                }
            )
            await self._notify_step(on_step, produced)
            if produced.status == MessageStatus.ERROR:
                self.logger.warning("Review loop terminated by producer error.")
                break

            await self._emit(
                {
                    "type": "agent_started",
                    "agent": reviewer_name,
                    "turn": produced.turn,
                    "iteration": iteration,
                }
            )
            reviewer_started_at = time.perf_counter()
            reviewed = await reviewer.execute(produced)
            reviewer_elapsed_ms = int((time.perf_counter() - reviewer_started_at) * 1000)
            trace.append(reviewed)
            await self._emit(
                {
                    "type": "agent_completed",
                    "agent": reviewer_name,
                    "status": reviewed.status.value,
                    "content": reviewed.content,
                    "turn": reviewed.turn,
                    "iteration": iteration,
                    "elapsed_ms": reviewer_elapsed_ms,
                    "message": reviewed,
                }
            )
            await self._notify_step(on_step, reviewed)
            await self._emit(
                {
                    "type": "iteration_completed",
                    "iteration": iteration,
                    "reviewer_status": reviewed.status.value,
                }
            )
            if reviewed.status == MessageStatus.ERROR:
                self.logger.warning("Review loop terminated by reviewer error.")
                break

            if reviewed.status == MessageStatus.NEEDS_REVISION and iteration < self.plan.max_iterations:
                current_input = reviewed
                continue

            break

        return trace

    async def _run_parallel(self, task: Message, on_step: StepCallback | None = None) -> list[Message]:
        if len(self.plan.execution_order) < 2:
            raise ValueError("parallel workflow requires at least two agents in execution_order.")

        parallel_names = self.plan.execution_order[:-1]
        synthesizer_name = self.plan.execution_order[-1]
        trace: list[Message] = []

        for name in parallel_names:
            await self._emit(
                {
                    "type": "agent_started",
                    "agent": name,
                    "turn": task.turn,
                    "iteration": None,
                }
            )

        async def _execute_parallel_agent(name: str) -> tuple[str, Message, int]:
            started_at = time.perf_counter()
            try:
                output = await self.agents[name].execute(task)
            except Exception as exc:
                output = task.create_reply(
                    intent=f"Parallel branch failed: {name}",
                    content=f"Agent {name} raised {type(exc).__name__}: {exc}",
                    sender=name,
                    status=MessageStatus.ERROR,
                    metadata={"parallel_exception": True, "error_type": type(exc).__name__},
                )
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            return name, output, elapsed_ms

        branch_results = await asyncio.gather(
            *[_execute_parallel_agent(name) for name in parallel_names],
            return_exceptions=False,
        )

        successful: list[tuple[str, Message]] = []
        failed: list[tuple[str, Message]] = []

        for name, output, elapsed_ms in branch_results:
            trace.append(output)
            await self._emit(
                {
                    "type": "agent_completed",
                    "agent": name,
                    "status": output.status.value,
                    "content": output.content,
                    "turn": output.turn,
                    "iteration": None,
                    "elapsed_ms": elapsed_ms,
                    "message": output,
                }
            )
            await self._notify_step(on_step, output)
            if output.status == MessageStatus.SUCCESS:
                successful.append((name, output))
            else:
                failed.append((name, output))

        await self._emit(
            {
                "type": "parallel_completed",
                "parallel_agents": list(parallel_names),
                "successful_agents": [name for name, _ in successful],
                "failed_agents": [name for name, _ in failed],
                "success_count": len(successful),
                "failed_count": len(failed),
                "synthesizer": synthesizer_name,
            }
        )

        if not successful:
            failure_summary = self._build_parallel_failure_summary(task, parallel_names, failed)
            trace.append(failure_summary)
            await self._notify_step(on_step, failure_summary)
            return trace

        merged_input = self._merge_parallel_results(task, successful, failed)

        await self._emit(
            {
                "type": "agent_started",
                "agent": synthesizer_name,
                "turn": merged_input.turn,
                "iteration": None,
            }
        )
        started_at = time.perf_counter()
        synthesized = await self.agents[synthesizer_name].execute(merged_input)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        trace.append(synthesized)
        await self._emit(
            {
                "type": "agent_completed",
                "agent": synthesizer_name,
                "status": synthesized.status.value,
                "content": synthesized.content,
                "turn": synthesized.turn,
                "iteration": None,
                "elapsed_ms": elapsed_ms,
                "message": synthesized,
            }
        )
        await self._notify_step(on_step, synthesized)
        return trace

    def _merge_parallel_results(
        self,
        task: Message,
        successful: list[tuple[str, Message]],
        failed: list[tuple[str, Message]],
    ) -> Message:
        blocks: list[str] = []
        for name, message in successful:
            blocks.append(
                f"[Agent: {name}]\n"
                f"Status: {message.status.value}\n"
                f"Content:\n{message.content}"
            )

        failed_names = [name for name, _ in failed]
        if failed_names:
            blocks.append(
                f"[Failed agents ignored by fail-tolerant strategy]\n{', '.join(failed_names)}"
            )

        merged_content = (
            "Parallel branch outputs from multiple agents.\n"
            "Please synthesize them into one coherent final result.\n\n"
            + "\n\n---\n\n".join(blocks)
        )
        return task.create_reply(
            intent=(
                "Synthesize the successful parallel outputs into a single high-quality result. "
                "Resolve overlaps and preserve key points."
            ),
            content=merged_content,
            sender="parallel_merge",
            status=MessageStatus.SUCCESS,
            metadata={
                "parallel_successful_agents": [name for name, _ in successful],
                "parallel_failed_agents": failed_names,
                "parallel_branch_count": len(successful) + len(failed),
            },
        )

    def _build_parallel_failure_summary(
        self,
        task: Message,
        parallel_names: list[str],
        failed: list[tuple[str, Message]],
    ) -> Message:
        details = []
        for name, message in failed:
            details.append(f"- {name}: {message.content}")

        content = (
            "Parallel stage failed: all branches returned errors.\n"
            f"Branches: {', '.join(parallel_names)}\n"
            + ("\n".join(details) if details else "No branch output available.")
        )
        return task.create_reply(
            intent="Parallel stage failed before synthesis.",
            content=content,
            sender="parallel_stage",
            status=MessageStatus.ERROR,
            metadata={
                "parallel_failed_agents": [name for name, _ in failed],
                "parallel_branch_count": len(parallel_names),
            },
        )

    def _validate_plan(self) -> None:
        if not self.plan.execution_order:
            raise ValueError("execution_order cannot be empty.")

        unknown = [name for name in self.plan.execution_order if name not in self.agents]
        if unknown:
            raise ValueError(f"Agents not found in runtime map: {', '.join(unknown)}")

    async def _notify_step(self, on_step: StepCallback | None, message: Message) -> None:
        if on_step is None:
            return
        maybe_awaitable = on_step(message)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        maybe_awaitable = self.on_event(event)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable
