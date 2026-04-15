from __future__ import annotations

import argparse
import asyncio
import os

from .core.factory import AgentFactory, TeamPlan
from .core.message import Message
from .core.pipeline import WorkflowEngine
from .core.builtin_tools import create_default_registry


def _separator(title: str) -> None:
    print(f"\n=== {title} ===")


def _truncate(text: str, limit: int = 200) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _print_plan(plan: TeamPlan) -> None:
    _separator("Team Plan")
    print(f"Workflow: {plan.workflow}")
    print(f"Execution Order: {' -> '.join(plan.execution_order)}")
    print(f"Max Iterations: {plan.max_iterations}")
    if plan.rationale:
        print(f"Rationale: {plan.rationale}")
    print("Agents:")
    for spec in plan.agents:
        print(f"- {spec.name}: {spec.duty}")


def _confirm_or_exit(auto_confirm: bool) -> bool:
    if auto_confirm:
        return True
    try:
        answer = input("Continue execution? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


async def _run_task(task: str, *, auto_confirm: bool = False, llm_mode: str | None = None) -> int:
    _separator("Wanxiang")
    print(f"Task: {task}")

    factory = AgentFactory(
        tool_registry=create_default_registry(),
        llm_mode=llm_mode or os.getenv("WANXIANG_LLM_MODE"),
    )

    _separator("Stage 1/3 - Plan Team")
    print("Generating team plan...")
    try:
        plan = await factory.create_team(task)
    except Exception as exc:
        print(f"Plan generation failed: {exc}")
        return 1

    _print_plan(plan)
    if not _confirm_or_exit(auto_confirm):
        print("Execution canceled.")
        return 0

    _separator("Stage 2/3 - Instantiate Agents")
    async def on_tool_event(event: dict) -> None:
        tool = event.get("tool", "")
        event_type = event.get("type", "")
        agent_name = event.get("agent", "")
        if event_type == "tool_started":
            print(f"[Tool] agent={agent_name} tool={tool} started")
        elif event_type == "tool_completed":
            print(
                f"[Tool] agent={agent_name} tool={tool} "
                f"success={event.get('success')} elapsed={event.get('elapsed_ms')}ms"
            )

    agents = factory.instantiate_team(plan, on_tool_event=on_tool_event)
    print(f"Instantiated {len(agents)} agent(s).")

    _separator("Stage 3/3 - Run Workflow")
    engine = WorkflowEngine(agents=agents, plan=plan)
    task_message = Message(intent=task, content=task, sender="user")
    trace_counter = 0

    async def on_step(message: Message) -> None:
        nonlocal trace_counter
        trace_counter += 1
        summary = _truncate(message.content, limit=200)
        print(
            f"[Step {trace_counter}] sender={message.sender} "
            f"status={message.status.value} content={summary}"
        )

    try:
        trace = await engine.run(task_message, on_step=on_step)
    except Exception as exc:
        print(f"Workflow execution failed: {exc}")
        return 1

    _separator("Execution Trace Summary")
    print(f"Total messages: {len(trace)}")
    if not trace:
        print("No output produced.")
        return 1

    final_message = trace[-1]
    print(f"Final sender: {final_message.sender}")
    print(f"Final status: {final_message.status.value}")

    _separator("Final Content")
    print(final_message.content)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wanxiang workflow from the command line.")
    parser.add_argument("task", nargs="*", help="Task description for the multi-agent run.")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip interactive confirmation and execute immediately.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["auto", "api", "cli"],
        default=None,
        help="LLM backend mode. Default uses WANXIANG_LLM_MODE or auto detection.",
    )
    args = parser.parse_args()

    task = " ".join(args.task).strip()
    if not task:
        try:
            task = input("Enter task: ").strip()
        except EOFError:
            print("No task provided.")
            raise SystemExit(1) from None

    if not task:
        print("Task cannot be empty.")
        raise SystemExit(1)

    exit_code = asyncio.run(_run_task(task, auto_confirm=args.yes, llm_mode=args.llm_mode))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
