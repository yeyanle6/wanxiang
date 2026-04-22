from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from wanxiang.core.builtin_tools import create_default_registry
from wanxiang.core.factory import AgentFactory
from wanxiang.core.grader import grade_run
from wanxiang.core.message import Message, MessageStatus
from wanxiang.core.outcome_tagger import tag_run
from wanxiang.core.pipeline import WorkflowEngine
from wanxiang.core.storage import RunRecord, Storage
from wanxiang.core.tier import TierManager

from .events import RunEvent


@dataclass(slots=True)
class _RunState:
    queue: asyncio.Queue[RunEvent | None] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None
    completed: bool = False
    persisted: bool = False
    user_task: str = ""
    started_at: str = ""
    completed_at: str = ""
    final_status: str = MessageStatus.ERROR.value
    events: list[dict[str, Any]] = field(default_factory=list)
    level: int | None = None
    source: str = "user"
    probe: bool = False
    expected_keywords: list[str] | None = None


class RunManager:
    def __init__(
        self,
        factory: AgentFactory | None = None,
        llm_mode: str | None = None,
        tool_registry: Any = None,
        skill_forge: Any = None,
        tier_manager: TierManager | None = None,
        storage: Storage | None = None,
        usage_recorder: Any = None,
    ) -> None:
        if factory is not None:
            self.factory = factory
        else:
            registry = (
                tool_registry
                if tool_registry is not None
                else create_default_registry()
            )
            self.factory = AgentFactory(
                tool_registry=registry,
                llm_mode=llm_mode,
                skill_forge=skill_forge,
                usage_recorder=usage_recorder,
            )
        self.tier_manager = tier_manager if tier_manager is not None else TierManager()
        self.storage = storage
        self._runs: dict[str, _RunState] = {}
        self._lock = asyncio.Lock()
        self._persist_lock = asyncio.Lock()
        self._history_path = Path(__file__).resolve().parents[2] / "data" / "runs.jsonl"
        self._logger = logging.getLogger("wanxiang.run_manager")

    def has_run(self, run_id: str) -> bool:
        return run_id in self._runs

    async def read_raw_history(self) -> list[dict[str, Any]]:
        """Return every persisted run record, events included.

        Trace mining needs full event streams (not the summaries that
        `list_runs` returns). Reads from disk; does not include runs
        that are still in-flight.
        """
        return await asyncio.to_thread(self._read_history_records)

    async def list_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        max_items = max(1, limit)
        records = await asyncio.to_thread(self._read_history_records)
        records.sort(
            key=lambda item: str(item.get("completed_at") or item.get("started_at") or ""),
            reverse=True,
        )
        return [self._to_summary(item) for item in records[:max_items]]

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        state = self._runs.get(run_id)
        if state and state.events:
            return {
                "run_id": run_id,
                "task": state.user_task,
                "started_at": state.started_at,
                "completed_at": state.completed_at,
                "final_status": state.final_status,
                "events": list(state.events),
            }

        records = await asyncio.to_thread(self._read_history_records)
        for record in reversed(records):
            if str(record.get("run_id", "")) == run_id:
                return record
        return None

    async def start_run(
        self,
        task: str,
        *,
        level: int | None = None,
        source: str = "user",
        probe: bool = False,
        expected_keywords: list[str] | None = None,
    ) -> str:
        cleaned_task = task.strip()
        if not cleaned_task:
            raise ValueError("Task cannot be empty.")

        run_id = str(uuid4())
        state = _RunState(
            user_task=cleaned_task,
            level=level,
            source=source,
            probe=probe,
            expected_keywords=list(expected_keywords) if expected_keywords else None,
        )
        async with self._lock:
            self._runs[run_id] = state
            state.task = asyncio.create_task(self._execute_run(run_id, cleaned_task))
        return run_id

    async def wait_for_run(self, run_id: str) -> None:
        """Await a dispatched run's completion. No-op if run unknown.

        Autoschool calls this to track fire-and-forget dispatches so it
        can mark curriculum_queue rows 'done' once the run lands in
        storage. Exceptions from the run task are swallowed — persistence
        already logged them, and propagating here would kill the
        autoschool loop.
        """
        state = self._runs.get(run_id)
        if state is None or state.task is None:
            return
        try:
            await state.task
        except Exception:
            self._logger.exception("wait_for_run: run %s raised", run_id)

    async def stream_events(self, run_id: str):
        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(run_id)

        while True:
            event = await state.queue.get()
            if event is None:
                break
            yield event

    async def _execute_run(self, run_id: str, task: str) -> None:
        started_at_iso = datetime.now(timezone.utc).isoformat()
        started_at = time.perf_counter()
        trace_payload: list[dict[str, Any]] = []
        final_status = MessageStatus.ERROR.value
        run_started_sent = False
        llm_mode_configured = self.factory.llm_mode or "auto"
        llm_mode_effective: str | None = None
        completed_at_iso = started_at_iso
        state = self._runs.get(run_id)
        if state is not None:
            state.started_at = started_at_iso

        try:
            llm_mode_effective = await self._resolve_factory_mode()
            if state is not None and state.probe:
                plan = await self.factory.create_team_probe(task)
            else:
                plan = await self.factory.create_team(task)
            await self._publish(
                run_id,
                RunEvent.run_started(
                    run_id,
                    plan=plan.to_dict(),
                    llm_mode_configured=llm_mode_configured,
                    llm_mode_effective=llm_mode_effective,
                ),
            )
            run_started_sent = True

            event_handler = self._engine_event(run_id)
            agents = self.factory.instantiate_team(
                plan,
                on_tool_event=event_handler,
                effective_mode=llm_mode_effective,
            )
            engine = WorkflowEngine(agents=agents, plan=plan, on_event=event_handler)

            task_message = Message(intent=task, content=task, sender="user")
            trace = await engine.run(task_message)
            trace_payload = [item.to_dict() for item in trace]
            if trace:
                final_status = trace[-1].status.value
        except Exception as exc:
            if not run_started_sent:
                await self._publish(
                    run_id,
                    RunEvent.run_started(
                        run_id,
                        plan={
                            "workflow": "pipeline",
                            "max_iterations": 1,
                            "execution_order": [],
                            "rationale": f"run initialization failed: {type(exc).__name__}",
                            "agents": [],
                        },
                        llm_mode_configured=llm_mode_configured,
                        llm_mode_effective=llm_mode_effective,
                    ),
                )
            trace_payload = [self._error_trace_entry(content=str(exc))]
            final_status = MessageStatus.ERROR.value
        finally:
            total_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            completed_at_iso = datetime.now(timezone.utc).isoformat()
            state = self._runs.get(run_id)
            if state is not None:
                state.final_status = final_status
                state.completed_at = completed_at_iso
            await self._publish(
                run_id,
                RunEvent.run_completed(
                    run_id,
                    final_status=final_status,
                    total_steps=len(trace_payload),
                    total_elapsed_ms=total_elapsed_ms,
                    trace=trace_payload,
                ),
            )
            await self._persist_run(
                run_id,
                {
                    "run_id": run_id,
                    "task": task,
                    "started_at": started_at_iso,
                    "completed_at": completed_at_iso,
                    "final_status": final_status,
                },
            )
            await self._close_queue(run_id)

    def _engine_event(self, run_id: str):
        async def _handler(event: dict[str, Any]) -> None:
            event_type = str(event.get("type", ""))

            if event_type == "agent_started":
                await self._publish(
                    run_id,
                    RunEvent.agent_started(
                        run_id,
                        agent=str(event.get("agent", "")),
                        turn=int(event.get("turn", 0)),
                        iteration=self._maybe_int(event.get("iteration")),
                    ),
                )
                return

            if event_type == "agent_completed":
                content = str(event.get("content", ""))
                await self._publish(
                    run_id,
                    RunEvent.agent_completed(
                        run_id,
                        agent=str(event.get("agent", "")),
                        status=str(event.get("status", MessageStatus.ERROR.value)),
                        content=content,
                        content_preview=self._preview(content),
                        turn=int(event.get("turn", 0)),
                        elapsed_ms=int(event.get("elapsed_ms", 0)),
                        iteration=self._maybe_int(event.get("iteration")),
                    ),
                )
                return

            if event_type == "iteration_completed":
                await self._publish(
                    run_id,
                    RunEvent.iteration_completed(
                        run_id,
                        iteration=int(event.get("iteration", 0)),
                        reviewer_status=str(event.get("reviewer_status", MessageStatus.ERROR.value)),
                    ),
                )
                return

            if event_type == "parallel_completed":
                await self._publish(
                    run_id,
                    RunEvent.parallel_completed(
                        run_id,
                        parallel_agents=[
                            str(name) for name in event.get("parallel_agents", []) if str(name).strip()
                        ],
                        successful_agents=[
                            str(name) for name in event.get("successful_agents", []) if str(name).strip()
                        ],
                        failed_agents=[
                            str(name) for name in event.get("failed_agents", []) if str(name).strip()
                        ],
                        success_count=int(event.get("success_count", 0)),
                        failed_count=int(event.get("failed_count", 0)),
                        synthesizer=str(event.get("synthesizer", "")),
                    ),
                )
                return

            if event_type == "tool_started":
                await self._publish(
                    run_id,
                    RunEvent.tool_started(
                        run_id,
                        agent=str(event.get("agent", "")),
                        tool=str(event.get("tool", "")),
                        arguments=self._safe_object(event.get("arguments")),
                        turn=self._maybe_int(event.get("turn")),
                        tool_round=self._maybe_int(event.get("tool_round")),
                    ),
                )
                return

            if event_type == "tool_completed":
                tool_name = str(event.get("tool", ""))
                success = bool(event.get("success", False))
                self.tier_manager.record_result(tool_name, success, run_id)
                await self._publish(
                    run_id,
                    RunEvent.tool_completed(
                        run_id,
                        agent=str(event.get("agent", "")),
                        tool=tool_name,
                        success=success,
                        elapsed_ms=int(event.get("elapsed_ms", 0)),
                        content_preview=self._preview(str(event.get("content_preview", ""))),
                        turn=self._maybe_int(event.get("turn")),
                        tool_round=self._maybe_int(event.get("tool_round")),
                        error=self._maybe_str(event.get("error")),
                    ),
                )
                return

        return _handler

    async def _resolve_factory_mode(self) -> str | None:
        resolver = getattr(self.factory.client, "resolve_mode", None)
        if not callable(resolver):
            return self.factory.llm_mode or "auto"
        try:
            resolved = resolver(require_tools=False)
            if asyncio.iscoroutine(resolved):
                resolved = await resolved
            if resolved is None:
                return None
            return str(resolved)
        except Exception:
            return None

    async def _publish(self, run_id: str, event: RunEvent) -> None:
        state = self._runs.get(run_id)
        if state is None:
            return
        state.events.append(event.to_dict())
        await state.queue.put(event)

    async def _close_queue(self, run_id: str) -> None:
        state = self._runs.get(run_id)
        if state is None or state.completed:
            return
        state.completed = True
        await state.queue.put(None)

    def _preview(self, content: str, limit: int = 200) -> str:
        compact = " ".join(content.strip().split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    def _maybe_int(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _maybe_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _safe_object(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return {str(k): v for k, v in value.items()}
        return {}

    def _error_trace_entry(self, content: str) -> dict[str, Any]:
        return {
            "id": str(uuid4()),
            "intent": "run_failed",
            "content": content,
            "sender": "run_manager",
            "status": MessageStatus.ERROR.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parent_id": None,
            "context": [],
            "turn": 1,
            "metadata": {"source": "run_manager"},
        }

    async def _persist_run(self, run_id: str, base_record: dict[str, Any]) -> None:
        state = self._runs.get(run_id)
        if state is None or state.persisted:
            return

        record = {
            "run_id": run_id,
            "task": str(base_record.get("task", state.user_task)),
            "started_at": str(base_record.get("started_at", state.started_at)),
            "completed_at": str(base_record.get("completed_at", state.completed_at)),
            "final_status": str(base_record.get("final_status", state.final_status)),
            "events": list(state.events),
        }
        async with self._persist_lock:
            await asyncio.to_thread(self._append_history_record, record)
            if self.storage is not None:
                try:
                    await asyncio.to_thread(
                        self._write_storage_record,
                        record,
                        state.level,
                        state.source,
                        state.expected_keywords,
                    )
                except Exception:
                    self._logger.exception(
                        "SQLite dual-write failed for run %s; jsonl write succeeded", run_id
                    )
        state.persisted = True

    def _write_storage_record(
        self,
        record: dict[str, Any],
        level: int | None,
        source: str,
        expected_keywords: list[str] | None,
    ) -> None:
        events = record.get("events") if isinstance(record.get("events"), list) else []
        outcome = tag_run(events, record.get("final_status"))
        run_record = RunRecord(
            run_id=str(record["run_id"]),
            task=str(record.get("task", "")),
            started_at=str(record.get("started_at", "")),
            completed_at=str(record.get("completed_at") or "") or None,
            final_status=str(record.get("final_status") or "") or None,
            outcome=outcome,
            level=level,
            source=source,
            event_count=len(events),
            events=events,
        )
        self.storage.upsert_run(run_record)
        self.storage.update_outcome(run_record.run_id, outcome)

        # Grade: even runs without expected_keywords get a verdict based
        # on outcome alone. This is how graduation judges reading
        # graded_pass get a uniform signal regardless of task origin.
        grade = grade_run(events, outcome, expected_keywords)
        self.storage.update_grade(
            run_record.run_id, passed=grade.passed, reason=grade.reason
        )

    def _append_history_record(self, record: dict[str, Any]) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        with self._history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read_history_records(self) -> list[dict[str, Any]]:
        if not self._history_path.exists():
            return []

        rows: list[dict[str, Any]] = []
        with self._history_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
        return rows

    def _to_summary(self, record: dict[str, Any]) -> dict[str, Any]:
        events = record.get("events")
        event_list = events if isinstance(events, list) else []

        step_count = len([event for event in event_list if event.get("type") == "agent_completed"])
        for event in reversed(event_list):
            if event.get("type") == "run_completed":
                data = event.get("data")
                if isinstance(data, dict):
                    total_steps = data.get("total_steps")
                    if isinstance(total_steps, int):
                        step_count = total_steps
                break

        return {
            "run_id": str(record.get("run_id", "")),
            "task": str(record.get("task", "")),
            "started_at": str(record.get("started_at", "")),
            "completed_at": str(record.get("completed_at", "")),
            "final_status": str(record.get("final_status", MessageStatus.ERROR.value)),
            "step_count": step_count,
            "event_count": len(event_list),
        }
