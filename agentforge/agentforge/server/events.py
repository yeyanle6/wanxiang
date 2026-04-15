from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class RunEvent:
    type: str
    run_id: str
    data: dict[str, Any]
    timestamp: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def run_started(
        cls,
        run_id: str,
        plan: dict[str, Any],
        *,
        llm_mode_configured: str | None = None,
        llm_mode_effective: str | None = None,
    ) -> RunEvent:
        payload: dict[str, Any] = {"run_id": run_id, "plan": plan}
        if llm_mode_configured is not None:
            payload["llm_mode_configured"] = llm_mode_configured
        if llm_mode_effective is not None:
            payload["llm_mode_effective"] = llm_mode_effective
        return cls(
            type="run_started",
            run_id=run_id,
            data=payload,
        )

    @classmethod
    def agent_started(
        cls,
        run_id: str,
        *,
        agent: str,
        turn: int,
        iteration: int | None,
    ) -> RunEvent:
        payload: dict[str, Any] = {"agent": agent, "turn": turn}
        if iteration is not None:
            payload["iteration"] = iteration
        return cls(type="agent_started", run_id=run_id, data=payload)

    @classmethod
    def agent_completed(
        cls,
        run_id: str,
        *,
        agent: str,
        status: str,
        content: str,
        content_preview: str,
        turn: int,
        elapsed_ms: int,
        iteration: int | None,
    ) -> RunEvent:
        payload: dict[str, Any] = {
            "agent": agent,
            "status": status,
            "content": content,
            "content_preview": content_preview,
            "turn": turn,
            "elapsed_ms": elapsed_ms,
        }
        if iteration is not None:
            payload["iteration"] = iteration
        return cls(type="agent_completed", run_id=run_id, data=payload)

    @classmethod
    def tool_started(
        cls,
        run_id: str,
        *,
        agent: str,
        tool: str,
        arguments: dict[str, Any],
        turn: int | None,
        tool_round: int | None,
    ) -> RunEvent:
        payload: dict[str, Any] = {
            "agent": agent,
            "tool": tool,
            "arguments": arguments,
        }
        if turn is not None:
            payload["turn"] = turn
        if tool_round is not None:
            payload["tool_round"] = tool_round
        return cls(type="tool_started", run_id=run_id, data=payload)

    @classmethod
    def tool_completed(
        cls,
        run_id: str,
        *,
        agent: str,
        tool: str,
        success: bool,
        elapsed_ms: int,
        content_preview: str,
        turn: int | None,
        tool_round: int | None,
        error: str | None = None,
    ) -> RunEvent:
        payload: dict[str, Any] = {
            "agent": agent,
            "tool": tool,
            "success": success,
            "elapsed_ms": elapsed_ms,
            "content_preview": content_preview,
        }
        if turn is not None:
            payload["turn"] = turn
        if tool_round is not None:
            payload["tool_round"] = tool_round
        if error is not None:
            payload["error"] = error
        return cls(type="tool_completed", run_id=run_id, data=payload)

    @classmethod
    def iteration_completed(
        cls,
        run_id: str,
        *,
        iteration: int,
        reviewer_status: str,
    ) -> RunEvent:
        return cls(
            type="iteration_completed",
            run_id=run_id,
            data={"iteration": iteration, "reviewer_status": reviewer_status},
        )

    @classmethod
    def parallel_completed(
        cls,
        run_id: str,
        *,
        parallel_agents: list[str],
        successful_agents: list[str],
        failed_agents: list[str],
        success_count: int,
        failed_count: int,
        synthesizer: str,
    ) -> RunEvent:
        return cls(
            type="parallel_completed",
            run_id=run_id,
            data={
                "parallel_agents": parallel_agents,
                "successful_agents": successful_agents,
                "failed_agents": failed_agents,
                "success_count": success_count,
                "failed_count": failed_count,
                "synthesizer": synthesizer,
            },
        )

    @classmethod
    def run_completed(
        cls,
        run_id: str,
        *,
        final_status: str,
        total_steps: int,
        total_elapsed_ms: int,
        trace: list[dict[str, Any]],
    ) -> RunEvent:
        return cls(
            type="run_completed",
            run_id=run_id,
            data={
                "run_id": run_id,
                "final_status": final_status,
                "total_steps": total_steps,
                "total_elapsed_ms": total_elapsed_ms,
                "trace": trace,
            },
        )
