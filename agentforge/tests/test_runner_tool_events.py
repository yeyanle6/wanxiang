import asyncio

from agentforge.server.runner import RunManager, _RunState


def test_run_manager_maps_tool_events_into_run_events() -> None:
    manager = RunManager()
    run_id = "test-run"
    manager._runs[run_id] = _RunState()  # noqa: SLF001 - test-level setup
    handler = manager._engine_event(run_id)  # noqa: SLF001 - test-level access

    async def _exercise() -> None:
        await handler(
            {
                "type": "tool_started",
                "agent": "researcher",
                "tool": "current_time",
                "arguments": {},
                "turn": 2,
                "tool_round": 1,
            }
        )
        await handler(
            {
                "type": "tool_completed",
                "agent": "researcher",
                "tool": "current_time",
                "success": True,
                "elapsed_ms": 12,
                "content_preview": "2026-04-15T00:00:00+00:00",
                "turn": 2,
                "tool_round": 1,
            }
        )

    asyncio.run(_exercise())

    events = manager._runs[run_id].events  # noqa: SLF001 - test-level access
    assert len(events) == 2
    assert events[0]["type"] == "tool_started"
    assert events[0]["data"]["tool"] == "current_time"
    assert events[1]["type"] == "tool_completed"
    assert events[1]["data"]["success"] is True
