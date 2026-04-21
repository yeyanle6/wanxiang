"""Integration tests for Phase 6.3: TierManager wired into RunManager and SkillForge.

Covers:
  - RunManager creates a TierManager by default
  - tool_completed events feed record_result on the manager's TierManager
  - External TierManager can be injected into RunManager
  - SkillForge calls initialize_tool after successful registration
  - mine_traces includes tier_changes in the report
  - /api/tier endpoint returns TierSummaryResponse
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from wanxiang.core.tier import TierManager
from wanxiang.core.trace_mining import mine_traces
from wanxiang.server.runner import RunManager, _RunState


# ---------------------------------------------------------------------------
# RunManager — TierManager wiring
# ---------------------------------------------------------------------------


class TestRunManagerTierWiring:
    def test_creates_tier_manager_by_default(self):
        rm = RunManager()
        assert isinstance(rm.tier_manager, TierManager)

    def test_accepts_injected_tier_manager(self):
        tm = TierManager(window_size=5)
        rm = RunManager(tier_manager=tm)
        assert rm.tier_manager is tm

    def test_tool_completed_records_success(self):
        tm = TierManager()
        rm = RunManager(tier_manager=tm)
        run_id = "run-abc"
        rm._runs[run_id] = _RunState()  # noqa: SLF001

        async def _exercise():
            handler = rm._engine_event(run_id)  # noqa: SLF001
            await handler(
                {
                    "type": "tool_completed",
                    "agent": "researcher",
                    "tool": "web_search",
                    "success": True,
                    "elapsed_ms": 50,
                    "content_preview": "some result",
                }
            )

        asyncio.run(_exercise())

        tier = tm.get_tier("web_search")
        assert tier is not None
        assert tier.total_calls == 1
        assert tier.level == 1  # first success → Level 1

    def test_tool_completed_records_failure(self):
        tm = TierManager()
        rm = RunManager(tier_manager=tm)
        run_id = "run-fail"
        rm._runs[run_id] = _RunState()  # noqa: SLF001

        async def _exercise():
            handler = rm._engine_event(run_id)  # noqa: SLF001
            await handler(
                {
                    "type": "tool_completed",
                    "agent": "researcher",
                    "tool": "echo",
                    "success": False,
                    "elapsed_ms": 10,
                    "content_preview": "",
                }
            )

        asyncio.run(_exercise())

        tier = tm.get_tier("echo")
        assert tier is not None
        assert tier.total_calls == 1
        assert tier.level == 0  # first call failed → stays at 0

    def test_multiple_runs_accumulate_distinct_run_ids(self):
        tm = TierManager()
        rm = RunManager(tier_manager=tm)

        async def _fire(run_id: str):
            rm._runs[run_id] = _RunState()  # noqa: SLF001
            handler = rm._engine_event(run_id)  # noqa: SLF001
            await handler(
                {
                    "type": "tool_completed",
                    "agent": "a",
                    "tool": "current_time",
                    "success": True,
                    "elapsed_ms": 5,
                    "content_preview": "",
                }
            )

        async def _exercise():
            await _fire("run-1")
            await _fire("run-2")
            await _fire("run-3")

        asyncio.run(_exercise())

        tier = tm.get_tier("current_time")
        assert len(tier.successful_run_ids) == 3

    def test_shared_tier_manager_between_two_run_managers(self):
        tm = TierManager()
        rm1 = RunManager(tier_manager=tm)
        rm2 = RunManager(tier_manager=tm)

        async def _exercise():
            rm1._runs["r1"] = _RunState()  # noqa: SLF001
            h1 = rm1._engine_event("r1")  # noqa: SLF001
            await h1(
                {"type": "tool_completed", "agent": "a", "tool": "echo",
                 "success": True, "elapsed_ms": 1, "content_preview": ""}
            )
            rm2._runs["r2"] = _RunState()  # noqa: SLF001
            h2 = rm2._engine_event("r2")  # noqa: SLF001
            await h2(
                {"type": "tool_completed", "agent": "b", "tool": "echo",
                 "success": True, "elapsed_ms": 1, "content_preview": ""}
            )

        asyncio.run(_exercise())

        assert tm.get_tier("echo").total_calls == 2


# ---------------------------------------------------------------------------
# SkillForge — initialize_tool on successful registration
# ---------------------------------------------------------------------------


class TestSkillForgeTierWiring:
    def _make_forge(self, tm: TierManager):
        from wanxiang.core.sandbox import SandboxExecutor
        from wanxiang.core.skill_forge import SkillForge
        from wanxiang.core.tools import ToolRegistry

        registry = ToolRegistry()
        sandbox = MagicMock(spec=SandboxExecutor)
        synthesizer = MagicMock()
        forge = SkillForge(
            sandbox=sandbox,
            registry=registry,
            synthesizer=synthesizer,
            tier_manager=tm,
        )
        return forge, registry

    def test_initialize_tool_called_on_success(self):
        from wanxiang.core.sandbox import SandboxResult
        from wanxiang.core.skill_forge import parse_synthesizer_response

        tm = TierManager()
        forge, registry = self._make_forge(tm)

        handler_code = "def handler(x: int) -> int:\n    return x + 1\n"
        test_code = (
            "from handler_module import handler\n"
            "def test_it():\n    assert handler(1) == 2\n"
        )
        synth_payload = {
            "tool_name": "add_one",
            "description": "adds one",
            "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            "handler_code": handler_code,
            "test_code": test_code,
        }
        import json

        raw_response = json.dumps(synth_payload)
        reply_msg = MagicMock()
        reply_msg.content = raw_response
        forge.synthesizer.execute = AsyncMock(return_value=reply_msg)

        passed_result = SandboxResult(
            success=True, passed=True, exit_code=0,
            stdout="1 passed", stderr="",
            stdout_truncated=False, stderr_truncated=False,
            elapsed_ms=10, timed_out=False, error=None,
        )
        forge.sandbox.execute = AsyncMock(return_value=passed_result)

        result = asyncio.run(forge.forge("add one to a number"))

        assert result.success
        assert result.registered
        assert tm.get_tier("add_one") is not None
        assert tm.get_tier("add_one").level == 0  # initialized at level 0

    def test_no_tier_manager_does_not_crash(self):
        from wanxiang.core.sandbox import SandboxResult
        from wanxiang.core.skill_forge import SkillForge
        from wanxiang.core.tools import ToolRegistry

        registry = ToolRegistry()
        sandbox = MagicMock()
        synthesizer = MagicMock()
        forge = SkillForge(
            sandbox=sandbox,
            registry=registry,
            synthesizer=synthesizer,
            # tier_manager omitted — should default to None
        )
        assert forge.tier_manager is None

        handler_code = "def handler(x: int) -> int:\n    return x\n"
        test_code = (
            "from handler_module import handler\n"
            "def test_it():\n    assert handler(5) == 5\n"
        )
        import json

        synth_payload = {
            "tool_name": "identity",
            "description": "returns input",
            "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            "handler_code": handler_code,
            "test_code": test_code,
        }
        reply_msg = MagicMock()
        reply_msg.content = json.dumps(synth_payload)
        forge.synthesizer.execute = AsyncMock(return_value=reply_msg)

        passed_result = SandboxResult(
            success=True, passed=True, exit_code=0,
            stdout="1 passed", stderr="",
            stdout_truncated=False, stderr_truncated=False,
            elapsed_ms=10, timed_out=False, error=None,
        )
        forge.sandbox.execute = AsyncMock(return_value=passed_result)

        result = asyncio.run(forge.forge("return identity"))
        assert result.success  # no crash despite no tier_manager


# ---------------------------------------------------------------------------
# mine_traces — tier_changes in report
# ---------------------------------------------------------------------------


class TestMineTracesTierChanges:
    def test_tier_changes_empty_by_default(self):
        report = mine_traces([])
        assert report.tier_changes == []

    def test_tier_changes_passed_through(self):
        changes = [
            {
                "tool_name": "web_search",
                "old_level": 0,
                "new_level": 1,
                "reason": "first successful real-task call",
                "timestamp": "2026-04-21T00:00:00+00:00",
            }
        ]
        report = mine_traces([], tier_changes=changes)
        assert len(report.tier_changes) == 1
        assert report.tier_changes[0]["tool_name"] == "web_search"

    def test_tier_changes_in_to_dict(self):
        changes = [{"tool_name": "echo", "old_level": 1, "new_level": 0,
                    "reason": "downgraded", "timestamp": "2026-04-21T00:00:00+00:00"}]
        report = mine_traces([], tier_changes=changes)
        d = report.to_dict()
        assert "tier_changes" in d
        assert d["tier_changes"][0]["tool_name"] == "echo"

    def test_tier_changes_none_yields_empty_list(self):
        report = mine_traces([], tier_changes=None)
        assert report.tier_changes == []
