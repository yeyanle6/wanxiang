"""Tests for the probe-bypass lane: AgentFactory.create_team_probe +
RunManager.start_run(probe=True).

Goal: verify that autoschool's trivial L0/L1 tasks can skip the Director
LLM call and the policy layer's reviewer injection, landing on a single
agent pipeline that consumes ~1 LLM call instead of ~3.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from wanxiang.core.factory import AgentFactory
from wanxiang.core.plan import TeamPlan
from wanxiang.server.runner import RunManager


@pytest.fixture
def factory():
    # Minimal factory — no real LLM client needed, we don't call create_team.
    f = AgentFactory(api_key="test-key-not-used")
    return f


class TestCreateTeamProbe:
    def test_returns_single_agent_pipeline(self, factory):
        plan = asyncio.run(factory.create_team_probe("say hello"))
        assert isinstance(plan, TeamPlan)
        assert plan.workflow == "pipeline"
        assert len(plan.agents) == 1
        assert plan.agents[0].name == "responder"
        assert plan.execution_order == ["responder"]
        assert plan.max_iterations == 1

    def test_no_reviewer_injected(self, factory):
        plan = asyncio.run(factory.create_team_probe("say hello"))
        names = [a.name for a in plan.agents]
        assert "reviewer" not in names
        assert "writer" not in names

    def test_empty_task_rejected(self, factory):
        with pytest.raises(ValueError, match="non-empty"):
            asyncio.run(factory.create_team_probe("   "))

    def test_rationale_marks_bypass(self, factory):
        plan = asyncio.run(factory.create_team_probe("hello"))
        assert "bypass" in plan.rationale.lower() or "probe" in plan.rationale.lower()

    def test_no_llm_call_made(self, factory):
        # Scribble the client so we can detect any accidental call.
        factory.client = MagicMock()
        factory.client.generate = AsyncMock()

        asyncio.run(factory.create_team_probe("hello"))

        factory.client.generate.assert_not_called()

    def test_responder_has_minimal_identity(self, factory):
        plan = asyncio.run(factory.create_team_probe("hello"))
        agent = plan.agents[0]
        assert agent.allowed_tools in (None, [])
        assert agent.native_tools in (None, [])
        # The identity should discourage tool use / clarification loops.
        assert "concise" in agent.base_identity.lower() or "direct" in agent.base_identity.lower()


class TestStartRunProbeFlag:
    @pytest.mark.asyncio
    async def test_probe_true_is_recorded_in_state(self, tmp_path):
        rm = RunManager()
        rm._history_path = tmp_path / "runs.jsonl"
        rm._execute_run = AsyncMock(return_value=None)

        run_id = await rm.start_run("hello", probe=True)
        state = rm._runs[run_id]
        assert state.probe is True

    @pytest.mark.asyncio
    async def test_probe_defaults_false(self, tmp_path):
        rm = RunManager()
        rm._history_path = tmp_path / "runs.jsonl"
        rm._execute_run = AsyncMock(return_value=None)

        run_id = await rm.start_run("hello")
        state = rm._runs[run_id]
        assert state.probe is False


class TestExecuteRunRoutesToProbe:
    """Verify _execute_run dispatches to create_team_probe vs create_team."""

    @pytest.mark.asyncio
    async def test_probe_run_uses_create_team_probe(self, tmp_path):
        rm = RunManager()
        rm._history_path = tmp_path / "runs.jsonl"

        rm.factory.create_team_probe = AsyncMock(
            return_value=TeamPlan(
                agents=[], workflow="pipeline", execution_order=[], rationale="probe"
            )
        )
        rm.factory.create_team = AsyncMock()
        # Short-circuit the workflow execution past plan creation.
        from wanxiang.core.factory import AgentFactory
        rm.factory.instantiate_team = MagicMock(return_value={})
        rm.factory.llm_mode = None
        rm._resolve_factory_mode = AsyncMock(return_value=None)

        # Bypass engine: mock WorkflowEngine creation by erroring after plan
        # creation but before engine.run (the finally still runs _persist_run).
        run_id = await rm.start_run("hello", probe=True)
        # Wait for the background task to reach plan creation and fail.
        await asyncio.wait_for(rm._runs[run_id].task, timeout=5.0)

        rm.factory.create_team_probe.assert_called_once()
        rm.factory.create_team.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_run_uses_create_team(self, tmp_path):
        rm = RunManager()
        rm._history_path = tmp_path / "runs.jsonl"

        rm.factory.create_team = AsyncMock(
            return_value=TeamPlan(
                agents=[], workflow="pipeline", execution_order=[], rationale="normal"
            )
        )
        rm.factory.create_team_probe = AsyncMock()
        rm.factory.instantiate_team = MagicMock(return_value={})
        rm.factory.llm_mode = None
        rm._resolve_factory_mode = AsyncMock(return_value=None)

        run_id = await rm.start_run("hello")
        await asyncio.wait_for(rm._runs[run_id].task, timeout=5.0)

        rm.factory.create_team.assert_called_once()
        rm.factory.create_team_probe.assert_not_called()
