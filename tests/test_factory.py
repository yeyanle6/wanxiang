import asyncio

from wanxiang.core.factory import AgentFactory, SynthesisRequest, TeamPlan
from wanxiang.core.tools import ToolRegistry, ToolSpec


def _make_factory(tool_registry: ToolRegistry | None = None) -> AgentFactory:
    # No network calls in these tests; we only exercise policy logic.
    return AgentFactory(api_key="test-key", tool_registry=tool_registry)


def _spec(
    name: str,
    *,
    description: str = "",
    group: str = "",
    allowed_agents: list[str] | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {}},
        handler=lambda **_: None,
        group=group,
        allowed_agents=list(allowed_agents or []),
    )


def test_parallel_policy_adds_synthesizer_and_keeps_it_last() -> None:
    factory = _make_factory()
    plan = TeamPlan.from_dict(
        {
            "workflow": "parallel",
            "execution_order": ["researcher_a", "researcher_b"],
            "agents": [
                {"name": "researcher_a", "duty": "angle A", "base_identity": "You are researcher A."},
                {"name": "researcher_b", "duty": "angle B", "base_identity": "You are researcher B."},
            ],
        }
    )

    updated = factory._apply_planning_policies("compare two approaches", plan)
    names = [spec.name for spec in updated.agents]

    assert updated.workflow == "parallel"
    assert "synthesizer" in names
    assert updated.execution_order[-1] == "synthesizer"
    assert len([name for name in names if name != "synthesizer"]) >= 2


def test_parallel_policy_reorders_and_enforces_two_branches() -> None:
    factory = _make_factory()
    plan = TeamPlan.from_dict(
        {
            "workflow": "parallel",
            "execution_order": ["synthesizer", "researcher_a"],
            "agents": [
                {
                    "name": "synthesizer",
                    "duty": "merge branch results",
                    "base_identity": "You are a synthesizer.",
                },
                {"name": "researcher_a", "duty": "angle A", "base_identity": "You are researcher A."},
            ],
        }
    )

    updated = factory._apply_planning_policies("multi-source analysis", plan)
    names = [spec.name for spec in updated.agents]
    branches = [name for name in names if name != "synthesizer"]

    assert len(branches) >= 2
    assert updated.execution_order[-1] == "synthesizer"
    assert all(name in updated.execution_order for name in branches)


def test_content_task_policy_still_forces_review_loop() -> None:
    factory = _make_factory()
    plan = TeamPlan.from_dict(
        {
            "workflow": "parallel",
            "execution_order": ["researcher_a", "synthesizer"],
            "agents": [
                {"name": "researcher_a", "duty": "angle A", "base_identity": "You are researcher A."},
                {
                    "name": "synthesizer",
                    "duty": "merge branch results",
                    "base_identity": "You are a synthesizer.",
                },
            ],
        }
    )

    updated = factory._apply_planning_policies("写一篇关于多Agent系统的博客", plan)

    assert updated.workflow == "review_loop"
    assert len(updated.execution_order) == 2
    assert updated.execution_order[0] == "writer"
    assert updated.execution_order[1] == "reviewer"


def test_team_plan_parses_and_serializes_native_tools() -> None:
    plan = TeamPlan.from_dict(
        {
            "workflow": "parallel",
            "execution_order": ["researcher", "synthesizer"],
            "agents": [
                {
                    "name": "researcher",
                    "duty": "research latest facts",
                    "base_identity": "You are a researcher.",
                    "native_tools": [
                        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
                    ],
                },
                {
                    "name": "synthesizer",
                    "duty": "merge findings",
                    "base_identity": "You are a synthesizer.",
                },
            ],
        }
    )
    data = plan.to_dict()
    agent_map = {item["name"]: item for item in data["agents"]}

    assert agent_map["researcher"]["native_tools"][0]["name"] == "web_search"
    cfg = plan.agents[0].to_agent_config()
    assert cfg.native_tools[0]["type"] == "web_search_20250305"


def test_content_policy_strips_web_search_from_reviewer() -> None:
    factory = _make_factory()
    plan = TeamPlan.from_dict(
        {
            "workflow": "review_loop",
            "execution_order": ["writer", "reviewer"],
            "agents": [
                {
                    "name": "writer",
                    "duty": "draft content",
                    "base_identity": "You are a writer.",
                    "native_tools": [
                        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
                    ],
                },
                {
                    "name": "reviewer",
                    "duty": "quality review",
                    "base_identity": "You are a reviewer.",
                    "native_tools": [
                        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
                    ],
                },
            ],
        }
    )

    updated = factory._apply_planning_policies("写一篇行业报告", plan)
    reviewer = next(spec for spec in updated.agents if spec.name == "reviewer")
    writer = next(spec for spec in updated.agents if spec.name == "writer")

    assert writer.native_tools and writer.native_tools[0]["name"] == "web_search"
    assert reviewer.native_tools == []


def test_content_policy_forces_canonical_writer_when_candidate_is_synthesizer() -> None:
    factory = _make_factory()
    plan = TeamPlan.from_dict(
        {
            "workflow": "review_loop",
            "execution_order": ["synthesizer", "reviewer"],
            "agents": [
                {
                    "name": "synthesizer",
                    "duty": "Synthesize research and write final report",
                    "base_identity": "You are a synthesis expert and report writer.",
                    "native_tools": [
                        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
                    ],
                },
                {
                    "name": "reviewer",
                    "duty": "quality review",
                    "base_identity": "You are a reviewer.",
                },
            ],
        }
    )

    updated = factory._apply_planning_policies(
        "从多个角度研究2026年AI Agent和skill框架的最新发展趋势，汇总成一份报告",
        plan,
    )
    writer = next(spec for spec in updated.agents if spec.name == "writer")

    assert updated.workflow == "review_loop"
    assert updated.execution_order == ["writer", "reviewer"]
    assert writer.native_tools and writer.native_tools[0]["name"] == "web_search"


def test_instantiate_team_injects_team_context_and_effective_mode() -> None:
    factory = _make_factory()
    plan = TeamPlan.from_dict(
        {
            "workflow": "review_loop",
            "max_iterations": 3,
            "execution_order": ["writer", "reviewer"],
            "agents": [
                {
                    "name": "writer",
                    "duty": "draft content",
                    "base_identity": "You are a writer.",
                    "native_tools": [
                        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
                    ],
                },
                {
                    "name": "reviewer",
                    "duty": "quality review",
                    "base_identity": "You are a reviewer.",
                },
            ],
        }
    )

    team = factory.instantiate_team(plan, effective_mode="cli")
    reviewer_cfg = team["reviewer"].config
    writer_cfg = team["writer"].config

    assert reviewer_cfg.team_context.get("effective_mode") == "cli"
    peers = reviewer_cfg.team_context.get("agents")
    assert isinstance(peers, list) and len(peers) == 2
    writer_peer = next(p for p in peers if p["name"] == "writer")
    assert writer_peer["native_tools"][0]["name"] == "web_search"

    # Writer also gets the snapshot (so any agent can reason about peers).
    assert writer_cfg.team_context.get("effective_mode") == "cli"


def test_research_content_policy_skips_native_web_search_in_cli_mode() -> None:
    factory = _make_factory()
    plan = TeamPlan.from_dict(
        {
            "workflow": "review_loop",
            "max_iterations": 1,
            "execution_order": ["writer", "reviewer"],
            "agents": [
                {
                    "name": "writer",
                    "duty": "Drafts research report",
                    "base_identity": "You are a writer.",
                    "native_tools": [
                        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
                    ],
                },
                {
                    "name": "reviewer",
                    "duty": "quality review",
                    "base_identity": "You are a reviewer.",
                },
            ],
        }
    )

    updated = factory._apply_planning_policies(
        "从多个角度研究2026年AI Agent框架的最新发展趋势，汇总成一份报告",
        plan,
        effective_mode="cli",
    )
    writer = next(spec for spec in updated.agents if spec.name == "writer")
    reviewer = next(spec for spec in updated.agents if spec.name == "reviewer")

    # CLI mode does not support native tools — policy should have stripped web_search.
    assert writer.native_tools == []
    assert reviewer.native_tools == []
    assert updated.workflow == "review_loop"
    assert updated.max_iterations >= 3


def test_research_content_policy_assigns_web_search_and_raises_iterations() -> None:
    factory = _make_factory()
    plan = TeamPlan.from_dict(
        {
            "workflow": "review_loop",
            "max_iterations": 1,
            "execution_order": ["writer", "reviewer"],
            "agents": [
                {
                    "name": "writer",
                    "duty": "Drafts research report",
                    "base_identity": "You are a writer.",
                    "native_tools": [],
                },
                {
                    "name": "reviewer",
                    "duty": "quality review",
                    "base_identity": "You are a reviewer.",
                    "native_tools": [],
                },
            ],
        }
    )

    updated = factory._apply_planning_policies(
        "从多个角度研究2026年AI Agent和skill框架的最新发展趋势，汇总成一份报告",
        plan,
    )
    writer = next(spec for spec in updated.agents if spec.name == "writer")
    reviewer = next(spec for spec in updated.agents if spec.name == "reviewer")

    assert updated.workflow == "review_loop"
    assert updated.max_iterations >= 3
    assert writer.native_tools and writer.native_tools[0]["name"] == "web_search"
    assert reviewer.native_tools == []


def test_format_tools_for_planner_groups_by_source_with_descriptions() -> None:
    registry = ToolRegistry()
    registry.register(_spec("echo", description="Echo text back"))
    registry.register(_spec("current_time", description="Return UTC time"))
    registry.register(
        _spec(
            "read_text_file",
            description="Read contents of a text file",
            group="filesystem",
            allowed_agents=["researcher"],
        )
    )
    registry.register(
        _spec(
            "write_file",
            description="Write to a file",
            group="filesystem",
            allowed_agents=["researcher"],
        )
    )

    factory = _make_factory(tool_registry=registry)
    rendered = factory._format_tools_for_planner()

    # Builtin tools come first and carry no restriction marker.
    builtin_idx = rendered.index("builtin:")
    fs_idx = rendered.index("filesystem (MCP server):")
    assert builtin_idx < fs_idx

    assert "- echo: Echo text back" in rendered
    assert "- current_time: Return UTC time" in rendered
    assert "- read_text_file: Read contents of a text file (restricted to ['researcher'])" in rendered
    assert "- write_file: Write to a file (restricted to ['researcher'])" in rendered


def test_format_tools_for_planner_returns_none_marker_when_empty() -> None:
    registry = ToolRegistry()
    factory = _make_factory(tool_registry=registry)
    assert factory._format_tools_for_planner() == "(none)"


def test_apply_tool_restrictions_strips_disallowed_tool_from_agent() -> None:
    registry = ToolRegistry()
    registry.register(
        _spec(
            "read_text_file",
            description="Read a file",
            group="filesystem",
            allowed_agents=["researcher"],
        )
    )
    registry.register(_spec("echo", description="Echo"))  # open to all

    factory = _make_factory(tool_registry=registry)
    plan = TeamPlan.from_dict(
        {
            "workflow": "pipeline",
            "execution_order": ["writer", "reviewer"],
            "agents": [
                {
                    "name": "writer",
                    "duty": "draft",
                    "base_identity": "You are a writer.",
                    "allowed_tools": ["read_text_file", "echo"],
                },
                {
                    "name": "reviewer",
                    "duty": "review",
                    "base_identity": "You are a reviewer.",
                },
            ],
        }
    )

    updated = factory._apply_tool_restrictions(plan)
    writer = next(spec for spec in updated.agents if spec.name == "writer")

    # read_text_file is stripped (writer not in allowed_agents).
    # echo stays (no restriction).
    assert writer.allowed_tools == ["echo"]


def test_apply_tool_restrictions_keeps_tool_when_agent_is_authorized() -> None:
    registry = ToolRegistry()
    registry.register(
        _spec(
            "read_text_file",
            description="Read a file",
            group="filesystem",
            allowed_agents=["researcher", "reader"],
        )
    )

    factory = _make_factory(tool_registry=registry)
    plan = TeamPlan.from_dict(
        {
            "workflow": "pipeline",
            "execution_order": ["researcher", "reviewer"],
            "agents": [
                {
                    "name": "researcher",
                    "duty": "research",
                    "base_identity": "You are a researcher.",
                    "allowed_tools": ["read_text_file"],
                },
                {
                    "name": "reviewer",
                    "duty": "review",
                    "base_identity": "You are a reviewer.",
                },
            ],
        }
    )

    updated = factory._apply_tool_restrictions(plan)
    researcher = next(spec for spec in updated.agents if spec.name == "researcher")
    assert researcher.allowed_tools == ["read_text_file"]


def test_apply_tool_restrictions_runs_inside_policy_pipeline() -> None:
    """End-to-end: full _apply_planning_policies must also strip disallowed tools."""
    registry = ToolRegistry()
    registry.register(
        _spec(
            "read_text_file",
            description="Read a file",
            group="filesystem",
            allowed_agents=["researcher"],
        )
    )

    factory = _make_factory(tool_registry=registry)
    plan = TeamPlan.from_dict(
        {
            "workflow": "pipeline",
            "execution_order": ["analyst", "reviewer"],
            "agents": [
                {
                    "name": "analyst",
                    "duty": "analyze",
                    "base_identity": "You are an analyst.",
                    "allowed_tools": ["read_text_file"],
                },
                {
                    "name": "reviewer",
                    "duty": "review",
                    "base_identity": "You are a reviewer.",
                },
            ],
        }
    )

    updated = factory._apply_planning_policies("analyze something", plan)
    analyst = next(spec for spec in updated.agents if spec.name == "analyst")
    # Director named the agent 'analyst', but the tool is restricted to 'researcher'.
    # Policy should have stripped it.
    assert analyst.allowed_tools == []


# ---------------------------------------------------------------------------
# needs_synthesis parsing + Factory synthesis stage (Phase 4.2)
# ---------------------------------------------------------------------------


class _ForgeResult:
    def __init__(self, success: bool, tool_name: str | None, error: str | None = None):
        self.success = success
        self.registered = success
        self.error = error
        self.attempts: list = []
        self.tool_spec = None
        if success and tool_name:
            # Real ToolSpec so downstream code that reads .name works.
            self.tool_spec = ToolSpec(
                name=tool_name,
                description="synthesized stub",
                input_schema={"type": "object", "properties": {}, "required": []},
                handler=lambda: "stub",
            )


class _ScriptedSkillForge:
    """Duck-typed SkillForge stand-in — records calls, returns scripted results."""

    def __init__(self, results: list[_ForgeResult]) -> None:
        self._results = list(results)
        self.calls: list[str] = []

    async def forge(self, requirement: str):
        self.calls.append(requirement)
        if not self._results:
            return _ForgeResult(False, None, error="no more scripted results")
        return self._results.pop(0)


def test_team_plan_parses_needs_synthesis_with_suggested_names() -> None:
    plan = TeamPlan.from_dict(
        {
            "workflow": "pipeline",
            "execution_order": ["analyst", "reviewer"],
            "agents": [
                {
                    "name": "analyst",
                    "duty": "analyze numbers",
                    "base_identity": "You analyze things.",
                    "allowed_tools": ["chinese_to_arabic"],
                },
                {
                    "name": "reviewer",
                    "duty": "review",
                    "base_identity": "You review things.",
                },
            ],
            "needs_synthesis": [
                {
                    "requirement": "convert chinese numerals to arabic",
                    "suggested_name": "chinese_to_arabic",
                }
            ],
        }
    )
    assert len(plan.needs_synthesis) == 1
    request = plan.needs_synthesis[0]
    assert isinstance(request, SynthesisRequest)
    assert request.requirement == "convert chinese numerals to arabic"
    assert request.suggested_name == "chinese_to_arabic"

    # Round-trip through to_dict preserves the field.
    data = plan.to_dict()
    assert data["needs_synthesis"][0]["suggested_name"] == "chinese_to_arabic"


def test_team_plan_parses_empty_and_malformed_synthesis_entries() -> None:
    plan = TeamPlan.from_dict(
        {
            "workflow": "pipeline",
            "execution_order": ["a", "b"],
            "agents": [
                {"name": "a", "duty": "a", "base_identity": "a"},
                {"name": "b", "duty": "b", "base_identity": "b"},
            ],
            "needs_synthesis": [
                {"requirement": "", "suggested_name": "skip_me"},  # empty requirement → dropped
                {"suggested_name": "no_requirement"},  # missing requirement → dropped
                "not a dict",  # wrong type → dropped
                {"requirement": "valid"},  # no suggested_name is OK
            ],
        }
    )
    assert len(plan.needs_synthesis) == 1
    assert plan.needs_synthesis[0].requirement == "valid"
    assert plan.needs_synthesis[0].suggested_name is None


def test_synthesis_stage_attaches_new_tool_to_requesting_agent() -> None:
    forge = _ScriptedSkillForge([_ForgeResult(True, "chinese_to_arabic")])
    registry = ToolRegistry()
    factory = AgentFactory(
        api_key="test", tool_registry=registry, skill_forge=forge
    )

    plan = TeamPlan.from_dict(
        {
            "workflow": "pipeline",
            "execution_order": ["analyst", "reviewer"],
            "agents": [
                {
                    "name": "analyst",
                    "duty": "convert Chinese digits in a string",
                    "base_identity": "You analyze things.",
                    "allowed_tools": ["chinese_to_arabic"],
                },
                {
                    "name": "reviewer",
                    "duty": "review the output",
                    "base_identity": "You review things.",
                },
            ],
            "needs_synthesis": [
                {
                    "requirement": "convert chinese numerals to arabic",
                    "suggested_name": "chinese_to_arabic",
                }
            ],
        }
    )

    asyncio.run(factory._run_synthesis_stage(plan))

    # Forge was called once with the right requirement.
    assert forge.calls == ["convert chinese numerals to arabic"]

    # Analyst still has the tool (suggested_name == real registered name here).
    analyst = next(a for a in plan.agents if a.name == "analyst")
    assert "chinese_to_arabic" in analyst.allowed_tools

    # Factory logged the outcome.
    assert len(factory.synthesis_log) == 1
    entry = factory.synthesis_log[0]
    assert entry["success"] is True
    assert entry["registered"] is True
    assert entry["tool_name"] == "chinese_to_arabic"


def test_synthesis_stage_rewrites_placeholder_name_to_registered_name() -> None:
    # Director suggested "dict_it" but forge decided to register "dictionary_lookup".
    forge = _ScriptedSkillForge([_ForgeResult(True, "dictionary_lookup")])
    factory = AgentFactory(
        api_key="test", tool_registry=ToolRegistry(), skill_forge=forge
    )
    plan = TeamPlan.from_dict(
        {
            "workflow": "pipeline",
            "execution_order": ["worker", "reviewer"],
            "agents": [
                {
                    "name": "worker",
                    "duty": "look up things",
                    "base_identity": "identity",
                    "allowed_tools": ["dict_it"],
                },
                {"name": "reviewer", "duty": "r", "base_identity": "r"},
            ],
            "needs_synthesis": [
                {"requirement": "dictionary lookup", "suggested_name": "dict_it"}
            ],
        }
    )

    asyncio.run(factory._run_synthesis_stage(plan))

    worker = next(a for a in plan.agents if a.name == "worker")
    # Placeholder replaced with the real registered name.
    assert "dict_it" not in worker.allowed_tools
    assert "dictionary_lookup" in worker.allowed_tools


def test_synthesis_stage_handles_forge_failure_and_keeps_plan_runnable() -> None:
    forge = _ScriptedSkillForge([_ForgeResult(False, None, error="pytest failed")])
    factory = AgentFactory(
        api_key="test", tool_registry=ToolRegistry(), skill_forge=forge
    )
    plan = TeamPlan.from_dict(
        {
            "workflow": "pipeline",
            "execution_order": ["worker", "reviewer"],
            "agents": [
                {
                    "name": "worker",
                    "duty": "do something",
                    "base_identity": "identity",
                    "allowed_tools": ["maybe_tool"],
                },
                {"name": "reviewer", "duty": "r", "base_identity": "r"},
            ],
            "needs_synthesis": [
                {"requirement": "something hard", "suggested_name": "maybe_tool"}
            ],
        }
    )

    asyncio.run(factory._run_synthesis_stage(plan))

    worker = next(a for a in plan.agents if a.name == "worker")
    # Placeholder name stays in allowed_tools because nothing replaced it,
    # but the tool is not in the registry. BaseAgent's existing allowlist
    # filter will drop unknown tools at agent instantiation; the run still
    # proceeds without crashing.
    assert worker.allowed_tools == ["maybe_tool"]

    entry = factory.synthesis_log[0]
    assert entry["success"] is False
    assert "pytest failed" in (entry["error"] or "")


def test_synthesis_stage_without_forge_logs_warning_and_proceeds() -> None:
    factory = AgentFactory(
        api_key="test", tool_registry=ToolRegistry(), skill_forge=None
    )
    plan = TeamPlan.from_dict(
        {
            "workflow": "pipeline",
            "execution_order": ["worker", "reviewer"],
            "agents": [
                {"name": "worker", "duty": "work", "base_identity": "identity"},
                {"name": "reviewer", "duty": "r", "base_identity": "r"},
            ],
            "needs_synthesis": [
                {"requirement": "would be nice", "suggested_name": "nice_tool"},
            ],
        }
    )

    asyncio.run(factory._run_synthesis_stage(plan))

    # Single entry per request, marked failed with the "no SkillForge" reason.
    assert len(factory.synthesis_log) == 1
    entry = factory.synthesis_log[0]
    assert entry["success"] is False
    assert "SkillForge" in (entry["error"] or "")


def test_synthesis_stage_leaves_tool_in_registry_when_no_agent_requests_it() -> None:
    forge = _ScriptedSkillForge([_ForgeResult(True, "orphan_tool")])
    factory = AgentFactory(
        api_key="test", tool_registry=ToolRegistry(), skill_forge=forge
    )
    plan = TeamPlan.from_dict(
        {
            "workflow": "pipeline",
            "execution_order": ["worker", "reviewer"],
            "agents": [
                {"name": "worker", "duty": "w", "base_identity": "identity"},
                {"name": "reviewer", "duty": "r", "base_identity": "r"},
            ],
            "needs_synthesis": [
                {"requirement": "orphan capability"}  # no suggested_name
            ],
        }
    )

    asyncio.run(factory._run_synthesis_stage(plan))

    # Not attached to any agent, but logged as success.
    assert factory.synthesis_log[0]["success"] is True
    worker = next(a for a in plan.agents if a.name == "worker")
    assert "orphan_tool" not in (worker.allowed_tools or [])
