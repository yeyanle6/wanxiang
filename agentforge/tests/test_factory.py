from wanxiang.core.factory import AgentFactory, TeamPlan


def _make_factory() -> AgentFactory:
    # No network calls in these tests; we only exercise policy logic.
    return AgentFactory(api_key="test-key")


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
