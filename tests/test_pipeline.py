import asyncio

from wanxiang.core.factory import TeamPlan
from wanxiang.core.message import Message, MessageStatus
from wanxiang.core.pipeline import WorkflowEngine


class MockAgent:
    def __init__(self, name: str, scripted_outputs: list[tuple[MessageStatus, str]]) -> None:
        self.name = name
        self.scripted_outputs = scripted_outputs
        self.calls: list[Message] = []

    async def execute(self, message: Message) -> Message:
        self.calls.append(message)
        index = min(len(self.calls) - 1, len(self.scripted_outputs) - 1)
        status, content = self.scripted_outputs[index]
        return message.create_reply(
            intent=f"{self.name} response",
            content=content,
            sender=self.name,
            status=status,
            metadata={"mock_call_index": index + 1},
        )


class DelayedMockAgent(MockAgent):
    def __init__(
        self,
        name: str,
        scripted_outputs: list[tuple[MessageStatus, str]],
        *,
        delay_s: float = 0.0,
    ) -> None:
        super().__init__(name=name, scripted_outputs=scripted_outputs)
        self.delay_s = delay_s

    async def execute(self, message: Message) -> Message:
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        return await super().execute(message)


def _build_plan(workflow: str, order: list[str], max_iterations: int = 1) -> TeamPlan:
    return TeamPlan.from_dict(
        {
            "workflow": workflow,
            "execution_order": order,
            "max_iterations": max_iterations,
            "agents": [
                {
                    "name": name,
                    "duty": f"{name} duty",
                    "base_identity": f"You are {name}.",
                }
                for name in order
            ],
        }
    )


def test_pipeline_mode_executes_in_order_and_collects_trace() -> None:
    agents = {
        "writer": MockAgent("writer", [(MessageStatus.SUCCESS, "draft")]),
        "reviewer": MockAgent("reviewer", [(MessageStatus.SUCCESS, "reviewed")]),
        "publisher": MockAgent("publisher", [(MessageStatus.SUCCESS, "published")]),
    }
    plan = _build_plan("pipeline", ["writer", "reviewer", "publisher"])
    engine = WorkflowEngine(agents=agents, plan=plan)

    task = Message(intent="Write and publish article.", content="Task", sender="user")
    trace = asyncio.run(engine.run(task))

    assert len(trace) == 3
    assert [m.sender for m in trace] == ["writer", "reviewer", "publisher"]
    assert [m.content for m in trace] == ["draft", "reviewed", "published"]
    assert len(agents["writer"].calls) == 1
    assert len(agents["reviewer"].calls) == 1
    assert len(agents["publisher"].calls) == 1


def test_review_loop_retries_then_succeeds() -> None:
    producer = MockAgent(
        "writer",
        [
            (MessageStatus.SUCCESS, "draft v1"),
            (MessageStatus.SUCCESS, "draft v2"),
        ],
    )
    reviewer = MockAgent(
        "reviewer",
        [
            (MessageStatus.NEEDS_REVISION, "Need clearer structure."),
            (MessageStatus.SUCCESS, "Approved."),
        ],
    )
    agents = {"writer": producer, "reviewer": reviewer}
    plan = _build_plan("review_loop", ["writer", "reviewer"], max_iterations=3)
    engine = WorkflowEngine(agents=agents, plan=plan)

    task = Message(intent="Create final article draft.", content="Task", sender="user")
    trace = asyncio.run(engine.run(task))

    assert len(trace) == 4
    assert [m.sender for m in trace] == ["writer", "reviewer", "writer", "reviewer"]
    assert trace[-1].status == MessageStatus.SUCCESS
    assert len(producer.calls) == 2
    assert len(reviewer.calls) == 2
    # Producer's second input should include previous produced content via context accumulation.
    assert producer.calls[1].content == "Need clearer structure."
    assert "draft v1" in producer.calls[1].context


def test_review_loop_stops_at_max_iterations() -> None:
    producer = MockAgent(
        "writer",
        [
            (MessageStatus.SUCCESS, "draft v1"),
            (MessageStatus.SUCCESS, "draft v2"),
            (MessageStatus.SUCCESS, "draft v3"),
        ],
    )
    reviewer = MockAgent(
        "reviewer",
        [
            (MessageStatus.NEEDS_REVISION, "Revise 1"),
            (MessageStatus.NEEDS_REVISION, "Revise 2"),
            (MessageStatus.NEEDS_REVISION, "Revise 3"),
        ],
    )
    agents = {"writer": producer, "reviewer": reviewer}
    plan = _build_plan("review_loop", ["writer", "reviewer"], max_iterations=2)
    engine = WorkflowEngine(agents=agents, plan=plan)

    task = Message(intent="Draft with strict quality gate.", content="Task", sender="user")
    trace = asyncio.run(engine.run(task))

    assert len(trace) == 4
    assert trace[-1].sender == "reviewer"
    assert trace[-1].status == MessageStatus.NEEDS_REVISION
    assert len(producer.calls) == 2
    assert len(reviewer.calls) == 2


def test_pipeline_stops_on_error_and_skips_remaining_agents() -> None:
    agents = {
        "writer": MockAgent("writer", [(MessageStatus.SUCCESS, "draft")]),
        "reviewer": MockAgent("reviewer", [(MessageStatus.ERROR, "LLM timeout")]),
        "publisher": MockAgent("publisher", [(MessageStatus.SUCCESS, "published")]),
    }
    plan = _build_plan("pipeline", ["writer", "reviewer", "publisher"])
    engine = WorkflowEngine(agents=agents, plan=plan)

    task = Message(intent="Publish article.", content="Task", sender="user")
    trace = asyncio.run(engine.run(task))

    assert len(trace) == 2
    assert trace[-1].status == MessageStatus.ERROR
    assert [m.sender for m in trace] == ["writer", "reviewer"]
    assert len(agents["publisher"].calls) == 0


def test_parallel_mode_runs_branches_then_synthesizer() -> None:
    agents = {
        "researcher_a": MockAgent("researcher_a", [(MessageStatus.SUCCESS, "A perspective")]),
        "researcher_b": MockAgent("researcher_b", [(MessageStatus.SUCCESS, "B perspective")]),
        "researcher_c": MockAgent("researcher_c", [(MessageStatus.SUCCESS, "C perspective")]),
        "synthesizer": MockAgent("synthesizer", [(MessageStatus.SUCCESS, "Merged report")]),
    }
    plan = _build_plan(
        "parallel",
        ["researcher_a", "researcher_b", "researcher_c", "synthesizer"],
    )
    engine = WorkflowEngine(agents=agents, plan=plan)

    task = Message(intent="Research topic from multiple angles.", content="Task", sender="user")
    trace = asyncio.run(engine.run(task))

    assert len(trace) == 4
    assert trace[-1].sender == "synthesizer"
    assert trace[-1].status == MessageStatus.SUCCESS
    assert len(agents["synthesizer"].calls) == 1
    merged_input = agents["synthesizer"].calls[0]
    assert "A perspective" in merged_input.content
    assert "B perspective" in merged_input.content
    assert "C perspective" in merged_input.content


def test_parallel_mode_is_fail_tolerant_for_partial_failures() -> None:
    agents = {
        "researcher_a": MockAgent("researcher_a", [(MessageStatus.SUCCESS, "A perspective")]),
        "researcher_b": MockAgent("researcher_b", [(MessageStatus.ERROR, "source timeout")]),
        "researcher_c": MockAgent("researcher_c", [(MessageStatus.SUCCESS, "C perspective")]),
        "synthesizer": MockAgent("synthesizer", [(MessageStatus.SUCCESS, "Merged report")]),
    }
    plan = _build_plan(
        "parallel",
        ["researcher_a", "researcher_b", "researcher_c", "synthesizer"],
    )
    engine = WorkflowEngine(agents=agents, plan=plan)

    task = Message(intent="Research topic from multiple angles.", content="Task", sender="user")
    trace = asyncio.run(engine.run(task))

    assert len(trace) == 4
    assert trace[-1].sender == "synthesizer"
    assert trace[-1].status == MessageStatus.SUCCESS
    merged_input = agents["synthesizer"].calls[0]
    assert "A perspective" in merged_input.content
    assert "C perspective" in merged_input.content
    assert "source timeout" not in merged_input.content
    assert "researcher_b" in merged_input.content


def test_parallel_mode_returns_error_when_all_branches_fail() -> None:
    agents = {
        "researcher_a": MockAgent("researcher_a", [(MessageStatus.ERROR, "A failed")]),
        "researcher_b": MockAgent("researcher_b", [(MessageStatus.ERROR, "B failed")]),
        "synthesizer": MockAgent("synthesizer", [(MessageStatus.SUCCESS, "Merged report")]),
    }
    plan = _build_plan("parallel", ["researcher_a", "researcher_b", "synthesizer"])
    engine = WorkflowEngine(agents=agents, plan=plan)

    task = Message(intent="Research topic from multiple angles.", content="Task", sender="user")
    trace = asyncio.run(engine.run(task))

    assert len(trace) == 3
    assert trace[-1].sender == "parallel_stage"
    assert trace[-1].status == MessageStatus.ERROR
    assert len(agents["synthesizer"].calls) == 0


def test_parallel_events_emit_all_starts_before_parallel_completions() -> None:
    captured_events: list[dict] = []

    async def on_event(event: dict) -> None:
        captured_events.append(event)

    agents = {
        "researcher_a": DelayedMockAgent(
            "researcher_a", [(MessageStatus.SUCCESS, "A perspective")], delay_s=0.03
        ),
        "researcher_b": DelayedMockAgent(
            "researcher_b", [(MessageStatus.SUCCESS, "B perspective")], delay_s=0.01
        ),
        "researcher_c": DelayedMockAgent(
            "researcher_c", [(MessageStatus.SUCCESS, "C perspective")], delay_s=0.02
        ),
        "synthesizer": MockAgent("synthesizer", [(MessageStatus.SUCCESS, "Merged report")]),
    }
    parallel_names = {"researcher_a", "researcher_b", "researcher_c"}
    plan = _build_plan(
        "parallel",
        ["researcher_a", "researcher_b", "researcher_c", "synthesizer"],
    )
    engine = WorkflowEngine(agents=agents, plan=plan, on_event=on_event)

    task = Message(intent="Research topic from multiple angles.", content="Task", sender="user")
    asyncio.run(engine.run(task))

    start_indices = [
        i
        for i, event in enumerate(captured_events)
        if event.get("type") == "agent_started" and event.get("agent") in parallel_names
    ]
    completed_indices = [
        i
        for i, event in enumerate(captured_events)
        if event.get("type") == "agent_completed" and event.get("agent") in parallel_names
    ]
    assert start_indices
    assert completed_indices
    assert max(start_indices) < min(completed_indices)

    parallel_completed_index = next(
        i for i, event in enumerate(captured_events) if event.get("type") == "parallel_completed"
    )
    synth_started_index = next(
        i
        for i, event in enumerate(captured_events)
        if event.get("type") == "agent_started" and event.get("agent") == "synthesizer"
    )
    assert parallel_completed_index < synth_started_index


# ---- parallel stagger -----------------------------------------------------


def test_parallel_stagger_defaults_to_8s():
    """WorkflowEngine exposes the parallel_stagger_s config with a safe default."""
    plan = _build_plan("parallel", ["r_a", "r_b", "synth"])
    agents = {n: MockAgent(n, [(MessageStatus.SUCCESS, n)]) for n in ["r_a", "r_b", "synth"]}
    engine = WorkflowEngine(agents=agents, plan=plan)
    assert engine.parallel_stagger_s == 8.0


def test_parallel_stagger_custom_value_respected():
    plan = _build_plan("parallel", ["r_a", "r_b", "synth"])
    agents = {n: MockAgent(n, [(MessageStatus.SUCCESS, n)]) for n in ["r_a", "r_b", "synth"]}
    engine = WorkflowEngine(agents=agents, plan=plan, parallel_stagger_s=1.5)
    assert engine.parallel_stagger_s == 1.5


def test_parallel_stagger_negative_clamped_to_zero():
    plan = _build_plan("parallel", ["r_a", "r_b", "synth"])
    agents = {n: MockAgent(n, [(MessageStatus.SUCCESS, n)]) for n in ["r_a", "r_b", "synth"]}
    engine = WorkflowEngine(agents=agents, plan=plan, parallel_stagger_s=-5.0)
    assert engine.parallel_stagger_s == 0.0


def test_parallel_stagger_delays_branch_launch():
    """With stagger > 0, the second branch starts measurably later than the first."""
    parallel_names = ["r_a", "r_b"]
    agents: dict[str, MockAgent] = {
        name: MockAgent(name, [(MessageStatus.SUCCESS, f"{name} output")])
        for name in parallel_names + ["synth"]
    }

    launch_times: dict[str, float] = {}

    class LaunchRecorder(MockAgent):
        async def execute(self, message: Message) -> Message:
            loop = asyncio.get_event_loop()
            launch_times[self.name] = loop.time()
            return await super().execute(message)

    for name in parallel_names:
        agents[name] = LaunchRecorder(name, [(MessageStatus.SUCCESS, f"{name} output")])

    plan = _build_plan("parallel", parallel_names + ["synth"])
    engine = WorkflowEngine(agents=agents, plan=plan, parallel_stagger_s=0.25)

    task = Message(intent="Research", content="Task", sender="user")
    asyncio.run(engine.run(task))

    assert "r_a" in launch_times and "r_b" in launch_times
    delta = launch_times["r_b"] - launch_times["r_a"]
    # Allow some scheduling slack but require the stagger to actually delay
    # the second branch by most of the configured amount.
    assert delta >= 0.20, f"expected ≥0.20s stagger, got {delta:.3f}s"


def test_parallel_stagger_zero_launches_all_simultaneously():
    parallel_names = ["r_a", "r_b"]
    agents: dict[str, MockAgent] = {}
    launch_times: dict[str, float] = {}

    class LaunchRecorder(MockAgent):
        async def execute(self, message: Message) -> Message:
            loop = asyncio.get_event_loop()
            launch_times[self.name] = loop.time()
            return await super().execute(message)

    for name in parallel_names:
        agents[name] = LaunchRecorder(name, [(MessageStatus.SUCCESS, f"{name} output")])
    agents["synth"] = MockAgent("synth", [(MessageStatus.SUCCESS, "synth")])

    plan = _build_plan("parallel", parallel_names + ["synth"])
    engine = WorkflowEngine(agents=agents, plan=plan, parallel_stagger_s=0.0)

    task = Message(intent="Research", content="Task", sender="user")
    asyncio.run(engine.run(task))

    delta = abs(launch_times["r_b"] - launch_times["r_a"])
    # Without stagger, both branches should kick off essentially together.
    assert delta < 0.10, f"expected near-simultaneous launch, got {delta:.3f}s"
