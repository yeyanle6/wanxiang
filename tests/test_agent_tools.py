import asyncio
from typing import Any

from wanxiang.core.agent import AgentConfig, BaseAgent
from wanxiang.core.message import Message, MessageStatus
from wanxiang.core.tools import ToolRegistry, ToolSpec


class _ScriptedClient:
    def __init__(self, responses: list[dict[str, Any]], *, persona: str = "persona") -> None:
        self._responses = list(responses)
        self.persona = persona
        self.generate_calls = 0
        self.response_calls = 0
        self.last_response_messages: list[dict[str, Any]] | None = None
        self.last_response_tools: list[dict[str, Any]] | None = None

    async def generate(self, messages: list[dict[str, Any]], system: str | None = None) -> str:
        self.generate_calls += 1
        return self.persona

    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.response_calls += 1
        self.last_response_messages = list(messages)
        self.last_response_tools = list(tools or [])
        if not self._responses:
            return {"content": [{"type": "text", "text": "fallback"}]}
        return self._responses.pop(0)


class _ScriptedCliToolClient:
    def __init__(self, decisions: list[str], *, persona: str = "persona") -> None:
        self._decisions = list(decisions)
        self.persona = persona
        self.generate_calls = 0
        self.response_calls = 0

    async def resolve_mode(self, *, require_tools: bool = False) -> str:
        return "cli"

    async def generate(self, messages: list[dict[str, Any]], system: str | None = None) -> str:
        self.generate_calls += 1
        # First call is persona refinement; subsequent calls are tool-loop decisions.
        if self.generate_calls == 1:
            return self.persona
        if not self._decisions:
            return '{"action":"final","content":"fallback"}'
        return self._decisions.pop(0)

    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.response_calls += 1
        raise AssertionError("generate_response should not be used in CLI tool mode")


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="Echo back text",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=lambda text: f"Echo: {text}",
            timeout_s=1.0,
        )
    )
    return registry


def test_base_agent_tool_loop_success() -> None:
    client = _ScriptedClient(
        responses=[
            {
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"text": "hello"}}
                ]
            },
            {"content": [{"type": "text", "text": "Tool-assisted answer"}]},
        ]
    )
    config = AgentConfig(
        name="researcher",
        description="Researches with tools",
        base_identity="You are a researcher.",
        allowed_tools=["echo"],
        max_tool_rounds=3,
    )
    agent = BaseAgent(config=config, tool_registry=_build_registry())
    agent.client = client  # type: ignore[assignment]

    task = Message(intent="Find and summarize.", content="topic", sender="user")
    result = asyncio.run(agent.execute(task))

    assert result.status == MessageStatus.SUCCESS
    assert result.content == "Tool-assisted answer"
    assert client.generate_calls == 1
    assert client.response_calls == 2
    assert len(result.metadata.get("tool_calls", [])) == 1
    tool_call = result.metadata["tool_calls"][0]
    assert tool_call["tool"] == "echo"
    assert tool_call["success"] is True


def test_base_agent_tool_allowlist_blocks_unauthorized_tool() -> None:
    client = _ScriptedClient(
        responses=[
            {
                "content": [
                    {"type": "tool_use", "id": "toolu_x", "name": "secret_tool", "input": {"k": "v"}}
                ]
            },
            {"content": [{"type": "text", "text": "Cannot use forbidden tool; continue safely."}]},
        ]
    )
    config = AgentConfig(
        name="researcher",
        description="Researches with limited tools",
        base_identity="You are a researcher.",
        allowed_tools=["echo"],
        max_tool_rounds=2,
    )
    agent = BaseAgent(config=config, tool_registry=_build_registry())
    agent.client = client  # type: ignore[assignment]

    task = Message(intent="Use tool carefully.", content="topic", sender="user")
    result = asyncio.run(agent.execute(task))

    assert result.status == MessageStatus.SUCCESS
    assert "continue safely" in result.content
    assert len(result.metadata.get("tool_calls", [])) == 1
    tool_call = result.metadata["tool_calls"][0]
    assert tool_call["tool"] == "secret_tool"
    assert tool_call["success"] is False
    assert "not allowed" in (tool_call["error"] or "")


def test_base_agent_respects_max_tool_rounds_limit() -> None:
    client = _ScriptedClient(
        responses=[
            {"content": [{"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"text": "x"}}]},
            {"content": [{"type": "tool_use", "id": "toolu_2", "name": "echo", "input": {"text": "y"}}]},
        ]
    )
    config = AgentConfig(
        name="researcher",
        description="Researches with tools",
        base_identity="You are a researcher.",
        allowed_tools=["echo"],
        max_tool_rounds=1,
    )
    agent = BaseAgent(config=config, tool_registry=_build_registry())
    agent.client = client  # type: ignore[assignment]

    task = Message(intent="Loop tool calls", content="topic", sender="user")
    result = asyncio.run(agent.execute(task))

    assert result.status == MessageStatus.ERROR
    assert "Exceeded max_tool_rounds=1" in result.content


def test_base_agent_emits_tool_events() -> None:
    captured_events: list[dict[str, Any]] = []

    async def on_tool_event(event: dict[str, Any]) -> None:
        captured_events.append(event)

    client = _ScriptedClient(
        responses=[
            {"content": [{"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"text": "hi"}}]},
            {"content": [{"type": "text", "text": "done"}]},
        ]
    )
    config = AgentConfig(
        name="researcher",
        description="Researches with tools",
        base_identity="You are a researcher.",
        allowed_tools=["echo"],
        max_tool_rounds=2,
    )
    agent = BaseAgent(config=config, tool_registry=_build_registry(), on_tool_event=on_tool_event)
    agent.client = client  # type: ignore[assignment]

    task = Message(intent="Use tool", content="task", sender="user")
    result = asyncio.run(agent.execute(task))

    assert result.status == MessageStatus.SUCCESS
    assert [item.get("type") for item in captured_events] == ["tool_started", "tool_completed"]
    assert captured_events[0].get("tool") == "echo"
    assert captured_events[1].get("success") is True


def test_base_agent_cli_tool_mode_works_without_native_tool_blocks() -> None:
    client = _ScriptedCliToolClient(
        decisions=[
            '{"action":"tool","tool":"echo","arguments":{"text":"hello from cli"}}',
            '{"action":"final","content":"CLI final answer"}',
        ]
    )
    config = AgentConfig(
        name="researcher",
        description="Researches with tools",
        base_identity="You are a researcher.",
        allowed_tools=["echo"],
        max_tool_rounds=3,
    )
    agent = BaseAgent(config=config, tool_registry=_build_registry())
    agent.client = client  # type: ignore[assignment]

    task = Message(intent="Use tool via CLI mode", content="topic", sender="user")
    result = asyncio.run(agent.execute(task))

    assert result.status == MessageStatus.SUCCESS
    assert result.content == "CLI final answer"
    assert result.metadata.get("tool_mode") == "cli"
    assert len(result.metadata.get("tool_calls", [])) == 1
    assert result.metadata["tool_calls"][0]["tool"] == "echo"
    assert client.response_calls == 0


def test_base_agent_api_native_web_search_tool_does_not_use_local_registry_execution() -> None:
    client = _ScriptedClient(
        responses=[
            {
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtool_1",
                        "name": "web_search",
                        "input": {"query": "latest multi-agent adoption trends"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtool_1",
                        "content": [
                            {"type": "web_search_result", "title": "Result A", "url": "https://example.com/a"}
                        ],
                    },
                    {"type": "text", "text": "Native web search answer"},
                ],
            }
        ]
    )
    config = AgentConfig(
        name="researcher",
        description="Researches using native web search",
        base_identity="You are a researcher.",
        native_tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }
        ],
        max_tool_rounds=3,
    )
    agent = BaseAgent(config=config, tool_registry=_build_registry())
    agent.client = client  # type: ignore[assignment]

    task = Message(intent="Research latest info", content="topic", sender="user")
    result = asyncio.run(agent.execute(task))

    assert result.status == MessageStatus.SUCCESS
    assert result.content == "Native web search answer"
    assert result.metadata.get("tool_mode") == "api"
    assert result.metadata.get("tool_calls", []) == []
    assert client.response_calls == 1
    sent_tools = client.last_response_tools or []
    assert any(item.get("name") == "web_search" for item in sent_tools)
    assert any(item.get("type") == "web_search_20250305" for item in sent_tools)


def test_base_agent_api_native_web_search_emits_tool_events() -> None:
    captured_events: list[dict[str, Any]] = []

    async def on_tool_event(event: dict[str, Any]) -> None:
        captured_events.append(event)

    client = _ScriptedClient(
        responses=[
            {
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtool_2",
                        "name": "web_search",
                        "input": {"query": "agent orchestration trends"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtool_2",
                        "content": [
                            {
                                "type": "web_search_result",
                                "title": "Trend Report",
                                "url": "https://example.com/trend",
                                "text": "Key points",
                            }
                        ],
                    },
                    {"type": "text", "text": "Final answer with current info"},
                ],
            }
        ]
    )
    config = AgentConfig(
        name="researcher",
        description="Researches with native web search",
        base_identity="You are a researcher.",
        native_tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        max_tool_rounds=3,
    )
    agent = BaseAgent(config=config, tool_registry=_build_registry(), on_tool_event=on_tool_event)
    agent.client = client  # type: ignore[assignment]

    task = Message(intent="Research with native tool", content="topic", sender="user")
    result = asyncio.run(agent.execute(task))

    assert result.status == MessageStatus.SUCCESS
    assert [item.get("type") for item in captured_events] == ["tool_started", "tool_completed"]
    assert captured_events[0].get("tool") == "web_search"
    assert captured_events[1].get("tool") == "web_search"
    assert captured_events[1].get("success") is True


class _ScriptedCliGracefulDegradeClient:
    def __init__(self, *, persona: str = "persona", final_text: str = "fallback draft") -> None:
        self.persona = persona
        self.final_text = final_text
        self.generate_calls = 0
        self.response_calls = 0

    async def resolve_mode(self, *, require_tools: bool = False) -> str:
        return "cli"

    async def generate(self, messages: list[dict[str, Any]], system: str | None = None) -> str:
        self.generate_calls += 1
        if self.generate_calls == 1:
            return self.persona
        return self.final_text

    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.response_calls += 1
        raise AssertionError("generate_response must not be called when native tools are stripped in CLI mode")


def test_base_agent_cli_mode_gracefully_strips_native_tools() -> None:
    client = _ScriptedCliGracefulDegradeClient(final_text="Fallback draft without web_search")
    config = AgentConfig(
        name="writer",
        description="Drafts content",
        base_identity="You are a writer.",
        native_tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        max_tool_rounds=2,
    )
    agent = BaseAgent(config=config, tool_registry=_build_registry())
    agent.client = client  # type: ignore[assignment]

    task = Message(intent="Write a report", content="topic", sender="user")
    result = asyncio.run(agent.execute(task))

    assert result.status == MessageStatus.SUCCESS
    assert result.content == "Fallback draft without web_search"
    assert result.metadata.get("tool_calls", []) == []
    # native_tools were stripped; no tool loop was entered.
    assert client.response_calls == 0


def test_reviewer_prompt_relaxes_citations_when_writer_lacks_web_search() -> None:
    config = AgentConfig(
        name="reviewer",
        description="Quality gate reviewer",
        base_identity="You are a reviewer agent.",
        team_context={
            "effective_mode": "cli",
            "agents": [
                {
                    "name": "writer",
                    "duty": "draft the report",
                    "base_identity": "You are a writer.",
                    "allowed_tools": [],
                    "native_tools": [
                        {"name": "web_search", "type": "web_search_20250305"}
                    ],
                },
                {
                    "name": "reviewer",
                    "duty": "quality review",
                    "base_identity": "You are a reviewer.",
                    "allowed_tools": [],
                    "native_tools": [],
                },
            ],
        },
    )
    agent = BaseAgent(config=config, tool_registry=_build_registry())
    message = Message(intent="review draft", content="draft body", sender="writer")

    prompt = agent.build_prompt(message, persona="strict")
    body = str(prompt[0].get("content", ""))

    assert "Team tool availability:" in body
    assert "native_web_search=no" in body
    assert "Do NOT require specific" in body
    # Must not demand citations in this configuration.
    assert "REQUIRE at least one cited source" not in body


def test_reviewer_prompt_requires_citations_when_writer_has_web_search_in_api_mode() -> None:
    config = AgentConfig(
        name="reviewer",
        description="Quality gate reviewer",
        base_identity="You are a reviewer agent.",
        team_context={
            "effective_mode": "api",
            "agents": [
                {
                    "name": "writer",
                    "duty": "draft the report",
                    "base_identity": "You are a writer.",
                    "allowed_tools": [],
                    "native_tools": [
                        {"name": "web_search", "type": "web_search_20250305"}
                    ],
                },
                {
                    "name": "reviewer",
                    "duty": "quality review",
                    "base_identity": "You are a reviewer.",
                    "allowed_tools": [],
                    "native_tools": [],
                },
            ],
        },
    )
    agent = BaseAgent(config=config, tool_registry=_build_registry())
    message = Message(intent="review draft", content="draft body", sender="writer")

    prompt = agent.build_prompt(message, persona="strict")
    body = str(prompt[0].get("content", ""))

    assert "native_web_search=yes" in body
    assert "REQUIRE at least one cited source" in body


def test_reviewer_prompt_without_team_context_keeps_legacy_rules() -> None:
    config = AgentConfig(
        name="reviewer",
        description="Quality gate reviewer",
        base_identity="You are a reviewer agent.",
    )
    agent = BaseAgent(config=config, tool_registry=_build_registry())
    message = Message(intent="review draft", content="draft body", sender="writer")

    prompt = agent.build_prompt(message, persona="strict")
    body = str(prompt[0].get("content", ""))

    # No tool-aware block injected when team_context is empty.
    assert "Team tool availability:" not in body
    # Legacy reviewer instructions remain.
    assert "You are the quality gate reviewer." in body


def test_writer_prompt_takes_precedence_over_reviewer_keywords_in_duty() -> None:
    config = AgentConfig(
        name="writer",
        description="Drafts content and incorporates review feedback",
        base_identity="You are a writer agent for long-form reports.",
    )
    agent = BaseAgent(config=config, tool_registry=_build_registry())
    message = Message(intent="Write a report", content="task", sender="user")

    prompt = agent.build_prompt(message, persona="concise")
    assert isinstance(prompt, list) and len(prompt) == 1
    body = str(prompt[0].get("content", ""))
    assert "You are the writer." in body
    assert "You are the quality gate reviewer." not in body
