"""Unit tests for SkillForge with a scripted synthesizer (no LLM calls).

The real LLM integration test for SynthesizerAgent + SkillForge lands
in a separate @pytest.mark.integration file so CI stays hermetic.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from wanxiang.core.message import Message, MessageStatus
from wanxiang.core.sandbox import SandboxExecutor
from wanxiang.core.skill_forge import (
    SkillForge,
    parse_synthesizer_response,
)
from wanxiang.core.tools import ToolRegistry


class ScriptedSynthesizer:
    """Duck-typed stand-in for a BaseAgent that returns scripted responses."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[Message] = []

    async def execute(self, message: Message) -> Message:
        self.calls.append(message)
        reply_text = self._replies.pop(0) if self._replies else "{}"
        return message.create_reply(
            intent=message.intent,
            content=reply_text,
            sender="skill_synthesizer",
            status=MessageStatus.SUCCESS,
            metadata={"scripted": True},
        )


def _spec_json(
    tool_name: str,
    handler_code: str,
    test_code: str,
    *,
    description: str = "A synthesized tool.",
    schema: dict | None = None,
) -> str:
    return json.dumps(
        {
            "tool_name": tool_name,
            "description": description,
            "input_schema": schema
            or {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            "handler_code": handler_code,
            "test_code": test_code,
        }
    )


_GOOD_HANDLER = (
    "def handler(value: str) -> str:\n"
    "    return f'echo:{value}'\n"
)
_GOOD_TEST = (
    "from handler import handler\n"
    "\n"
    "def test_echo():\n"
    "    assert handler('hi') == 'echo:hi'\n"
)

_BAD_HANDLER = (
    "def handler(value: str) -> str:\n"
    "    return 'wrong'\n"  # does not satisfy the test
)

_INFINITE_HANDLER = (
    "def handler(value: str) -> str:\n"
    "    while True:\n"
    "        pass\n"
)
_INFINITE_TEST = (
    "from handler import handler\n"
    "\n"
    "def test_loop():\n"
    "    assert handler('x') == 'done'\n"
)


def _make_forge(synthesizer, *, max_retries: int = 3) -> tuple[SkillForge, ToolRegistry]:
    registry = ToolRegistry()
    sandbox = SandboxExecutor(timeout_s=15.0)
    forge = SkillForge(
        sandbox=sandbox,
        registry=registry,
        synthesizer=synthesizer,
        max_retries=max_retries,
    )
    return forge, registry


# ---------------------------------------------------------------------------
# parse_synthesizer_response
# ---------------------------------------------------------------------------


def test_parse_raw_json_succeeds() -> None:
    raw = _spec_json("t", _GOOD_HANDLER, _GOOD_TEST)
    data, err = parse_synthesizer_response(raw)
    assert err is None and data is not None
    assert data["tool_name"] == "t"


def test_parse_json_inside_markdown_fence_succeeds() -> None:
    raw = "```json\n" + _spec_json("t", _GOOD_HANDLER, _GOOD_TEST) + "\n```"
    data, err = parse_synthesizer_response(raw)
    assert err is None and data is not None
    assert data["tool_name"] == "t"


def test_parse_missing_required_keys_fails() -> None:
    raw = json.dumps({"tool_name": "t", "description": "x"})
    data, err = parse_synthesizer_response(raw)
    assert data is None
    assert "missing required fields" in (err or "")


def test_parse_non_json_text_fails() -> None:
    raw = "sure, here's your tool: `read_file`"
    data, err = parse_synthesizer_response(raw)
    assert data is None
    assert "not valid JSON" in (err or "")


def test_parse_empty_string_fails() -> None:
    data, err = parse_synthesizer_response("")
    assert data is None and err is not None


def test_parse_invalid_field_types_fails() -> None:
    raw = json.dumps(
        {
            "tool_name": "t",
            "description": "d",
            "input_schema": "not-an-object",  # invalid
            "handler_code": "pass",
            "test_code": "pass",
        }
    )
    data, err = parse_synthesizer_response(raw)
    assert data is None
    assert "input_schema" in (err or "")


# ---------------------------------------------------------------------------
# SkillForge — mocked LLM scenarios
# ---------------------------------------------------------------------------


def test_forge_succeeds_on_first_attempt() -> None:
    synth = ScriptedSynthesizer([_spec_json("echo_tool", _GOOD_HANDLER, _GOOD_TEST)])
    forge, registry = _make_forge(synth)

    result = asyncio.run(forge.forge("An echo tool that prefixes 'echo:'"))

    assert result.success is True
    assert result.registered is True
    assert result.tool_spec is not None
    assert result.tool_spec.name == "echo_tool"
    assert result.tool_spec.group == "synthesized"
    assert len(result.attempts) == 1
    assert registry.get("echo_tool") is not None

    # Registered tool is callable and returns the handler's output.
    call_result = asyncio.run(registry.execute("echo_tool", {"value": "hi"}))
    assert call_result.success is True
    assert call_result.content == "echo:hi"


def test_forge_recovers_from_failing_test_on_retry() -> None:
    # First attempt: handler returns 'wrong' and test expects 'echo:hi' → pytest fails.
    # Second attempt: handler fixed.
    synth = ScriptedSynthesizer(
        [
            _spec_json("echo_tool", _BAD_HANDLER, _GOOD_TEST),
            _spec_json("echo_tool", _GOOD_HANDLER, _GOOD_TEST),
        ]
    )
    forge, registry = _make_forge(synth)

    result = asyncio.run(forge.forge("Echo tool"))

    assert result.success is True
    assert result.registered is True
    assert len(result.attempts) == 2

    first, second = result.attempts
    assert first.sandbox_result is not None
    assert first.sandbox_result.passed is False
    assert first.error is not None
    assert second.sandbox_result is not None
    assert second.sandbox_result.passed is True

    # Synthesizer was invoked twice, and the second call's prompt included
    # feedback from the first failure.
    assert len(synth.calls) == 2
    second_prompt = synth.calls[1].content
    assert "Feedback on your previous attempt" in second_prompt
    # Pytest failure output should appear somewhere in the feedback.
    assert "test_echo" in second_prompt or "stdout" in second_prompt.lower()


def test_forge_exhausts_retries_when_tests_keep_failing() -> None:
    synth = ScriptedSynthesizer(
        [
            _spec_json("echo_tool", _BAD_HANDLER, _GOOD_TEST),
            _spec_json("echo_tool", _BAD_HANDLER, _GOOD_TEST),
        ]
    )
    forge, registry = _make_forge(synth, max_retries=2)

    result = asyncio.run(forge.forge("Echo tool"))

    assert result.success is False
    assert result.registered is False
    assert result.tool_spec is None
    assert len(result.attempts) == 2
    assert result.error is not None
    assert "Exceeded max_retries" in result.error
    assert registry.get("echo_tool") is None


def test_forge_reports_parse_failure_and_retries() -> None:
    synth = ScriptedSynthesizer(
        [
            "sure here's your tool!",  # not JSON
            _spec_json("echo_tool", _GOOD_HANDLER, _GOOD_TEST),
        ]
    )
    forge, registry = _make_forge(synth)

    result = asyncio.run(forge.forge("Echo tool"))

    assert result.success is True
    first, second = result.attempts
    assert first.sandbox_result is None  # never reached the sandbox
    assert first.error is not None
    assert "not valid JSON" in first.error
    assert second.sandbox_result is not None and second.sandbox_result.passed is True

    # Feedback on the second prompt should mention the parse issue.
    second_prompt = synth.calls[1].content
    assert "could not be parsed" in second_prompt.lower() or "not valid json" in second_prompt.lower()


def test_forge_handles_sandbox_timeout_and_feeds_back() -> None:
    synth = ScriptedSynthesizer(
        [
            _spec_json("loop_tool", _INFINITE_HANDLER, _INFINITE_TEST),
            _spec_json("loop_tool", _GOOD_HANDLER, _GOOD_TEST),
        ]
    )
    # Use a tight sandbox timeout for the forge's internal sandbox.
    registry = ToolRegistry()
    forge = SkillForge(
        sandbox=SandboxExecutor(timeout_s=0.5),
        registry=registry,
        synthesizer=synth,
        max_retries=3,
    )

    result = asyncio.run(forge.forge("Something"))

    assert result.success is True
    assert len(result.attempts) == 2
    first = result.attempts[0]
    assert first.sandbox_result is not None
    assert first.sandbox_result.timed_out is True
    second_prompt = synth.calls[1].content
    assert "timed out" in second_prompt.lower()


def test_forge_rejects_name_collision_with_existing_tool() -> None:
    synth = ScriptedSynthesizer(
        [
            _spec_json("already_there", _GOOD_HANDLER, _GOOD_TEST),
            _spec_json("fresh_name", _GOOD_HANDLER, _GOOD_TEST),
        ]
    )
    registry = ToolRegistry()
    # Pre-register a tool with the colliding name.
    from wanxiang.core.tools import ToolSpec

    registry.register(
        ToolSpec(
            name="already_there",
            description="Pre-existing",
            input_schema={"type": "object"},
            handler=lambda: "nope",
            timeout_s=1.0,
        )
    )
    forge = SkillForge(
        sandbox=SandboxExecutor(timeout_s=15.0),
        registry=registry,
        synthesizer=synth,
        max_retries=3,
    )

    result = asyncio.run(forge.forge("something"))

    assert result.success is True
    assert result.registered is True
    assert result.tool_spec is not None and result.tool_spec.name == "fresh_name"
    # First attempt rejected before sandbox ran.
    first = result.attempts[0]
    assert first.error is not None and "already registered" in first.error
    assert first.sandbox_result is None
    # Feedback steered the synthesizer away from the collision.
    second_prompt = synth.calls[1].content
    assert "already taken" in second_prompt.lower()


def test_forge_returns_error_on_empty_requirement() -> None:
    forge, _registry = _make_forge(ScriptedSynthesizer([]))
    result = asyncio.run(forge.forge("   "))
    assert result.success is False
    assert result.error is not None and "non-empty" in result.error
    assert result.attempts == []


def test_max_retries_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        SkillForge(
            sandbox=SandboxExecutor(),
            registry=ToolRegistry(),
            synthesizer=ScriptedSynthesizer([]),
            max_retries=0,
        )
