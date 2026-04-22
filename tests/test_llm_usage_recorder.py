"""Tests for LLMClient.usage_recorder — Phase 3 token accounting.

API mode delivers exact counts via response.usage; CLI mode estimates
via character count. Budget gating (Phase 4) will subscribe here.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from wanxiang.core.llm_client import (
    LLMClient,
    _estimate_tokens_from_messages,
    _estimate_tokens_from_response,
)


def _client(**overrides) -> LLMClient:
    defaults = dict(
        model="test-model",
        max_tokens=100,
        temperature=0.0,
        api_key="test-key",
        mode="api",
    )
    defaults.update(overrides)
    return LLMClient(**defaults)


# ---- Recorder invocation ------------------------------------------------


class TestRecorderWithExactUsage:
    def test_api_usage_pumped_verbatim(self):
        calls: list[tuple[int, int, str]] = []
        client = _client(usage_recorder=lambda i, o, m: calls.append((i, o, m)))
        client._dispatch_generate_response = AsyncMock(
            return_value={
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 100, "output_tokens": 25},
            }
        )
        client.resolve_mode = AsyncMock(return_value="api")

        asyncio.run(client.generate_response(messages=[{"role": "user", "content": "x"}]))

        assert calls == [(100, 25, "api")]

    def test_no_recorder_is_safe(self):
        client = _client()  # usage_recorder=None
        client._dispatch_generate_response = AsyncMock(
            return_value={"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 1, "output_tokens": 1}}
        )
        client.resolve_mode = AsyncMock(return_value="api")
        # Must not raise.
        asyncio.run(client.generate_response(messages=[{"role": "user", "content": "x"}]))


class TestRecorderEstimatesForCli:
    def test_cli_mode_uses_estimate(self):
        calls: list[tuple[int, int, str]] = []
        client = _client(usage_recorder=lambda i, o, m: calls.append((i, o, m)))
        # CLI result has no usage field.
        client._dispatch_generate_response = AsyncMock(
            return_value={
                "id": "cli-fallback",
                "content": [{"type": "text", "text": "a" * 30}],  # ~10 tokens
            }
        )
        client.resolve_mode = AsyncMock(return_value="cli")

        system = "You are helpful" * 2  # ~30 chars → ~10 tokens
        messages = [{"role": "user", "content": "b" * 60}]  # ~20 tokens
        asyncio.run(client.generate_response(messages=messages, system=system))

        assert len(calls) == 1
        input_tokens, output_tokens, mode = calls[0]
        assert mode == "cli"
        assert input_tokens > 0
        assert output_tokens > 0

    def test_missing_usage_on_api_also_estimates(self):
        # Defensive: if the API response shape changes and usage is absent,
        # the recorder still gets a best-effort count.
        calls: list[tuple[int, int, str]] = []
        client = _client(usage_recorder=lambda i, o, m: calls.append((i, o, m)))
        client._dispatch_generate_response = AsyncMock(
            return_value={"content": [{"type": "text", "text": "output"}]}
        )
        client.resolve_mode = AsyncMock(return_value="api")

        asyncio.run(client.generate_response(messages=[{"role": "user", "content": "x"}]))

        assert len(calls) == 1
        # Tokens still reported even without `usage`.
        assert calls[0][0] >= 1 or calls[0][1] >= 1


class TestRecorderExceptionIsolation:
    def test_recorder_raising_does_not_break_generate(self):
        def boom(i, o, m):
            raise RuntimeError("bookkeeping failure")

        client = _client(usage_recorder=boom)
        client._dispatch_generate_response = AsyncMock(
            return_value={"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 1, "output_tokens": 1}}
        )
        client.resolve_mode = AsyncMock(return_value="api")
        # Must not propagate.
        result = asyncio.run(client.generate_response(messages=[{"role": "user", "content": "x"}]))
        assert result["content"][0]["text"] == "hi"


# ---- Estimator helpers --------------------------------------------------


class TestEstimateTokens:
    def test_zero_for_empty_input(self):
        assert _estimate_tokens_from_messages([], None) == 1  # max(1, ...)

    def test_scales_with_chars(self):
        short = _estimate_tokens_from_messages(
            [{"role": "user", "content": "a" * 30}], None
        )
        long = _estimate_tokens_from_messages(
            [{"role": "user", "content": "a" * 300}], None
        )
        assert long > short * 5  # order-of-magnitude check

    def test_includes_system_prompt(self):
        with_system = _estimate_tokens_from_messages(
            [], "a" * 300
        )
        assert with_system > 50

    def test_response_empty_returns_zero(self):
        assert _estimate_tokens_from_response({"content": []}) == 0

    def test_response_text_counted(self):
        result = {"content": [{"type": "text", "text": "hello world"}]}
        assert _estimate_tokens_from_response(result) >= 1

    def test_response_ignores_non_text_blocks(self):
        result = {
            "content": [
                {"type": "tool_use", "input": {"x": 1}},
                {"type": "text", "text": "out"},
            ]
        }
        # Only the text block counts.
        assert _estimate_tokens_from_response(result) == max(1, len("out") // 3)


class TestListContentMessages:
    def test_structured_content_counted(self):
        # Some clients pass content as a list of blocks rather than a string.
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "a" * 60}]}
        ]
        tokens = _estimate_tokens_from_messages(messages, None)
        assert tokens >= 15
