import asyncio

import pytest

from wanxiang.core.llm_client import (
    DEFAULT_LLM_CALL_TIMEOUT_S,
    LLMClient,
)


def test_resolve_mode_api_when_explicit_and_key_present() -> None:
    client = LLMClient(api_key="k", mode="api")
    mode = asyncio.run(client.resolve_mode())
    assert mode == "api"


def test_resolve_mode_api_raises_when_key_missing() -> None:
    client = LLMClient(api_key=None, mode="api")
    try:
        asyncio.run(client.resolve_mode())
        raise AssertionError("expected api mode without key to fail")
    except RuntimeError as exc:
        assert "ANTHROPIC_API_KEY" in str(exc)


def test_resolve_mode_cli_when_authenticated() -> None:
    client = LLMClient(api_key=None, mode="cli")
    client._claude_bin = "/usr/bin/claude"  # noqa: SLF001 - test setup
    client._cli_auth_cache = True  # noqa: SLF001 - test setup
    mode = asyncio.run(client.resolve_mode())
    assert mode == "cli"


def test_resolve_mode_auto_prefers_api() -> None:
    client = LLMClient(api_key="k", mode="auto")
    client._claude_bin = "/usr/bin/claude"  # noqa: SLF001 - test setup
    client._cli_auth_cache = True  # noqa: SLF001 - test setup
    mode = asyncio.run(client.resolve_mode(require_tools=True))
    assert mode == "api"


def test_resolve_mode_auto_falls_back_to_cli() -> None:
    client = LLMClient(api_key=None, mode="auto")
    client._claude_bin = "/usr/bin/claude"  # noqa: SLF001 - test setup
    client._cli_auth_cache = True  # noqa: SLF001 - test setup
    mode = asyncio.run(client.resolve_mode(require_tools=True))
    assert mode == "cli"


# ---- hard timeout ---------------------------------------------------------


def test_default_timeout_value() -> None:
    client = LLMClient(api_key="k", mode="api")
    assert client.llm_call_timeout_s == DEFAULT_LLM_CALL_TIMEOUT_S


def test_custom_timeout_value() -> None:
    client = LLMClient(api_key="k", mode="api", llm_call_timeout_s=5.0)
    assert client.llm_call_timeout_s == 5.0


def test_invalid_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="llm_call_timeout_s"):
        LLMClient(api_key="k", mode="api", llm_call_timeout_s=0)
    with pytest.raises(ValueError, match="llm_call_timeout_s"):
        LLMClient(api_key="k", mode="api", llm_call_timeout_s=-1.0)


def test_timeout_wraps_slow_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the underlying dispatch never returns, wait_for converts the
    TimeoutError into a RuntimeError with a recognizable keyword."""
    client = LLMClient(api_key="k", mode="api", llm_call_timeout_s=0.2)

    async def _hang(**kwargs: object) -> dict:
        await asyncio.sleep(5.0)
        return {}

    monkeypatch.setattr(client, "_dispatch_generate_response", _hang)

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(client.generate_response(messages=[{"role": "user", "content": "hi"}]))

    assert "LLM call exceeded" in str(excinfo.value)
    assert "Mode: api" in str(excinfo.value)


def test_fast_dispatch_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fast responses are not affected by the timeout wrapper."""
    client = LLMClient(api_key="k", mode="api", llm_call_timeout_s=2.0)

    async def _fast(**kwargs: object) -> dict:
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(client, "_dispatch_generate_response", _fast)

    result = asyncio.run(client.generate_response(messages=[{"role": "user", "content": "hi"}]))
    assert result["content"][0]["text"] == "ok"


def test_terminate_subprocess_handles_already_exited() -> None:
    """_terminate_subprocess must tolerate processes that already exited."""
    from wanxiang.core.llm_client import _terminate_subprocess

    class _FakeDoneProc:
        returncode = 0

        def kill(self) -> None:  # pragma: no cover - should not be called
            raise AssertionError("kill called on already-exited process")

        async def wait(self) -> int:
            return 0

    asyncio.run(_terminate_subprocess(_FakeDoneProc()))


def test_terminate_subprocess_kills_running() -> None:
    """_terminate_subprocess must actually kill a running process."""
    from wanxiang.core.llm_client import _terminate_subprocess

    killed = {"flag": False}
    waited = {"flag": False}

    class _FakeRunningProc:
        returncode = None

        def kill(self) -> None:
            killed["flag"] = True

        async def wait(self) -> int:
            waited["flag"] = True
            return -9

    asyncio.run(_terminate_subprocess(_FakeRunningProc()))
    assert killed["flag"] is True
    assert waited["flag"] is True
