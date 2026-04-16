import asyncio

import pytest

from wanxiang.core.llm_client import (
    DEFAULT_LLM_CALL_TIMEOUT_S,
    DEFAULT_TIMEOUT_RETRY_WAITS_S,
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
    # Disable retries so this test isolates the single-attempt timeout
    # path (separate retry-ladder tests cover multi-attempt behavior).
    client = LLMClient(
        api_key="k", mode="api", llm_call_timeout_s=0.2, timeout_retry_waits_s=()
    )

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


# ---- timeout-only retry ladder -------------------------------------------


def test_default_retry_ladder_is_30_then_60() -> None:
    assert DEFAULT_TIMEOUT_RETRY_WAITS_S == (30.0, 60.0)


def test_negative_retry_wait_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LLMClient(
            api_key="k",
            mode="api",
            timeout_retry_waits_s=(30.0, -5.0),
        )


def test_retry_eventually_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hang on first attempt, succeed on second. Retry must fire and
    the final return value must match the second attempt."""
    client = LLMClient(
        api_key="k",
        mode="api",
        llm_call_timeout_s=0.2,
        timeout_retry_waits_s=(0.0,),  # zero-sleep retry keeps test fast
    )

    call_count = {"n": 0}

    async def _flaky(**kwargs: object) -> dict:
        call_count["n"] += 1
        if call_count["n"] == 1:
            await asyncio.sleep(5.0)  # forces first attempt to time out
            return {}
        return {"content": [{"type": "text", "text": "recovered"}]}

    monkeypatch.setattr(client, "_dispatch_generate_response", _flaky)

    result = asyncio.run(
        client.generate_response(messages=[{"role": "user", "content": "hi"}])
    )
    assert result["content"][0]["text"] == "recovered"
    assert call_count["n"] == 2


def test_retry_exhausted_raises_with_attempt_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every attempt times out, the final RuntimeError carries
    the attempt count so mining can see "after N attempts"."""
    client = LLMClient(
        api_key="k",
        mode="api",
        llm_call_timeout_s=0.1,
        timeout_retry_waits_s=(0.0, 0.0),  # max_attempts = 3
    )

    async def _always_hangs(**kwargs: object) -> dict:
        await asyncio.sleep(5.0)
        return {}

    monkeypatch.setattr(client, "_dispatch_generate_response", _always_hangs)

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(client.generate_response(messages=[{"role": "user", "content": "hi"}]))

    msg = str(excinfo.value)
    assert "LLM call exceeded" in msg
    assert "after 3 attempts" in msg
    assert "Mode: api" in msg


def test_non_timeout_errors_do_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only asyncio.TimeoutError should trigger retry. A real error
    (bad JSON, auth failure) should propagate immediately."""
    client = LLMClient(
        api_key="k",
        mode="api",
        llm_call_timeout_s=2.0,
        timeout_retry_waits_s=(0.0, 0.0),
    )

    call_count = {"n": 0}

    async def _raises_logic_error(**kwargs: object) -> dict:
        call_count["n"] += 1
        raise ValueError("bad JSON from LLM")

    monkeypatch.setattr(client, "_dispatch_generate_response", _raises_logic_error)

    with pytest.raises(ValueError, match="bad JSON"):
        asyncio.run(client.generate_response(messages=[{"role": "user", "content": "hi"}]))

    # Must not have retried; only one attempt.
    assert call_count["n"] == 1


def test_no_retries_configured_behaves_as_single_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LLMClient(
        api_key="k",
        mode="api",
        llm_call_timeout_s=0.1,
        timeout_retry_waits_s=(),  # no retries
    )

    async def _hang(**kwargs: object) -> dict:
        await asyncio.sleep(5.0)
        return {}

    monkeypatch.setattr(client, "_dispatch_generate_response", _hang)

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(client.generate_response(messages=[{"role": "user", "content": "hi"}]))

    assert "after 1 attempts" in str(excinfo.value)
