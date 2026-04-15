import asyncio

from wanxiang.core.llm_client import LLMClient


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
