import asyncio
from datetime import datetime

from wanxiang.core.builtin_tools import create_default_registry


def test_default_registry_contains_echo_and_current_time() -> None:
    registry = create_default_registry()
    assert set(registry.list_tools()) == {"echo", "current_time"}


def test_default_echo_tool_executes() -> None:
    registry = create_default_registry()
    result = asyncio.run(registry.execute("echo", {"text": "abc"}))
    assert result.success is True
    assert result.content == "Echo: abc"


def test_default_current_time_tool_executes() -> None:
    registry = create_default_registry()
    result = asyncio.run(registry.execute("current_time", {}))
    assert result.success is True
    parsed = datetime.fromisoformat(result.content)
    assert parsed.tzinfo is not None
