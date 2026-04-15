import asyncio

from wanxiang.core.tools import ToolRegistry, ToolSpec


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="Echo input text",
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


def test_register_and_list_tools() -> None:
    registry = _build_registry()
    assert registry.list_tools() == ["echo"]
    assert registry.get("echo") is not None


def test_register_duplicate_tool_raises_error() -> None:
    registry = _build_registry()
    try:
        registry.register(
            ToolSpec(
                name="echo",
                description="duplicate",
                input_schema={"type": "object"},
                handler=lambda: "x",
            )
        )
        raise AssertionError("expected duplicate registration to fail")
    except ValueError as exc:
        assert "already registered" in str(exc)


def test_filter_for_agent_keeps_allowed_order_and_existing_only() -> None:
    registry = _build_registry()
    registry.register(
        ToolSpec(
            name="current_time",
            description="Current UTC time",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda: "2026-01-01T00:00:00Z",
        )
    )

    filtered = registry.filter_for_agent(["current_time", "missing", "echo", "echo"])
    assert [tool.name for tool in filtered] == ["current_time", "echo"]


def test_execute_successful_sync_tool() -> None:
    registry = _build_registry()
    result = asyncio.run(registry.execute("echo", {"text": "hello"}))

    assert result.tool_name == "echo"
    assert result.success is True
    assert result.error is None
    assert result.content == "Echo: hello"
    assert result.elapsed_ms >= 0


def test_execute_unknown_tool_returns_error_result() -> None:
    registry = _build_registry()
    result = asyncio.run(registry.execute("not_exists", {"x": 1}))

    assert result.tool_name == "not_exists"
    assert result.success is False
    assert "Unknown tool" in (result.error or "")


def test_execute_validation_failure_returns_error_result() -> None:
    registry = _build_registry()
    result = asyncio.run(registry.execute("echo", {"text": 123}))

    assert result.success is False
    assert "Invalid arguments" in (result.error or "")
    assert "expected type 'string'" in (result.error or "")


def test_execute_timeout_returns_error_result() -> None:
    async def slow_tool(delay_s: float) -> str:
        await asyncio.sleep(delay_s)
        return "done"

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="slow",
            description="Slow async tool",
            input_schema={
                "type": "object",
                "properties": {"delay_s": {"type": "number"}},
                "required": ["delay_s"],
            },
            handler=slow_tool,
            timeout_s=0.01,
        )
    )

    result = asyncio.run(registry.execute("slow", {"delay_s": 0.1}))
    assert result.success is False
    assert "timed out" in (result.error or "")


def test_execute_handler_exception_returns_error_result() -> None:
    def broken_tool() -> str:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="broken",
            description="Always fails",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=broken_tool,
            timeout_s=1.0,
        )
    )

    result = asyncio.run(registry.execute("broken", {}))
    assert result.success is False
    assert "RuntimeError" in (result.error or "")
    assert "boom" in (result.error or "")
