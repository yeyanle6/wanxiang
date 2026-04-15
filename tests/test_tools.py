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
    # jsonschema reports the type mismatch with "is not of type 'string'".
    assert "not of type" in (result.error or "")
    assert "string" in (result.error or "")


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


# ---------------------------------------------------------------------------
# JSON Schema validation — jsonschema-backed coverage for Phase 3D tool hardening.
# ---------------------------------------------------------------------------


def _make_registry_with_tool(name: str, schema: dict, handler=lambda **kw: "ok") -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name=name,
            description=f"Tool {name}",
            input_schema=schema,
            handler=handler,
            timeout_s=1.0,
        )
    )
    return registry


def test_enum_value_is_validated() -> None:
    schema = {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["fast", "slow"]}},
        "required": ["mode"],
    }
    registry = _make_registry_with_tool("setter", schema)

    ok = asyncio.run(registry.execute("setter", {"mode": "fast"}))
    bad = asyncio.run(registry.execute("setter", {"mode": "medium"}))

    assert ok.success is True
    assert bad.success is False
    assert "enum" in (bad.error or "").lower() or "medium" in (bad.error or "")


def test_numeric_range_is_validated() -> None:
    schema = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["score"],
    }
    registry = _make_registry_with_tool("rater", schema)

    assert asyncio.run(registry.execute("rater", {"score": 42})).success is True
    assert asyncio.run(registry.execute("rater", {"score": -1})).success is False
    assert asyncio.run(registry.execute("rater", {"score": 101})).success is False


def test_string_pattern_is_validated() -> None:
    schema = {
        "type": "object",
        "properties": {
            "slug": {"type": "string", "pattern": r"^[a-z0-9-]+$"},
        },
        "required": ["slug"],
    }
    registry = _make_registry_with_tool("slug_tool", schema)

    assert asyncio.run(registry.execute("slug_tool", {"slug": "hello-world"})).success is True
    bad = asyncio.run(registry.execute("slug_tool", {"slug": "Hello World"}))
    assert bad.success is False
    # jsonschema surfaces pattern mismatch as "does not match '<regex>'".
    assert "does not match" in (bad.error or "").lower()
    assert "slug" in (bad.error or "")


def test_nested_object_is_validated() -> None:
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                },
                "required": ["name", "age"],
            }
        },
        "required": ["user"],
    }
    registry = _make_registry_with_tool("user_tool", schema, handler=lambda **kw: "ok")

    ok = asyncio.run(registry.execute("user_tool", {"user": {"name": "alice", "age": 30}}))
    assert ok.success is True

    missing_age = asyncio.run(registry.execute("user_tool", {"user": {"name": "alice"}}))
    assert missing_age.success is False
    assert "age" in (missing_age.error or "")

    bad_age_type = asyncio.run(
        registry.execute("user_tool", {"user": {"name": "alice", "age": "thirty"}})
    )
    assert bad_age_type.success is False
    # Error path should identify the nested field.
    assert "user" in (bad_age_type.error or "") or "age" in (bad_age_type.error or "")


def test_array_items_are_validated() -> None:
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
        },
        "required": ["tags"],
    }
    registry = _make_registry_with_tool("tagger", schema)

    assert asyncio.run(registry.execute("tagger", {"tags": ["a", "b"]})).success is True
    bad_item = asyncio.run(registry.execute("tagger", {"tags": ["a", 2]}))
    assert bad_item.success is False
    empty_list = asyncio.run(registry.execute("tagger", {"tags": []}))
    assert empty_list.success is False


def test_additional_properties_false_rejects_unknown_fields() -> None:
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    registry = _make_registry_with_tool("strict", schema)

    assert asyncio.run(registry.execute("strict", {"text": "hi"})).success is True
    bad = asyncio.run(registry.execute("strict", {"text": "hi", "extra": 1}))
    assert bad.success is False
    assert "additional" in (bad.error or "").lower() or "extra" in (bad.error or "")


def test_any_of_schema_accepts_alternatives() -> None:
    schema = {
        "type": "object",
        "properties": {
            "value": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        },
        "required": ["value"],
    }
    registry = _make_registry_with_tool("flex", schema)

    assert asyncio.run(registry.execute("flex", {"value": "text"})).success is True
    assert asyncio.run(registry.execute("flex", {"value": 42})).success is True
    bad = asyncio.run(registry.execute("flex", {"value": True}))
    # bool is not string nor integer per JSON Schema; jsonschema enforces this.
    assert bad.success is False


def test_empty_schema_accepts_any_arguments() -> None:
    registry = _make_registry_with_tool("wide_open", {})

    # With no schema constraints, arguments flow through.
    assert asyncio.run(registry.execute("wide_open", {})).success is True


def test_malformed_schema_is_reported_cleanly() -> None:
    # jsonschema type value must be a known string or list of strings;
    # a numeric type value is a schema error, distinct from a validation error.
    schema = {"type": "object", "properties": {"x": {"type": 123}}}
    registry = _make_registry_with_tool("busted", schema)

    result = asyncio.run(registry.execute("busted", {"x": "anything"}))
    assert result.success is False
    assert "schema" in (result.error or "").lower()
