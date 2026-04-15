import asyncio

import pytest

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


# ---------------------------------------------------------------------------
# Step 2: Output Guard — UTF-8-safe truncation + annotation.
# ---------------------------------------------------------------------------


def test_output_under_limit_is_passed_through_unchanged() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="short",
            description="Short output",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda: "hi",
            timeout_s=1.0,
            max_output_bytes=1_000,
        )
    )
    result = asyncio.run(registry.execute("short", {}))
    assert result.success is True
    assert result.content == "hi"
    assert result.truncated is False
    assert result.output_bytes == len("hi".encode("utf-8"))


def test_output_over_limit_is_truncated_with_annotation() -> None:
    big = "A" * 200  # 200 ASCII bytes
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="big_tool",
            description="Returns big output",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda: big,
            timeout_s=1.0,
            max_output_bytes=50,
        )
    )
    result = asyncio.run(registry.execute("big_tool", {}))
    assert result.success is True
    assert result.truncated is True
    assert "Output truncated" in result.content
    assert "200 bytes" in result.content
    assert "50 bytes" in result.content
    # Content still starts with the original data.
    assert result.content.startswith("A" * 50)


def test_truncation_does_not_split_multibyte_utf8_characters() -> None:
    # 中文每个字符 3 bytes. 30 chars = 90 bytes. Limit to 50 bytes — the
    # cut lands mid-character (50 / 3 = 16.67). Safe truncate must drop
    # the partial byte sequence cleanly and never raise.
    text = "中" * 30
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="zh_tool",
            description="Returns CJK",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda: text,
            timeout_s=1.0,
            max_output_bytes=50,
        )
    )
    result = asyncio.run(registry.execute("zh_tool", {}))
    assert result.success is True
    assert result.truncated is True
    # Extract just the body (strip the annotation) and make sure it's
    # valid UTF-8 with only whole '中' characters.
    body = result.content.split("\n\n[Output truncated:")[0]
    assert body == "中" * (50 // 3)  # exactly 16 whole characters
    # No UnicodeDecodeError raised getting here is the real assertion.


def test_default_max_output_bytes_applies() -> None:
    # A ToolSpec without max_output_bytes uses DEFAULT_MAX_OUTPUT_BYTES (50_000).
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="default_tool",
            description="Uses default cap",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda: "x" * 100_000,
            timeout_s=1.0,
        )
    )
    result = asyncio.run(registry.execute("default_tool", {}))
    assert result.truncated is True
    # Body before annotation should be exactly the default cap.
    body = result.content.split("\n\n[Output truncated:")[0]
    assert len(body.encode("utf-8")) == 50_000


def test_register_rejects_non_positive_max_output_bytes() -> None:
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="max_output_bytes"):
        registry.register(
            ToolSpec(
                name="busted",
                description="bad",
                input_schema={"type": "object"},
                handler=lambda: "x",
                timeout_s=1.0,
                max_output_bytes=0,
            )
        )


# ---------------------------------------------------------------------------
# Step 3: Call audit — ring-buffered structured log + query API.
# ---------------------------------------------------------------------------


def test_audit_log_records_successful_call() -> None:
    registry = _build_registry()
    asyncio.run(registry.execute("echo", {"text": "hi"}))
    log = registry.get_audit_log()
    assert len(log) == 1
    entry = log[0]
    assert entry["tool_name"] == "echo"
    assert entry["success"] is True
    assert entry["truncated"] is False
    assert entry["elapsed_ms"] >= 0
    assert entry["input_bytes"] > 0  # {"text": "hi"} is non-empty
    assert entry["output_bytes"] == len("Echo: hi".encode("utf-8"))
    assert entry["error"] is None
    assert "T" in entry["timestamp"]  # iso8601 sanity check


def test_audit_log_records_failure_with_error_message() -> None:
    registry = _build_registry()
    asyncio.run(registry.execute("echo", {"text": 123}))  # type mismatch
    log = registry.get_audit_log()
    assert len(log) == 1
    assert log[0]["success"] is False
    assert log[0]["error"] is not None
    assert "Invalid arguments" in log[0]["error"]
    assert log[0]["output_bytes"] == 0


def test_audit_log_records_unknown_tool_call() -> None:
    registry = _build_registry()
    asyncio.run(registry.execute("no_such_tool", {}))
    log = registry.get_audit_log()
    assert len(log) == 1
    assert log[0]["tool_name"] == "no_such_tool"
    assert log[0]["success"] is False
    assert "Unknown tool" in (log[0]["error"] or "")


def test_audit_log_is_ring_buffered() -> None:
    registry = ToolRegistry(audit_log_capacity=3)
    registry.register(
        ToolSpec(
            name="echo",
            description="echo",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=lambda text: f"Echo: {text}",
            timeout_s=1.0,
        )
    )
    for i in range(10):
        asyncio.run(registry.execute("echo", {"text": f"msg{i}"}))

    log = registry.get_audit_log()
    # Capacity 3 means only the last 3 records survive.
    assert len(log) == 3


def test_audit_log_filters_by_tool_name() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="a",
            description="a",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda: "A",
            timeout_s=1.0,
        )
    )
    registry.register(
        ToolSpec(
            name="b",
            description="b",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda: "B",
            timeout_s=1.0,
        )
    )
    asyncio.run(registry.execute("a", {}))
    asyncio.run(registry.execute("b", {}))
    asyncio.run(registry.execute("a", {}))

    only_a = registry.get_audit_log(tool="a")
    assert [entry["tool_name"] for entry in only_a] == ["a", "a"]

    only_b = registry.get_audit_log(tool="b")
    assert [entry["tool_name"] for entry in only_b] == ["b"]


def test_audit_log_limit_returns_most_recent_n() -> None:
    registry = _build_registry()
    for i in range(5):
        asyncio.run(registry.execute("echo", {"text": f"n={i}"}))
    log = registry.get_audit_log(limit=2)
    assert len(log) == 2
    # Most recent last means last two calls: n=3 and n=4.
    assert log[0]["tool_name"] == "echo"
    assert log[-1]["output_bytes"] == len("Echo: n=4".encode("utf-8"))


def test_malformed_schema_is_reported_cleanly() -> None:
    # jsonschema type value must be a known string or list of strings;
    # a numeric type value is a schema error, distinct from a validation error.
    schema = {"type": "object", "properties": {"x": {"type": 123}}}
    registry = _make_registry_with_tool("busted", schema)

    result = asyncio.run(registry.execute("busted", {"x": "anything"}))
    assert result.success is False
    assert "schema" in (result.error or "").lower()
