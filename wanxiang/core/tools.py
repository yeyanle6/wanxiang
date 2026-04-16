from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque


ToolHandler = Callable[..., Any]

# 50 KB default output cap. Large enough for filesystem reads and search
# results, small enough that a runaway tool can't flood the LLM context.
DEFAULT_MAX_OUTPUT_BYTES = 50_000

# Ring-buffer capacity for the registry's call audit log. Long-running
# servers must not grow memory without bound; 1000 recent calls is
# enough for trace mining while staying flat.
DEFAULT_AUDIT_LOG_CAPACITY = 1000


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    timeout_s: float = 30.0
    requires_confirmation: bool = False
    # Source group: empty string = builtin. MCP bridge sets this to the
    # server name so the planner prompt can group related tools together.
    group: str = ""
    # Exact agent names allowed to use this tool. Empty list = any agent.
    # Enforced by AgentFactory policy layer when applying team plans.
    allowed_agents: list[str] = field(default_factory=list)
    # Upper bound on stringified output in bytes. Output over the limit
    # is truncated on a UTF-8 boundary and annotated for the LLM.
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES

    def to_claude_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def is_agent_allowed(self, agent_name: str) -> bool:
        if not self.allowed_agents:
            return True
        return agent_name in self.allowed_agents


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    success: bool
    content: str
    elapsed_ms: int
    error: str | None = None
    # Set to True if the registry truncated content to fit max_output_bytes.
    truncated: bool = False
    # Size of the content actually delivered (post-truncation), in bytes.
    output_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "content": self.content,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "truncated": self.truncated,
            "output_bytes": self.output_bytes,
        }


@dataclass(slots=True)
class ToolCallRecord:
    tool_name: str
    timestamp: str
    success: bool
    elapsed_ms: int
    input_bytes: int
    output_bytes: int
    truncated: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "timestamp": self.timestamp,
            "success": self.success,
            "elapsed_ms": self.elapsed_ms,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "truncated": self.truncated,
            "error": self.error,
        }


def _safe_truncate_utf8(text: str, max_bytes: int) -> tuple[str, bool, int]:
    """Truncate `text` to fit within `max_bytes` on a UTF-8 boundary.

    Returns (truncated_text, was_truncated, delivered_byte_size).
    `delivered_byte_size` counts the final payload bytes *including*
    any appended annotation, so audit records stay consistent with
    what the LLM actually sees.
    """
    encoded = text.encode("utf-8")
    original_size = len(encoded)
    if max_bytes <= 0 or original_size <= max_bytes:
        return text, False, original_size

    # errors='ignore' drops the trailing partial multi-byte sequence
    # cleanly without raising UnicodeDecodeError.
    body = encoded[:max_bytes].decode("utf-8", errors="ignore")
    annotation = (
        f"\n\n[Output truncated: {original_size} bytes → {max_bytes} bytes]"
    )
    delivered = body + annotation
    return delivered, True, len(delivered.encode("utf-8"))


class ToolRegistry:
    def __init__(self, audit_log_capacity: int = DEFAULT_AUDIT_LOG_CAPACITY) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._audit_log: Deque[ToolCallRecord] = deque(maxlen=audit_log_capacity)
        self.logger = logging.getLogger("wanxiang.tools")

    def register(self, spec: ToolSpec) -> None:
        name = spec.name.strip()
        if not name:
            raise ValueError("Tool name cannot be empty.")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        if spec.timeout_s <= 0:
            raise ValueError(f"Tool timeout must be positive for {name}.")
        if spec.max_output_bytes <= 0:
            raise ValueError(
                f"Tool max_output_bytes must be positive for {name}."
            )
        self._tools[name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_tool_groups(self) -> dict[str, str]:
        """Return a ``{tool_name: group}`` map for all registered tools.

        Used by trace mining to classify tools into builtin / mcp /
        synthesized buckets without having to replicate the registry's
        grouping logic.
        """
        return {name: spec.group for name, spec in self._tools.items()}

    def filter_for_agent(self, allowed: list[str]) -> list[ToolSpec]:
        seen: set[str] = set()
        filtered: list[ToolSpec] = []
        for name in allowed:
            if name in seen:
                continue
            seen.add(name)
            spec = self.get(name)
            if spec is not None:
                filtered.append(spec)
        return filtered

    def get_audit_log(
        self, *, limit: int | None = None, tool: str | None = None
    ) -> list[dict[str, Any]]:
        """Return recent tool-call records, newest last.

        `limit` caps the number of records returned (takes the most
        recent N). `tool` filters by tool_name exact match before
        applying the limit.
        """
        records = list(self._audit_log)
        if tool:
            records = [r for r in records if r.tool_name == tool]
        if limit is not None and limit >= 0:
            records = records[-limit:]
        return [r.to_dict() for r in records]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        spec = self.get(name)
        started = time.monotonic()
        input_bytes = self._measure_input_bytes(arguments)

        if spec is None:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            result = ToolResult(
                tool_name=name,
                success=False,
                content="",
                elapsed_ms=elapsed_ms,
                error=f"Unknown tool: {name}",
                truncated=False,
                output_bytes=0,
            )
            self._record(result, input_bytes=input_bytes)
            return result

        validation_error = self._validate_arguments(spec.input_schema, arguments)
        if validation_error is not None:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            result = ToolResult(
                tool_name=name,
                success=False,
                content="",
                elapsed_ms=elapsed_ms,
                error=f"Invalid arguments: {validation_error}",
                truncated=False,
                output_bytes=0,
            )
            self._record(result, input_bytes=input_bytes)
            return result

        try:
            raw_result = await asyncio.wait_for(
                self._invoke_handler(spec, arguments),
                timeout=spec.timeout_s,
            )
            content = self._stringify_result(raw_result)
            content, truncated, delivered_bytes = _safe_truncate_utf8(
                content, spec.max_output_bytes
            )
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if truncated:
                self.logger.warning(
                    "Tool output truncated: name=%s delivered_bytes=%s limit=%s",
                    name,
                    delivered_bytes,
                    spec.max_output_bytes,
                )
            self.logger.info(
                "Tool executed: name=%s success=true elapsed_ms=%s output_bytes=%s truncated=%s",
                name,
                elapsed_ms,
                delivered_bytes,
                truncated,
            )
            result = ToolResult(
                tool_name=name,
                success=True,
                content=content,
                elapsed_ms=elapsed_ms,
                truncated=truncated,
                output_bytes=delivered_bytes,
            )
            self._record(result, input_bytes=input_bytes)
            return result
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            self.logger.warning(
                "Tool timeout: name=%s elapsed_ms=%s timeout_s=%s",
                name,
                elapsed_ms,
                spec.timeout_s,
            )
            result = ToolResult(
                tool_name=name,
                success=False,
                content="",
                elapsed_ms=elapsed_ms,
                error=f"Tool timed out after {spec.timeout_s:.2f}s",
                truncated=False,
                output_bytes=0,
            )
            self._record(result, input_bytes=input_bytes)
            return result
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            self.logger.exception(
                "Tool failed: name=%s elapsed_ms=%s error=%s",
                name,
                elapsed_ms,
                type(exc).__name__,
            )
            result = ToolResult(
                tool_name=name,
                success=False,
                content="",
                elapsed_ms=elapsed_ms,
                error=f"{type(exc).__name__}: {exc}",
                truncated=False,
                output_bytes=0,
            )
            self._record(result, input_bytes=input_bytes)
            return result

    def _record(self, result: ToolResult, *, input_bytes: int) -> None:
        self._audit_log.append(
            ToolCallRecord(
                tool_name=result.tool_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=result.success,
                elapsed_ms=result.elapsed_ms,
                input_bytes=input_bytes,
                output_bytes=result.output_bytes,
                truncated=result.truncated,
                error=result.error,
            )
        )

    def _measure_input_bytes(self, arguments: Any) -> int:
        try:
            return len(
                json.dumps(arguments, ensure_ascii=False).encode("utf-8")
            )
        except Exception:
            return 0

    async def _invoke_handler(self, spec: ToolSpec, arguments: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(spec.handler):
            maybe = spec.handler(**arguments)
            return await maybe
        return await asyncio.to_thread(spec.handler, **arguments)

    def _validate_arguments(self, schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
        """Validate `arguments` against JSON Schema using jsonschema library.

        Delegates to the reference JSON Schema validator so we get full
        Draft 7+ coverage for free: nested objects, enum, minimum/maximum,
        pattern, array items, anyOf/oneOf, additionalProperties, etc.
        """
        if not isinstance(arguments, dict):
            return "arguments must be an object"
        if not isinstance(schema, dict) or not schema:
            return None
        try:
            import jsonschema
        except ImportError:  # pragma: no cover - dependency should always be installed
            return "jsonschema library is not installed"
        try:
            jsonschema.validate(instance=arguments, schema=schema)
        except jsonschema.ValidationError as exc:
            location = ".".join(str(part) for part in exc.absolute_path)
            if location:
                return f"{exc.message} (at '{location}')"
            return exc.message
        except jsonschema.SchemaError as exc:
            return f"invalid input_schema: {exc.message}"
        return None

    def _stringify_result(self, raw_result: Any) -> str:
        if isinstance(raw_result, str):
            return raw_result
        try:
            return json.dumps(raw_result, ensure_ascii=False)
        except Exception:
            return str(raw_result)
