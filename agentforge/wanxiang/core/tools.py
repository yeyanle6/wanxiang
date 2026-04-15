from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable


ToolHandler = Callable[..., Any]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    timeout_s: float = 30.0
    requires_confirmation: bool = False

    def to_claude_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    success: bool
    content: str
    elapsed_ms: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "content": self.content,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.logger = logging.getLogger("wanxiang.tools")

    def register(self, spec: ToolSpec) -> None:
        name = spec.name.strip()
        if not name:
            raise ValueError("Tool name cannot be empty.")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        if spec.timeout_s <= 0:
            raise ValueError(f"Tool timeout must be positive for {name}.")
        self._tools[name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

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

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        spec = self.get(name)
        started = time.monotonic()

        if spec is None:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            return ToolResult(
                tool_name=name,
                success=False,
                content="",
                elapsed_ms=elapsed_ms,
                error=f"Unknown tool: {name}",
            )

        validation_error = self._validate_arguments(spec.input_schema, arguments)
        if validation_error is not None:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            return ToolResult(
                tool_name=name,
                success=False,
                content="",
                elapsed_ms=elapsed_ms,
                error=f"Invalid arguments: {validation_error}",
            )

        try:
            raw_result = await asyncio.wait_for(
                self._invoke_handler(spec, arguments),
                timeout=spec.timeout_s,
            )
            content = self._stringify_result(raw_result)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            self.logger.info(
                "Tool executed: name=%s success=true elapsed_ms=%s",
                name,
                elapsed_ms,
            )
            return ToolResult(
                tool_name=name,
                success=True,
                content=content,
                elapsed_ms=elapsed_ms,
            )
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            self.logger.warning(
                "Tool timeout: name=%s elapsed_ms=%s timeout_s=%s",
                name,
                elapsed_ms,
                spec.timeout_s,
            )
            return ToolResult(
                tool_name=name,
                success=False,
                content="",
                elapsed_ms=elapsed_ms,
                error=f"Tool timed out after {spec.timeout_s:.2f}s",
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            self.logger.exception(
                "Tool failed: name=%s elapsed_ms=%s error=%s",
                name,
                elapsed_ms,
                type(exc).__name__,
            )
            return ToolResult(
                tool_name=name,
                success=False,
                content="",
                elapsed_ms=elapsed_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _invoke_handler(self, spec: ToolSpec, arguments: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(spec.handler):
            maybe = spec.handler(**arguments)
            return await maybe
        return await asyncio.to_thread(spec.handler, **arguments)

    def _validate_arguments(self, schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
        if not isinstance(arguments, dict):
            return "arguments must be an object"
        if not isinstance(schema, dict):
            return None

        expected_type = schema.get("type")
        if expected_type and expected_type != "object":
            return "input_schema.type must be 'object'"

        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in arguments:
                    return f"missing required field '{key}'"

        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, value in arguments.items():
                prop_schema = properties.get(key)
                if not isinstance(prop_schema, dict):
                    if schema.get("additionalProperties", True):
                        continue
                    return f"unexpected field '{key}'"
                expected = prop_schema.get("type")
                if expected and not self._matches_type(value, str(expected)):
                    return f"field '{key}' expected type '{expected}'"
        return None

    def _matches_type(self, value: Any, expected: str) -> bool:
        if expected == "string":
            return isinstance(value, str)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)
        if expected == "object":
            return isinstance(value, dict)
        if expected == "array":
            return isinstance(value, list)
        if expected == "null":
            return value is None
        return True

    def _stringify_result(self, raw_result: Any) -> str:
        if isinstance(raw_result, str):
            return raw_result
        try:
            return json.dumps(raw_result, ensure_ascii=False)
        except Exception:
            return str(raw_result)
