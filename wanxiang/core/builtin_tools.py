from __future__ import annotations

from datetime import datetime, timezone

from .tools import ToolRegistry, ToolSpec


def _echo_handler(text: str) -> str:
    return f"Echo: {text}"


def _current_time_handler() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="Echoes back the input text. Useful for testing tool integration.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to echo back"},
                },
                "required": ["text"],
            },
            handler=_echo_handler,
        )
    )
    registry.register(
        ToolSpec(
            name="current_time",
            description="Returns the current UTC date and time. Use when you need to know what time it is.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=_current_time_handler,
        )
    )
    return registry
