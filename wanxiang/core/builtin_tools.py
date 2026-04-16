from __future__ import annotations

from datetime import datetime, timezone

from .tools import ToolRegistry, ToolSpec


def _echo_handler(text: str) -> str:
    return f"Echo: {text}"


def _current_time_handler() -> str:
    return datetime.now(timezone.utc).isoformat()


def _web_search_handler(query: str, max_results: int = 5) -> str:
    from ddgs import DDGS

    clamped = max(1, min(max_results, 10))
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=clamped))
    if not results:
        return "No results found."
    lines = []
    for r in results:
        title = r.get("title", "")
        url = r.get("href", "")
        snippet = r.get("body", "")
        lines.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")
    return "\n\n---\n\n".join(lines)


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
    registry.register(
        ToolSpec(
            name="web_search",
            description=(
                "Search the web using DuckDuckGo. Returns titles, URLs and "
                "snippets for the top results. Use for fact-checking, finding "
                "recent information, or researching topics."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (1-10, default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            handler=_web_search_handler,
            timeout_s=15.0,
        )
    )
    return registry
