"""Bridge an MCP client's tools into the local ToolRegistry.

MCP tools/list returns tool specs whose `inputSchema` is already
JSON Schema — identical to what Claude API expects for `input_schema`.
So we pass the schema through unchanged and wrap the async call_tool
invocation in a local handler.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from .mcp_client import MCPClient
from .tools import ToolRegistry, ToolSpec


logger = logging.getLogger("wanxiang.mcp.bridge")


def extract_text_from_mcp_result(result: dict[str, Any]) -> str:
    """Extract human-readable text from a tools/call result.

    If the result has `isError: true`, raises RuntimeError so the
    ToolRegistry surfaces a ToolResult(success=False, error=...).
    """
    content = result.get("content")
    text = _content_to_text(content)
    if result.get("isError"):
        raise RuntimeError(
            f"MCP tool reported error: {text or 'unknown error'}"
        )
    return text


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", ""))
        if block_type == "text":
            parts.append(str(block.get("text", "")))
        elif block_type == "image":
            mime = str(block.get("mimeType", "unknown"))
            parts.append(f"[image · {mime}]")
        elif block_type == "resource":
            resource = block.get("resource") or {}
            uri = str(resource.get("uri", ""))
            parts.append(f"[resource · {uri}]" if uri else "[resource]")
    return "\n".join(part for part in parts if part).strip()


def _make_mcp_handler(client: MCPClient, tool_name: str) -> Callable[..., Any]:
    async def handler(**kwargs: Any) -> str:
        result = await client.call_tool(tool_name, kwargs)
        return extract_text_from_mcp_result(result)

    handler.__name__ = f"mcp_{tool_name}"
    handler.__qualname__ = f"mcp_handler.{client.server_name}.{tool_name}"
    return handler


async def register_mcp_tools(
    registry: ToolRegistry,
    client: MCPClient,
    *,
    prefix: str = "",
    timeout_s: float = 30.0,
) -> list[str]:
    """Fetch the MCP server's tools and register them into the registry.

    Args:
        registry: Target tool registry.
        client: An already-started MCPClient.
        prefix: If non-empty, registered names become `"{prefix}.{raw_name}"`.
            Use this only to resolve collisions between multiple servers;
            prefer the raw name when possible so Claude's tool_use blocks
            (which reference names as declared to the API) match directly.
        timeout_s: Per-call timeout attached to the resulting ToolSpec.

    Returns the list of names that were actually registered (skipping
    empty-name or duplicate entries).
    """
    tools = await client.list_tools()
    registered: list[str] = []
    for tool in tools:
        raw_name = str(tool.get("name", "")).strip()
        if not raw_name:
            continue
        registered_name = f"{prefix}.{raw_name}" if prefix else raw_name
        input_schema = tool.get("inputSchema") or {
            "type": "object",
            "properties": {},
        }
        description = str(tool.get("description") or f"MCP tool: {raw_name}")
        spec = ToolSpec(
            name=registered_name,
            description=description,
            input_schema=input_schema,
            handler=_make_mcp_handler(client, raw_name),
            timeout_s=timeout_s,
        )
        try:
            registry.register(spec)
        except ValueError as exc:
            logger.warning(
                "Skipping MCP tool %s from server=%s: %s",
                registered_name,
                client.server_name,
                exc,
            )
            continue
        registered.append(registered_name)

    logger.info(
        "Registered %d MCP tools from server=%s: %s",
        len(registered),
        client.server_name,
        registered,
    )
    return registered
