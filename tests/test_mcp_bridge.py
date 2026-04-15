"""Unit tests for MCP → ToolRegistry bridge."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from wanxiang.core.mcp_bridge import (
    extract_text_from_mcp_result,
    register_mcp_tools,
)
from wanxiang.core.tools import ToolRegistry


class _FakeMCPClient:
    """Duck-typed MCPClient used for bridge-level tests.

    Records call_tool invocations and returns scripted results.
    """

    def __init__(self, tools: list[dict], call_result: dict | None = None) -> None:
        self.server_name = "fake"
        self._tools = tools
        self._call_result = call_result or {
            "content": [{"type": "text", "text": "ok"}],
            "isError": False,
        }
        self.call_log: list[tuple[str, dict]] = []

    async def list_tools(self) -> list[dict]:
        return list(self._tools)

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.call_log.append((name, dict(arguments)))
        return self._call_result


def test_extract_text_joins_text_blocks() -> None:
    result = {
        "content": [
            {"type": "text", "text": "line 1"},
            {"type": "text", "text": "line 2"},
        ],
        "isError": False,
    }
    assert extract_text_from_mcp_result(result) == "line 1\nline 2"


def test_extract_text_surfaces_image_and_resource_tags() -> None:
    result = {
        "content": [
            {"type": "image", "mimeType": "image/png"},
            {"type": "resource", "resource": {"uri": "file:///foo.txt"}},
            {"type": "text", "text": "caption"},
        ],
        "isError": False,
    }
    text = extract_text_from_mcp_result(result)
    assert "[image · image/png]" in text
    assert "[resource · file:///foo.txt]" in text
    assert "caption" in text


def test_extract_text_raises_when_is_error_is_true() -> None:
    result = {
        "content": [{"type": "text", "text": "path does not exist"}],
        "isError": True,
    }
    with pytest.raises(RuntimeError) as excinfo:
        extract_text_from_mcp_result(result)
    assert "path does not exist" in str(excinfo.value)


def test_extract_text_handles_string_content() -> None:
    assert extract_text_from_mcp_result({"content": "plain string"}) == "plain string"


def test_extract_text_handles_missing_content() -> None:
    assert extract_text_from_mcp_result({}) == ""


def test_register_mcp_tools_passes_schema_through_unchanged() -> None:
    async def scenario() -> None:
        client = _FakeMCPClient(
            tools=[
                {
                    "name": "read_file",
                    "description": "Read a file from the sandbox.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ]
        )
        registry = ToolRegistry()
        registered = await register_mcp_tools(registry, client)  # type: ignore[arg-type]

        assert registered == ["read_file"]
        spec = registry.get("read_file")
        assert spec is not None
        # Schema is passed through unchanged — no format conversion.
        assert spec.input_schema["required"] == ["path"]
        assert spec.input_schema["properties"]["path"]["type"] == "string"
        assert spec.description == "Read a file from the sandbox."

    asyncio.run(scenario())


def test_register_mcp_tools_with_prefix_namespaces_names() -> None:
    async def scenario() -> None:
        client = _FakeMCPClient(
            tools=[
                {"name": "read_file", "description": "", "inputSchema": {"type": "object"}}
            ]
        )
        registry = ToolRegistry()
        registered = await register_mcp_tools(registry, client, prefix="fs")  # type: ignore[arg-type]

        assert registered == ["fs.read_file"]
        assert registry.get("fs.read_file") is not None
        assert registry.get("read_file") is None

    asyncio.run(scenario())


def test_handler_calls_mcp_client_and_returns_text_via_registry() -> None:
    async def scenario() -> None:
        client = _FakeMCPClient(
            tools=[
                {
                    "name": "read_file",
                    "description": "Read",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
            call_result={
                "content": [{"type": "text", "text": "file body"}],
                "isError": False,
            },
        )
        registry = ToolRegistry()
        await register_mcp_tools(registry, client)  # type: ignore[arg-type]

        result = await registry.execute("read_file", {"path": "/tmp/x.txt"})
        assert result.success is True
        assert result.content == "file body"
        assert client.call_log == [("read_file", {"path": "/tmp/x.txt"})]

    asyncio.run(scenario())


def test_handler_error_surfaces_as_tool_result_failure() -> None:
    async def scenario() -> None:
        client = _FakeMCPClient(
            tools=[
                {
                    "name": "read_file",
                    "description": "Read",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
            call_result={
                "content": [{"type": "text", "text": "ENOENT"}],
                "isError": True,
            },
        )
        registry = ToolRegistry()
        await register_mcp_tools(registry, client)  # type: ignore[arg-type]

        result = await registry.execute("read_file", {"path": "/missing"})
        assert result.success is False
        assert result.error is not None
        assert "ENOENT" in result.error

    asyncio.run(scenario())


def test_duplicate_registration_is_skipped_not_fatal() -> None:
    async def scenario() -> None:
        client = _FakeMCPClient(
            tools=[
                {"name": "read_file", "description": "", "inputSchema": {"type": "object"}},
                {"name": "read_file", "description": "", "inputSchema": {"type": "object"}},
            ]
        )
        registry = ToolRegistry()
        registered = await register_mcp_tools(registry, client)  # type: ignore[arg-type]

        # First wins; second is skipped with a warning (not raised).
        assert registered == ["read_file"]

    asyncio.run(scenario())


def test_register_mcp_tools_attaches_group_and_allowed_agents() -> None:
    async def scenario() -> None:
        client = _FakeMCPClient(
            tools=[
                {"name": "read_file", "description": "Read a file", "inputSchema": {"type": "object"}},
                {"name": "write_file", "description": "Write a file", "inputSchema": {"type": "object"}},
            ]
        )
        client.server_name = "filesystem"
        registry = ToolRegistry()
        await register_mcp_tools(  # type: ignore[arg-type]
            registry, client, allowed_agents=["researcher", "reader"]
        )

        spec_read = registry.get("read_file")
        spec_write = registry.get("write_file")
        assert spec_read is not None and spec_write is not None
        # group carries the server name; allowed_agents is attached verbatim.
        assert spec_read.group == "filesystem"
        assert spec_read.allowed_agents == ["researcher", "reader"]
        assert spec_write.group == "filesystem"
        assert spec_write.allowed_agents == ["researcher", "reader"]
        # is_agent_allowed enforces the list.
        assert spec_read.is_agent_allowed("researcher") is True
        assert spec_read.is_agent_allowed("writer") is False


    asyncio.run(scenario())


def test_register_mcp_tools_with_empty_allowed_agents_means_any() -> None:
    async def scenario() -> None:
        client = _FakeMCPClient(
            tools=[{"name": "echo_mcp", "description": "", "inputSchema": {"type": "object"}}]
        )
        registry = ToolRegistry()
        await register_mcp_tools(registry, client)  # type: ignore[arg-type]

        spec = registry.get("echo_mcp")
        assert spec is not None
        assert spec.allowed_agents == []
        assert spec.is_agent_allowed("anyone") is True

    asyncio.run(scenario())


def test_empty_name_tools_are_ignored() -> None:
    async def scenario() -> None:
        client = _FakeMCPClient(
            tools=[
                {"name": "", "description": "", "inputSchema": {}},
                {"name": "valid_tool", "description": "", "inputSchema": {"type": "object"}},
            ]
        )
        registry = ToolRegistry()
        registered = await register_mcp_tools(registry, client)  # type: ignore[arg-type]

        assert registered == ["valid_tool"]

    asyncio.run(scenario())
