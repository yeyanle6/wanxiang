"""Unit tests for the MCP stdio client.

Uses an in-memory transport to avoid spawning real subprocesses.
Real end-to-end validation against a filesystem MCP server lands in
Phase 3C.2 together with the ToolRegistry bridge.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from wanxiang.core.mcp_client import (
    MCPClient,
    MCPError,
    MCPTransport,
)


class MemoryTransport(MCPTransport):
    """Scripted in-memory transport for testing JSON-RPC flows."""

    def __init__(self) -> None:
        self._outbox: list[dict] = []
        self._inbox: asyncio.Queue[bytes] = asyncio.Queue()
        self._closed = False

    @property
    def sent(self) -> list[dict]:
        return list(self._outbox)

    def enqueue_server_message(self, payload: dict) -> None:
        line = (json.dumps(payload) + "\n").encode("utf-8")
        self._inbox.put_nowait(line)

    def enqueue_raw(self, raw: bytes) -> None:
        self._inbox.put_nowait(raw)

    def enqueue_eof(self) -> None:
        self._inbox.put_nowait(b"")

    async def send(self, line: bytes) -> None:
        if self._closed:
            raise RuntimeError("transport closed")
        text = line.decode("utf-8").strip()
        self._outbox.append(json.loads(text))

    async def readline(self) -> bytes:
        return await self._inbox.get()

    async def close(self) -> None:
        self._closed = True
        # Unblock any pending readline so the reader task can exit cleanly.
        self.enqueue_eof()


def _sent_request(transport: MemoryTransport, method: str) -> dict:
    for msg in transport.sent:
        if msg.get("method") == method and "id" in msg:
            return msg
    raise AssertionError(f"No request with method={method} was sent. sent={transport.sent}")


def _sent_notification(transport: MemoryTransport, method: str) -> dict:
    for msg in transport.sent:
        if msg.get("method") == method and "id" not in msg:
            return msg
    raise AssertionError(f"No notification with method={method} was sent. sent={transport.sent}")


@pytest.fixture
def transport() -> MemoryTransport:
    return MemoryTransport()


def test_initialize_handshake_sends_correct_payload_and_sets_server_info() -> None:
    async def scenario() -> None:
        t = MemoryTransport()
        client = MCPClient(t, server_name="fake", request_timeout_s=1.0)

        # Pre-arm server: respond to initialize then let initialized notification pass.
        async def drive() -> None:
            # Wait until client sent the initialize request, then respond.
            while not any(m.get("method") == "initialize" for m in t.sent):
                await asyncio.sleep(0)
            init_msg = _sent_request(t, "initialize")
            t.enqueue_server_message(
                {
                    "jsonrpc": "2.0",
                    "id": init_msg["id"],
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake-mcp", "version": "0.9.0"},
                    },
                }
            )

        driver = asyncio.create_task(drive())
        await client.start()
        await driver

        init_req = _sent_request(t, "initialize")
        params = init_req["params"]
        assert params["clientInfo"]["name"] == "wanxiang"
        assert params["clientInfo"]["version"] == "0.1.0"
        assert params["protocolVersion"] == "2024-11-05"
        _sent_notification(t, "notifications/initialized")

        assert client.server_info == {"name": "fake-mcp", "version": "0.9.0"}
        assert client.server_capabilities == {"tools": {}}

        await client.close()

    asyncio.run(scenario())


def test_list_tools_returns_server_tools() -> None:
    async def scenario() -> None:
        t = MemoryTransport()
        client = MCPClient(t, server_name="fake", request_timeout_s=1.0)

        async def drive() -> None:
            while not any(m.get("method") == "initialize" for m in t.sent):
                await asyncio.sleep(0)
            init_msg = _sent_request(t, "initialize")
            t.enqueue_server_message(
                {"jsonrpc": "2.0", "id": init_msg["id"], "result": {}}
            )
            while not any(m.get("method") == "tools/list" for m in t.sent):
                await asyncio.sleep(0)
            list_msg = _sent_request(t, "tools/list")
            t.enqueue_server_message(
                {
                    "jsonrpc": "2.0",
                    "id": list_msg["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "read_file",
                                "description": "Read a file",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"path": {"type": "string"}},
                                    "required": ["path"],
                                },
                            }
                        ]
                    },
                }
            )

        driver = asyncio.create_task(drive())
        await client.start()
        tools = await client.list_tools()
        await driver

        assert len(tools) == 1
        assert tools[0]["name"] == "read_file"
        assert tools[0]["inputSchema"]["required"] == ["path"]

        await client.close()

    asyncio.run(scenario())


def test_call_tool_roundtrip_returns_result() -> None:
    async def scenario() -> None:
        t = MemoryTransport()
        client = MCPClient(t, server_name="fake", request_timeout_s=1.0)

        async def drive() -> None:
            while not any(m.get("method") == "initialize" for m in t.sent):
                await asyncio.sleep(0)
            init_msg = _sent_request(t, "initialize")
            t.enqueue_server_message(
                {"jsonrpc": "2.0", "id": init_msg["id"], "result": {}}
            )
            while not any(m.get("method") == "tools/call" for m in t.sent):
                await asyncio.sleep(0)
            call_msg = _sent_request(t, "tools/call")
            assert call_msg["params"]["name"] == "read_file"
            assert call_msg["params"]["arguments"] == {"path": "/tmp/x.txt"}
            t.enqueue_server_message(
                {
                    "jsonrpc": "2.0",
                    "id": call_msg["id"],
                    "result": {
                        "content": [{"type": "text", "text": "hello"}],
                        "isError": False,
                    },
                }
            )

        driver = asyncio.create_task(drive())
        await client.start()
        result = await client.call_tool("read_file", {"path": "/tmp/x.txt"})
        await driver

        assert result["isError"] is False
        assert result["content"][0]["text"] == "hello"

        await client.close()

    asyncio.run(scenario())


def test_error_response_raises_mcp_error() -> None:
    async def scenario() -> None:
        t = MemoryTransport()
        client = MCPClient(t, server_name="fake", request_timeout_s=1.0)

        async def drive() -> None:
            while not any(m.get("method") == "initialize" for m in t.sent):
                await asyncio.sleep(0)
            init_msg = _sent_request(t, "initialize")
            t.enqueue_server_message(
                {"jsonrpc": "2.0", "id": init_msg["id"], "result": {}}
            )
            while not any(m.get("method") == "tools/call" for m in t.sent):
                await asyncio.sleep(0)
            call_msg = _sent_request(t, "tools/call")
            t.enqueue_server_message(
                {
                    "jsonrpc": "2.0",
                    "id": call_msg["id"],
                    "error": {
                        "code": -32602,
                        "message": "Invalid params",
                        "data": {"field": "path"},
                    },
                }
            )

        driver = asyncio.create_task(drive())
        await client.start()

        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("read_file", {})
        assert excinfo.value.code == -32602
        assert "Invalid params" in str(excinfo.value)
        assert excinfo.value.data == {"field": "path"}
        await driver
        await client.close()

    asyncio.run(scenario())


def test_request_times_out_when_server_is_silent() -> None:
    async def scenario() -> None:
        t = MemoryTransport()
        client = MCPClient(t, server_name="fake", request_timeout_s=0.15)

        async def drive_init() -> None:
            while not any(m.get("method") == "initialize" for m in t.sent):
                await asyncio.sleep(0)
            init_msg = _sent_request(t, "initialize")
            t.enqueue_server_message(
                {"jsonrpc": "2.0", "id": init_msg["id"], "result": {}}
            )

        driver = asyncio.create_task(drive_init())
        await client.start()
        await driver

        with pytest.raises(TimeoutError):
            await client.call_tool("read_file", {"path": "/x"})

        await client.close()

    asyncio.run(scenario())


def test_close_cancels_pending_requests() -> None:
    async def scenario() -> None:
        t = MemoryTransport()
        client = MCPClient(t, server_name="fake", request_timeout_s=5.0)

        async def drive_init() -> None:
            while not any(m.get("method") == "initialize" for m in t.sent):
                await asyncio.sleep(0)
            init_msg = _sent_request(t, "initialize")
            t.enqueue_server_message(
                {"jsonrpc": "2.0", "id": init_msg["id"], "result": {}}
            )

        driver = asyncio.create_task(drive_init())
        await client.start()
        await driver

        call_task = asyncio.create_task(client.call_tool("read_file", {"path": "/x"}))
        # Give the request a moment to register.
        await asyncio.sleep(0.01)
        await client.close()

        with pytest.raises(RuntimeError) as excinfo:
            await call_task
        assert "closed" in str(excinfo.value).lower()

    asyncio.run(scenario())


def test_server_eof_fails_pending_requests() -> None:
    async def scenario() -> None:
        t = MemoryTransport()
        client = MCPClient(t, server_name="fake", request_timeout_s=5.0)

        async def drive() -> None:
            while not any(m.get("method") == "initialize" for m in t.sent):
                await asyncio.sleep(0)
            init_msg = _sent_request(t, "initialize")
            t.enqueue_server_message(
                {"jsonrpc": "2.0", "id": init_msg["id"], "result": {}}
            )
            while not any(m.get("method") == "tools/call" for m in t.sent):
                await asyncio.sleep(0)
            # Simulate server crash mid-request.
            t.enqueue_eof()

        driver = asyncio.create_task(drive())
        await client.start()

        with pytest.raises(RuntimeError) as excinfo:
            await client.call_tool("read_file", {"path": "/x"})
        assert "closed the connection" in str(excinfo.value)
        await driver
        await client.close()

    asyncio.run(scenario())


def test_ignores_malformed_json_lines() -> None:
    async def scenario() -> None:
        t = MemoryTransport()
        client = MCPClient(t, server_name="fake", request_timeout_s=1.0)

        async def drive() -> None:
            while not any(m.get("method") == "initialize" for m in t.sent):
                await asyncio.sleep(0)
            # Inject garbage line first; client must keep reading and not crash.
            t.enqueue_raw(b"this-is-not-json\n")
            init_msg = _sent_request(t, "initialize")
            t.enqueue_server_message(
                {"jsonrpc": "2.0", "id": init_msg["id"], "result": {}}
            )

        driver = asyncio.create_task(drive())
        await client.start()
        await driver
        await client.close()

    asyncio.run(scenario())
