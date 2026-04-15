"""MCP (Model Context Protocol) stdio client.

Minimal async client that speaks JSON-RPC 2.0 over stdio to an external
MCP server process. The client is transport-agnostic at the protocol
layer — the stdio wrapping is a thin adapter over asyncio.subprocess.

Supported methods (others are out of scope for Phase 3C.1):
- initialize / notifications/initialized  (handshake)
- tools/list
- tools/call
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol


DEFAULT_PROTOCOL_VERSION = "2024-11-05"
DEFAULT_CLIENT_NAME = "wanxiang"
DEFAULT_CLIENT_VERSION = "0.1.0"


@dataclass(slots=True)
class MCPServerConfig:
    """Launch spec for a stdio-based MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 30.0
    cwd: str | None = None


class MCPError(Exception):
    """Raised when the MCP server returns a JSON-RPC error response."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"MCP error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class MCPTransport(Protocol):
    """Transport interface. A send/recv pair plus close."""

    async def send(self, line: bytes) -> None: ...

    async def readline(self) -> bytes: ...

    async def close(self) -> None: ...


class StdioTransport:
    """Wraps an asyncio.subprocess.Process as an MCPTransport."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process

    @classmethod
    async def spawn(cls, config: MCPServerConfig) -> StdioTransport:
        env = os.environ.copy()
        env.update(config.env)
        process = await asyncio.create_subprocess_exec(
            config.command,
            *config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=config.cwd,
        )
        return cls(process)

    async def send(self, line: bytes) -> None:
        stdin = self._process.stdin
        if stdin is None:
            raise RuntimeError("Transport stdin is not available.")
        stdin.write(line)
        await stdin.drain()

    async def readline(self) -> bytes:
        stdout = self._process.stdout
        if stdout is None:
            raise RuntimeError("Transport stdout is not available.")
        return await stdout.readline()

    async def close(self) -> None:
        if self._process.returncode is not None:
            return
        try:
            if self._process.stdin is not None:
                self._process.stdin.close()
        except Exception:
            pass
        try:
            self._process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            try:
                self._process.kill()
            except ProcessLookupError:
                return
            await self._process.wait()


class MCPClient:
    """JSON-RPC 2.0 MCP client. Transport-agnostic."""

    def __init__(
        self,
        transport: MCPTransport,
        *,
        server_name: str,
        request_timeout_s: float = 30.0,
        client_name: str = DEFAULT_CLIENT_NAME,
        client_version: str = DEFAULT_CLIENT_VERSION,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    ) -> None:
        self._transport = transport
        self.server_name = server_name
        self.request_timeout_s = request_timeout_s
        self.client_name = client_name
        self.client_version = client_version
        self.protocol_version = protocol_version
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._id_counter = 0
        self._id_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._closed = False
        self._server_capabilities: dict[str, Any] = {}
        self._server_info: dict[str, Any] = {}
        self.logger = logging.getLogger(f"wanxiang.mcp.{server_name}")

    @property
    def server_capabilities(self) -> dict[str, Any]:
        return dict(self._server_capabilities)

    @property
    def server_info(self) -> dict[str, Any]:
        return dict(self._server_info)

    async def start(self) -> None:
        if self._reader_task is not None:
            return
        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"mcp-reader-{self.server_name}"
        )
        result = await self._request(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version,
                },
            },
        )
        if isinstance(result, dict):
            self._server_capabilities = dict(result.get("capabilities") or {})
            self._server_info = dict(result.get("serverInfo") or {})
        await self._notify("notifications/initialized")

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._request("tools/list")
        if not isinstance(result, dict):
            return []
        tools = result.get("tools")
        return list(tools) if isinstance(tools, list) else []

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": name}
        payload["arguments"] = arguments or {}
        result = await self._request("tools/call", payload)
        return result if isinstance(result, dict) else {}

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None
        for future in self._pending.values():
            if not future.done():
                future.set_exception(
                    RuntimeError(f"MCP client for {self.server_name} was closed.")
                )
        self._pending.clear()
        try:
            await self._transport.close()
        except Exception:  # pragma: no cover - defensive
            self.logger.exception("Transport close failed for %s", self.server_name)

    async def __aenter__(self) -> MCPClient:
        await self.start()
        return self

    async def __aexit__(self, *_exc_info: Any) -> None:
        await self.close()

    async def _request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> Any:
        if self._closed:
            raise RuntimeError(f"MCP client for {self.server_name} is closed.")
        async with self._id_lock:
            self._id_counter += 1
            request_id = self._id_counter
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        try:
            await self._transport.send(self._encode(payload))
        except Exception:
            self._pending.pop(request_id, None)
            raise
        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout_s)
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise TimeoutError(
                f"MCP request '{method}' timed out after {self.request_timeout_s:.1f}s"
            ) from exc

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        await self._transport.send(self._encode(payload))

    async def _read_loop(self) -> None:
        try:
            while True:
                line = await self._transport.readline()
                if not line:
                    # Transport EOF — fail all pending requests.
                    self._fail_pending(
                        RuntimeError(
                            f"MCP server {self.server_name} closed the connection."
                        )
                    )
                    return
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    self.logger.warning(
                        "Non-JSON output from %s: %s", self.server_name, text[:200]
                    )
                    continue
                if not isinstance(message, dict):
                    continue
                self._dispatch(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.exception(
                "MCP read loop crashed for %s", self.server_name
            )
            self._fail_pending(exc)

    def _dispatch(self, message: dict[str, Any]) -> None:
        msg_id = message.get("id")
        if msg_id is not None and msg_id in self._pending:
            future = self._pending.pop(msg_id)
            if future.done():
                return
            error = message.get("error")
            if isinstance(error, dict):
                future.set_exception(
                    MCPError(
                        code=int(error.get("code", -1)),
                        message=str(error.get("message", "")),
                        data=error.get("data"),
                    )
                )
            else:
                future.set_result(message.get("result"))
            return
        # Unmatched — either a server-initiated request (we ignore) or a
        # notification. Log at debug level for visibility.
        self.logger.debug("Unhandled MCP message from %s: %s", self.server_name, message)

    def _fail_pending(self, exc: BaseException) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()

    @staticmethod
    def _encode(payload: dict[str, Any]) -> bytes:
        return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


async def launch_stdio_mcp_client(config: MCPServerConfig) -> MCPClient:
    """Spawn the MCP server process and return a started client."""
    transport = await StdioTransport.spawn(config)
    client = MCPClient(
        transport,
        server_name=config.name,
        request_timeout_s=config.timeout_s,
    )
    try:
        await client.start()
    except Exception:
        await client.close()
        raise
    return client
