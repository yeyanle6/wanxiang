"""Load MCP server declarations from YAML and manage their lifecycle.

The loader keeps startup and shutdown together: `MCPPool.start()` spawns
every declared server + registers its tools; `MCPPool.close()` tears
them all down. Bind this to the FastAPI lifespan context so children
can't leak.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .mcp_bridge import register_mcp_tools
from .mcp_client import MCPClient, MCPServerConfig, launch_stdio_mcp_client
from .tools import ToolRegistry


logger = logging.getLogger("wanxiang.mcp.loader")


@dataclass(slots=True)
class MCPServerDeclaration:
    """YAML-level MCP server declaration.

    Wraps the client-side launch spec (`config`) plus loader metadata.
    `allowed_agents` is parsed and stored so future policy layers can
    restrict tool visibility per-agent; it is not enforced in Phase 3C.2.
    """

    config: MCPServerConfig
    prefix: str = ""
    allowed_agents: list[str] = field(default_factory=list)


def load_mcp_declarations(path: str | Path) -> list[MCPServerDeclaration]:
    """Parse `mcp.yaml` into structured declarations.

    Expected schema::

        servers:
          - name: filesystem
            command: npx
            args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/ws"]
            env: {}                    # optional
            timeout_s: 30              # optional, default 30
            cwd: null                  # optional
            prefix: ""                 # optional; empty = use raw tool names
            allowed_agents: []         # optional; empty = all agents allowed
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load mcp.yaml") from exc

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: root value must be a mapping")

    raw_servers = raw.get("servers", [])
    if not isinstance(raw_servers, list):
        raise ValueError(f"{path}: 'servers' must be a list")

    declarations: list[MCPServerDeclaration] = []
    for idx, entry in enumerate(raw_servers):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: servers[{idx}] must be an object")
        name = str(entry.get("name", "")).strip()
        command = str(entry.get("command", "")).strip()
        if not name:
            raise ValueError(f"{path}: servers[{idx}] missing required 'name'")
        if not command:
            raise ValueError(f"{path}: servers[{idx}] missing required 'command'")

        raw_args = entry.get("args") or []
        if not isinstance(raw_args, list):
            raise ValueError(f"{path}: servers[{idx}].args must be a list")

        raw_env = entry.get("env") or {}
        if not isinstance(raw_env, dict):
            raise ValueError(f"{path}: servers[{idx}].env must be an object")

        timeout_s = float(entry.get("timeout_s", 30.0))
        if timeout_s <= 0:
            raise ValueError(f"{path}: servers[{idx}].timeout_s must be > 0")

        cwd_val = entry.get("cwd")
        cwd = str(cwd_val) if cwd_val is not None else None
        prefix = str(entry.get("prefix", "")).strip()

        raw_allowed = entry.get("allowed_agents") or []
        if not isinstance(raw_allowed, list):
            raise ValueError(
                f"{path}: servers[{idx}].allowed_agents must be a list"
            )
        allowed_agents = [
            str(a).strip() for a in raw_allowed if str(a).strip()
        ]

        declarations.append(
            MCPServerDeclaration(
                config=MCPServerConfig(
                    name=name,
                    command=command,
                    args=[str(a) for a in raw_args],
                    env={str(k): str(v) for k, v in raw_env.items()},
                    timeout_s=timeout_s,
                    cwd=cwd,
                ),
                prefix=prefix,
                allowed_agents=allowed_agents,
            )
        )
    return declarations


class MCPPool:
    """Owns a set of running MCP clients and their registered tools."""

    def __init__(self) -> None:
        self._clients: list[MCPClient] = []
        self._registered: dict[str, list[str]] = {}

    @property
    def clients(self) -> list[MCPClient]:
        return list(self._clients)

    @property
    def registered_tools(self) -> dict[str, list[str]]:
        """Map of server_name -> list of registered tool names."""
        return {k: list(v) for k, v in self._registered.items()}

    async def start(
        self,
        declarations: list[MCPServerDeclaration],
        registry: ToolRegistry,
    ) -> dict[str, list[str]]:
        """Spawn each declared server and register its tools.

        Failures are logged and skipped — one broken server does not
        prevent the rest from starting. Returns the same map as
        `registered_tools`.
        """
        for decl in declarations:
            server_name = decl.config.name
            try:
                client = await launch_stdio_mcp_client(decl.config)
            except Exception:
                logger.exception(
                    "Failed to launch MCP server '%s'; skipping", server_name
                )
                continue
            self._clients.append(client)
            try:
                names = await register_mcp_tools(
                    registry,
                    client,
                    prefix=decl.prefix,
                    timeout_s=decl.config.timeout_s,
                    allowed_agents=decl.allowed_agents,
                )
            except Exception:
                logger.exception(
                    "Failed to list/register tools for MCP server '%s'; "
                    "the client stays up but serves no tools",
                    server_name,
                )
                names = []
            self._registered[server_name] = names
        return self.registered_tools

    async def close(self) -> None:
        """Terminate every client process. Idempotent."""
        for client in self._clients:
            try:
                await client.close()
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "Failed to close MCP client '%s'", client.server_name
                )
        self._clients.clear()
        self._registered.clear()
