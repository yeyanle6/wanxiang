"""Tests for the YAML loader that drives MCP server spawning."""
from __future__ import annotations

from pathlib import Path

import pytest

from wanxiang.core.mcp_loader import load_mcp_declarations


def _write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "mcp.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_minimal_server_declaration(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
servers:
  - name: filesystem
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
""",
    )
    decls = load_mcp_declarations(path)
    assert len(decls) == 1
    decl = decls[0]
    assert decl.config.name == "filesystem"
    assert decl.config.command == "npx"
    assert decl.config.args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    assert decl.config.timeout_s == 30.0
    assert decl.prefix == ""
    assert decl.allowed_agents == []


def test_load_full_server_declaration_with_all_fields(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
servers:
  - name: filesystem
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    env:
      MCP_LOG_LEVEL: debug
    timeout_s: 60
    cwd: /tmp
    prefix: fs
    allowed_agents: ["researcher", "reader"]
""",
    )
    decls = load_mcp_declarations(path)
    decl = decls[0]
    assert decl.config.env == {"MCP_LOG_LEVEL": "debug"}
    assert decl.config.timeout_s == 60.0
    assert decl.config.cwd == "/tmp"
    assert decl.prefix == "fs"
    assert decl.allowed_agents == ["researcher", "reader"]


def test_load_multiple_servers(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
servers:
  - name: a
    command: cmd_a
  - name: b
    command: cmd_b
""",
    )
    decls = load_mcp_declarations(path)
    assert [d.config.name for d in decls] == ["a", "b"]


def test_empty_servers_list_is_valid(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "servers: []\n")
    assert load_mcp_declarations(path) == []


def test_missing_name_raises(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
servers:
  - command: npx
""",
    )
    with pytest.raises(ValueError, match="missing required 'name'"):
        load_mcp_declarations(path)


def test_missing_command_raises(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
servers:
  - name: filesystem
""",
    )
    with pytest.raises(ValueError, match="missing required 'command'"):
        load_mcp_declarations(path)


def test_non_positive_timeout_raises(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
servers:
  - name: filesystem
    command: npx
    timeout_s: 0
""",
    )
    with pytest.raises(ValueError, match="timeout_s"):
        load_mcp_declarations(path)


def test_args_must_be_a_list(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
servers:
  - name: filesystem
    command: npx
    args: "not-a-list"
""",
    )
    with pytest.raises(ValueError, match=r"args must be a list"):
        load_mcp_declarations(path)


def test_env_must_be_an_object(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
servers:
  - name: filesystem
    command: npx
    env: "not-an-object"
""",
    )
    with pytest.raises(ValueError, match=r"env must be an object"):
        load_mcp_declarations(path)


def test_servers_must_be_a_list(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "servers: {}\n")
    with pytest.raises(ValueError, match="'servers' must be a list"):
        load_mcp_declarations(path)
