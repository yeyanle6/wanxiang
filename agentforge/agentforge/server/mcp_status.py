from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass


@dataclass(slots=True)
class _CommandResult:
    returncode: int
    stdout: str
    stderr: str


async def _run_command(*args: str, timeout_s: float = 8.0) -> _CommandResult:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_raw, stderr_raw = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        return _CommandResult(returncode=124, stdout="", stderr=f"timeout after {timeout_s:.1f}s")

    return _CommandResult(
        returncode=int(process.returncode or 0),
        stdout=stdout_raw.decode("utf-8", errors="replace").strip(),
        stderr=stderr_raw.decode("utf-8", errors="replace").strip(),
    )


def parse_mcp_list_output(output: str) -> list[dict[str, object]]:
    servers: list[dict[str, object]] = []
    for raw_line in str(output).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("checking mcp server health"):
            continue
        if " - " not in line:
            continue
        name, status_part = line.rsplit(" - ", 1)
        status_text = status_part.strip()
        connected = "connected" in status_text.lower() and "not connected" not in status_text.lower()
        servers.append(
            {
                "name": name.strip(),
                "status": status_text,
                "connected": connected,
            }
        )
    return servers


async def probe_mcp_status() -> dict[str, object]:
    claude_bin = shutil.which("claude")
    if not claude_bin:
        return {
            "ready": False,
            "claude_installed": False,
            "logged_in": False,
            "auth_method": "unknown",
            "api_provider": "unknown",
            "servers": [],
            "connected_servers": 0,
            "errors": ["`claude` binary not found in PATH."],
        }

    errors: list[str] = []
    logged_in = False
    auth_method = "unknown"
    api_provider = "unknown"

    auth_result = await _run_command(claude_bin, "auth", "status")
    parsed_auth = False
    if auth_result.stdout:
        try:
            payload = json.loads(auth_result.stdout)
            logged_in = bool(payload.get("loggedIn", False))
            auth_method = str(payload.get("authMethod", "unknown"))
            api_provider = str(payload.get("apiProvider", "unknown"))
            parsed_auth = True
        except Exception:
            parsed_auth = False

    if not parsed_auth:
        if auth_result.returncode != 0:
            errors.append(
                auth_result.stderr or auth_result.stdout or "failed to run `claude auth status`"
            )
        else:
            errors.append("unable to parse `claude auth status` output")

    servers: list[dict[str, object]] = []
    mcp_list_result = await _run_command(claude_bin, "mcp", "list")
    if mcp_list_result.returncode != 0:
        errors.append(mcp_list_result.stderr or mcp_list_result.stdout or "failed to run `claude mcp list`")
    else:
        servers = parse_mcp_list_output(mcp_list_result.stdout)

    connected_servers = sum(1 for item in servers if bool(item.get("connected")))
    ready = bool(logged_in and connected_servers >= 1)
    return {
        "ready": ready,
        "claude_installed": True,
        "logged_in": logged_in,
        "auth_method": auth_method,
        "api_provider": api_provider,
        "servers": servers,
        "connected_servers": connected_servers,
        "errors": errors,
    }
