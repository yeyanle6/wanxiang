"""Ad-hoc end-to-end MCP verification (not part of pytest).

Spawns the real @modelcontextprotocol/server-filesystem against
/tmp/wanxiang-test, lists tools, reads demo.txt via the registry,
then shuts everything down cleanly.

Run: python tests/_e2e_mcp_filesystem.py
"""
from __future__ import annotations

import asyncio
import sys

from wanxiang.core.mcp_bridge import register_mcp_tools
from wanxiang.core.mcp_client import MCPServerConfig, launch_stdio_mcp_client
from wanxiang.core.tools import ToolRegistry


async def main() -> int:
    config = MCPServerConfig(
        name="filesystem",
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/tmp/wanxiang-test",
        ],
        timeout_s=60.0,
    )
    print("[e2e] spawning MCP server...")
    client = await launch_stdio_mcp_client(config)
    try:
        print(f"[e2e] server_info = {client.server_info}")
        print(f"[e2e] capabilities = {client.server_capabilities}")

        registry = ToolRegistry()
        names = await register_mcp_tools(registry, client)
        print(f"[e2e] registered tools ({len(names)}): {names}")

        # Try the canonical read_text_file first; fall back to read_file if that's what the server advertises.
        candidate = None
        for preferred in ("read_text_file", "read_file"):
            if preferred in names:
                candidate = preferred
                break
        if candidate is None:
            print(f"[e2e] WARN: neither read_text_file nor read_file found in tools; falling back to first tool: {names[0] if names else 'NONE'}")
            return 2

        print(f"[e2e] calling {candidate}(path=/tmp/wanxiang-test/demo.txt)...")
        result = await registry.execute(
            candidate, {"path": "/tmp/wanxiang-test/demo.txt"}
        )
        print(f"[e2e] success={result.success} elapsed_ms={result.elapsed_ms}")
        if result.success:
            print("[e2e] content preview:")
            print("----")
            print(result.content)
            print("----")
            return 0
        else:
            print(f"[e2e] ERROR: {result.error}")
            return 1
    finally:
        print("[e2e] closing client...")
        await client.close()
        print("[e2e] done.")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
