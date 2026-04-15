from agentforge.server.mcp_status import parse_mcp_list_output


def test_parse_mcp_list_output_extracts_connected_and_disconnected_servers() -> None:
    raw = """Checking MCP server health…

plugin:oh-my-claudecode:t: node /path/to/server.cjs - ✓ Connected
local:broken-server: npx broken-mcp - ✗ Failed to connect
"""
    parsed = parse_mcp_list_output(raw)
    assert len(parsed) == 2
    assert parsed[0]["name"] == "plugin:oh-my-claudecode:t: node /path/to/server.cjs"
    assert parsed[0]["connected"] is True
    assert parsed[1]["name"] == "local:broken-server: npx broken-mcp"
    assert parsed[1]["connected"] is False

