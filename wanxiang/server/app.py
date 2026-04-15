from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from ..core.builtin_tools import create_default_registry
from ..core.mcp_loader import MCPPool, load_mcp_declarations
from .mcp_status import probe_mcp_status
from .models import (
    MCPStatusResponse,
    RunDetailResponse,
    RunListResponse,
    RunRequest,
    RunResponse,
)
from .runner import RunManager


logger = logging.getLogger("wanxiang.server")

UI_FILE = Path(__file__).resolve().parents[2] / "wanxiang-ui.jsx"
MCP_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "mcp.yaml"

_run_manager: RunManager | None = None
_mcp_pool: MCPPool | None = None


def _get_run_manager() -> RunManager:
    if _run_manager is None:
        raise HTTPException(status_code=503, detail="Server not yet initialized.")
    return _run_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _run_manager, _mcp_pool
    tool_registry = create_default_registry()
    _mcp_pool = MCPPool()

    if MCP_CONFIG_PATH.exists():
        try:
            declarations = load_mcp_declarations(MCP_CONFIG_PATH)
            registered = await _mcp_pool.start(declarations, tool_registry)
            for server_name, names in registered.items():
                logger.info(
                    "MCP server '%s' registered %d tools: %s",
                    server_name,
                    len(names),
                    names,
                )
        except Exception:
            logger.exception("MCP pool startup failed; continuing without MCP tools")
    else:
        logger.info(
            "No MCP config at %s; starting without external MCP servers",
            MCP_CONFIG_PATH,
        )

    _run_manager = RunManager(
        llm_mode=os.getenv("WANXIANG_LLM_MODE"),
        tool_registry=tool_registry,
    )

    try:
        yield
    finally:
        if _mcp_pool is not None:
            await _mcp_pool.close()
        _run_manager = None
        _mcp_pool = None


app = FastAPI(title="Wanxiang Server", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Wanxiang UI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  </head>
  <body class="bg-slate-100">
    <div id="app"></div>
    <script type="text/babel" data-presets="react" src="/ui/wanxiang-ui.jsx"></script>
  </body>
</html>"""


@app.get("/ui/wanxiang-ui.jsx")
async def ui_script() -> FileResponse:
    if not UI_FILE.exists():
        raise HTTPException(status_code=404, detail=f"UI file not found: {UI_FILE}")
    return FileResponse(UI_FILE, media_type="application/javascript; charset=utf-8")


@app.post("/api/runs", response_model=RunResponse)
async def create_run(request: RunRequest) -> RunResponse:
    rm = _get_run_manager()
    try:
        run_id = await rm.start_run(request.task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RunResponse(run_id=run_id)


@app.get("/api/runs", response_model=RunListResponse)
async def list_runs(limit: int = Query(default=10, ge=1, le=100)) -> RunListResponse:
    rm = _get_run_manager()
    summaries = await rm.list_runs(limit=limit)
    return RunListResponse(runs=summaries)


@app.get("/api/runs/{run_id}", response_model=RunDetailResponse)
async def get_run(run_id: str) -> RunDetailResponse:
    rm = _get_run_manager()
    run = await rm.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return RunDetailResponse(**run)


@app.get("/api/mcp/status", response_model=MCPStatusResponse)
async def get_mcp_status() -> MCPStatusResponse:
    status = await probe_mcp_status()
    return MCPStatusResponse(**status)


@app.get("/api/mcp/wanxiang-pool")
async def get_mcp_pool_status() -> dict:
    """Report which MCP servers Wanxiang itself spawned and their tools."""
    if _mcp_pool is None:
        return {"ready": False, "servers": {}}
    return {
        "ready": True,
        "servers": _mcp_pool.registered_tools,
    }


@app.websocket("/api/runs/{run_id}/events")
async def run_events(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    rm = _run_manager
    if rm is None:
        await websocket.close(code=4503, reason="Server not yet initialized.")
        return
    if not rm.has_run(run_id):
        await websocket.close(code=4404, reason=f"Run not found: {run_id}")
        return

    try:
        async for event in rm.stream_events(run_id):
            await websocket.send_text(event.to_json())
    except WebSocketDisconnect:
        return


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Launch the Wanxiang web server.")
    parser.add_argument("--host", default=os.getenv("WANXIANG_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WANXIANG_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only).")
    parser.add_argument(
        "--log-level",
        default=os.getenv("WANXIANG_LOG_LEVEL", "INFO"),
        help="Log level for wanxiang.* loggers (default: INFO).",
    )
    args = parser.parse_args()

    # Route wanxiang.* logger output through uvicorn's handlers so lifespan
    # logs (MCP registration, etc.) are visible on the server console.
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    wanxiang_logger = logging.getLogger("wanxiang")
    wanxiang_logger.setLevel(level)
    if not wanxiang_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        wanxiang_logger.addHandler(handler)
        wanxiang_logger.propagate = False

    uvicorn.run(
        "wanxiang.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
