from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from .mcp_status import probe_mcp_status
from .models import (
    MCPStatusResponse,
    RunDetailResponse,
    RunListResponse,
    RunRequest,
    RunResponse,
)
from .runner import RunManager

app = FastAPI(title="Wanxiang Server", version="0.1.0")
run_manager = RunManager(llm_mode=os.getenv("WANXIANG_LLM_MODE"))
UI_FILE = Path(__file__).resolve().parents[2] / "wanxiang-ui.jsx"

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
    try:
        run_id = await run_manager.start_run(request.task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RunResponse(run_id=run_id)


@app.get("/api/runs", response_model=RunListResponse)
async def list_runs(limit: int = Query(default=10, ge=1, le=100)) -> RunListResponse:
    summaries = await run_manager.list_runs(limit=limit)
    return RunListResponse(runs=summaries)


@app.get("/api/runs/{run_id}", response_model=RunDetailResponse)
async def get_run(run_id: str) -> RunDetailResponse:
    run = await run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return RunDetailResponse(**run)


@app.get("/api/mcp/status", response_model=MCPStatusResponse)
async def get_mcp_status() -> MCPStatusResponse:
    status = await probe_mcp_status()
    return MCPStatusResponse(**status)


@app.websocket("/api/runs/{run_id}/events")
async def run_events(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    if not run_manager.has_run(run_id):
        await websocket.close(code=4404, reason=f"Run not found: {run_id}")
        return

    try:
        async for event in run_manager.stream_events(run_id):
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
    args = parser.parse_args()

    uvicorn.run(
        "wanxiang.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
