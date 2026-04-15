from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, description="Task description for the run.")


class RunResponse(BaseModel):
    run_id: str = Field(..., description="Unique id for this run.")


class RunSummary(BaseModel):
    run_id: str
    task: str
    started_at: str
    completed_at: str
    final_status: str
    step_count: int
    event_count: int


class RunListResponse(BaseModel):
    runs: list[RunSummary]


class RunDetailResponse(BaseModel):
    run_id: str
    task: str
    started_at: str
    completed_at: str
    final_status: str
    events: list[dict[str, Any]]


class MCPServerStatus(BaseModel):
    name: str
    status: str
    connected: bool


class MCPStatusResponse(BaseModel):
    ready: bool
    claude_installed: bool
    logged_in: bool
    auth_method: str
    api_provider: str
    servers: list[MCPServerStatus]
    connected_servers: int
    errors: list[str]
