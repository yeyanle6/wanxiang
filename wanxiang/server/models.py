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


# ---- Trace mining ---------------------------------------------------------
# Response schema mirrors wanxiang.core.trace_mining.TraceMiningReport.
# Dataclasses there are the source of truth; these models just wrap the
# shape for FastAPI's OpenAPI output and incoming validation.


class FailurePatternModel(BaseModel):
    keyword: str
    count: int
    example: str
    category: str = "unknown"


class ToolUsageStatsModel(BaseModel):
    tool: str
    group: str
    calls: int
    successes: int
    failures: int
    total_elapsed_ms: int
    avg_elapsed_ms: float
    truncated_calls: int


class SynthesisStatsModel(BaseModel):
    total_requests: int
    succeeded: int
    failed: int
    avg_attempts: float
    registered_tools: list[str]


class ReviewerStatsModel(BaseModel):
    total_review_runs: int
    converged_iteration_1: int
    converged_iteration_2: int
    converged_iteration_3_plus: int
    never_converged: int
    avg_iterations: float
    max_iterations_seen: int


class AgentTimingModel(BaseModel):
    agent: str
    calls: int
    total_elapsed_ms: int
    avg_elapsed_ms: float


class TraceMiningWindowModel(BaseModel):
    after: str | None
    before: str | None
    first_run_started_at: str | None
    last_run_completed_at: str | None


class TraceMiningResponse(BaseModel):
    window: TraceMiningWindowModel
    total_runs: int
    final_status_distribution: dict[str, int]
    workflow_mix: dict[str, int]
    infra_failure_patterns: list[FailurePatternModel]
    capability_gap_patterns: list[FailurePatternModel]
    tool_usage: dict[str, ToolUsageStatsModel]
    tool_usage_by_group: dict[str, int]
    synthesis_stats: SynthesisStatsModel
    agent_naming: dict[str, int]
    slowest_agents: list[AgentTimingModel]
    reviewer_convergence: ReviewerStatsModel
    tier_changes: list[dict[str, Any]] = Field(default_factory=list)


class SynthesisCandidateModel(BaseModel):
    tool_name: str
    failure_count: int
    sample_arguments: list[dict[str, Any]]
    suggested_requirement: str


class ForgeCandidatesResponse(BaseModel):
    candidates: list[SynthesisCandidateModel]
    forge_ready: bool


class ForgeTriggerRequest(BaseModel):
    tool_name: str = Field(..., min_length=1)
    requirement: str | None = Field(default=None)


class ForgeTriggerResponse(BaseModel):
    tool_name: str
    success: bool
    registered: bool
    attempts: int
    error: str | None = None


class TierToolModel(BaseModel):
    tool_name: str
    level: int
    total_calls: int
    window_size: int
    recent_failures: int
    recent_successes: int
    window_success_rate: float
    distinct_runs: int
    successful_runs: int
    last_updated: str
    tier_history: list[dict[str, Any]]


class TierSummaryResponse(BaseModel):
    total_tools_tracked: int
    by_level: dict[str, int]
    recent_changes: list[dict[str, Any]]
    tools: dict[str, TierToolModel]


class SkillRecordModel(BaseModel):
    tool_name: str
    description: str
    input_schema: dict[str, Any]
    approved: bool
    tier_level: int
    created_at: str


class SkillListResponse(BaseModel):
    skills: list[SkillRecordModel]
    total: int


class SkillApproveResponse(BaseModel):
    tool_name: str
    approved: bool
    registered: bool
    message: str
