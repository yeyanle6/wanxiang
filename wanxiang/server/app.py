from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from datetime import datetime

from ..core.agent import AgentConfig, BaseAgent
from ..core.builtin_tools import create_default_registry
from ..core.autoschool import Autoschool
from ..core.conversation import ConversationManager
from ..core.curriculum import commit_tasks, generate_l0_tasks
from ..core.gap_detector import detect_synthesis_candidates
from ..core.growth_budget import GrowthBudget
from ..core.mcp_loader import MCPPool, load_mcp_declarations
from ..core.outcome_tagger import tag_run
from ..core.project import create_project, load_project, update_status
from ..core.sandbox import SandboxExecutor
from ..core.seed_loader import enqueue_seed_tasks
from ..core.skill_forge import SkillForge
from ..core.skill_loader import approve_skill, list_skills, load_approved_skills
from ..core.storage import Storage
from ..core.tier import TierManager
from ..core.trace_mining import mine_traces
from ..core.workspace import bootstrap_workspace
from .mcp_status import probe_mcp_status
from .models import (
    ConversationCreateRequest,
    ConversationDetailResponse,
    ConversationMessageRequest,
    ConversationModel,
    ConversationTurnModel,
    ConversationTurnResponse,
    ForgeCandidatesResponse,
    ForgeTriggerRequest,
    ForgeTriggerResponse,
    MCPStatusResponse,
    ProjectCreateRequest,
    ProjectListResponse,
    ProjectModel,
    RunDetailResponse,
    RunListResponse,
    RunRequest,
    RunResponse,
    SkillApproveResponse,
    SkillListResponse,
    SkillRecordModel,
    TierSummaryResponse,
    TraceMiningResponse,
)
from .runner import RunManager


logger = logging.getLogger("wanxiang.server")

UI_FILE = Path(__file__).resolve().parents[2] / "wanxiang-ui.jsx"
MCP_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "mcp.yaml"
SKILL_SYNTHESIZER_YAML = (
    Path(__file__).resolve().parents[2] / "configs" / "agents" / "skill_synthesizer.yaml"
)
SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"
STORAGE_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "wanxiang.db"
RUNS_JSONL_PATH = Path(__file__).resolve().parents[2] / "data" / "runs.jsonl"
SEED_TASKS_YAML = Path(__file__).resolve().parents[2] / "configs" / "seed_tasks.yaml"

_run_manager: RunManager | None = None
_mcp_pool: MCPPool | None = None
_skill_forge: SkillForge | None = None
_storage: Storage | None = None
_autoschool: Autoschool | None = None


def _get_run_manager() -> RunManager:
    if _run_manager is None:
        raise HTTPException(status_code=503, detail="Server not yet initialized.")
    return _run_manager


def _get_storage() -> Storage:
    """Project / conversation routes require SQLite storage. Gated behind
    WANXIANG_STORAGE_DUAL_WRITE — without it, projects have no home.
    """
    if _storage is None:
        raise HTTPException(
            status_code=503,
            detail="Storage is not enabled. Set WANXIANG_STORAGE_DUAL_WRITE=1.",
        )
    return _storage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _run_manager, _mcp_pool, _skill_forge, _storage, _autoschool
    tool_registry = create_default_registry()
    _mcp_pool = MCPPool()

    if _storage_enabled():
        _storage = Storage(STORAGE_DB_PATH)
        if _should_bootstrap_storage(_storage):
            migrated = _storage.import_jsonl(RUNS_JSONL_PATH)
            if migrated:
                logger.info(
                    "Bootstrapped SQLite from %s: %d run(s) imported",
                    RUNS_JSONL_PATH,
                    migrated,
                )
        tagged = _retag_untagged_runs(_storage)
        if tagged:
            logger.info("Assigned outcome to %d run(s) with outcome=NULL", tagged)

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

    llm_mode = os.getenv("WANXIANG_LLM_MODE")
    tier_manager = TierManager()

    SKILLS_DIR.mkdir(exist_ok=True)
    loaded = load_approved_skills(SKILLS_DIR, tool_registry, tier_manager)
    if loaded:
        logger.info("Loaded %d approved skill(s) from %s", loaded, SKILLS_DIR)

    _skill_forge = _build_skill_forge(
        tool_registry=tool_registry,
        llm_mode=llm_mode,
        tier_manager=tier_manager,
        skills_dir=SKILLS_DIR,
    )

    budget: GrowthBudget | None = None
    usage_recorder = None
    if _storage is not None:
        budget = GrowthBudget(_storage)
        usage_recorder = budget.record_usage

    _run_manager = RunManager(
        llm_mode=llm_mode,
        tool_registry=tool_registry,
        skill_forge=_skill_forge,
        tier_manager=tier_manager,
        storage=_storage,
        usage_recorder=usage_recorder,
    )

    if _storage is not None and budget is not None and _autoschool_enabled():
        _bootstrap_curriculum(_storage, tool_registry)
        _autoschool = _build_autoschool(_storage, _run_manager, budget)
        if _autoschool is not None:
            await _autoschool.start()

    try:
        yield
    finally:
        if _autoschool is not None:
            try:
                await _autoschool.stop()
            except Exception:
                logger.exception("autoschool stop failed")
        if _mcp_pool is not None:
            await _mcp_pool.close()
        if _storage is not None:
            _storage.close()
        _run_manager = None
        _mcp_pool = None
        _skill_forge = None
        _storage = None
        _autoschool = None


def _storage_enabled() -> bool:
    flag = os.getenv("WANXIANG_STORAGE_DUAL_WRITE", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _autoschool_enabled() -> bool:
    flag = os.getenv("WANXIANG_AUTOSCHOOL_ENABLED", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _bootstrap_curriculum(storage: Storage, tool_registry) -> None:
    """Fill curriculum_queue from seed YAML + L0 template generator.

    Both operations are idempotent (enqueue_task_if_new dedups by
    (task, source)), so this is safe to call every startup — it only
    adds new tasks when the seed file grows or new tools register.
    """
    try:
        seed_added = enqueue_seed_tasks(storage, SEED_TASKS_YAML)
        if seed_added:
            logger.info("Autoschool: enqueued %d new seed task(s)", seed_added)
    except Exception:
        logger.exception("Autoschool: seed_loader failed; continuing")
    try:
        l0_added = commit_tasks(
            storage, generate_l0_tasks(tool_registry), source="l0_generator"
        )
        if l0_added:
            logger.info("Autoschool: enqueued %d new L0 task(s)", l0_added)
    except Exception:
        logger.exception("Autoschool: L0 generator failed; continuing")


def _build_autoschool(
    storage: Storage, run_manager: RunManager, budget: GrowthBudget
) -> Autoschool | None:
    try:
        tick_s = float(os.getenv("WANXIANG_AUTOSCHOOL_TICK_S", "60"))
        max_concurrent = int(os.getenv("WANXIANG_AUTOSCHOOL_MAX_CONCURRENT", "1"))
    except ValueError:
        logger.exception("Autoschool: invalid env config; disabling")
        return None
    return Autoschool(
        storage=storage,
        run_manager=run_manager,
        budget=budget,
        tick_interval_s=tick_s,
        max_concurrent=max_concurrent,
    )


def _should_bootstrap_storage(storage: Storage) -> bool:
    """Only import runs.jsonl when the SQLite DB has no runs yet.

    Idempotence guard — re-running the server must not duplicate history.
    """
    return len(storage.list_runs(limit=1)) == 0


def _retag_untagged_runs(storage: Storage) -> int:
    """Compute outcome for any run whose outcome is still NULL.

    Runs imported via import_jsonl land without an outcome label. This
    walks the DB once per startup and fills in the blanks. It's a no-op
    when every run is already tagged, so paying the scan on every boot
    is cheap.
    """
    count = 0
    for run in storage.list_runs(limit=10_000):
        if run.outcome is not None:
            continue
        full = storage.get_run(run.run_id, with_events=True)
        if full is None:
            continue
        outcome = tag_run(full.events or [], full.final_status)
        storage.update_outcome(run.run_id, outcome)
        count += 1
    return count


def _build_skill_forge(
    *,
    tool_registry,
    llm_mode: str | None,
    tier_manager: TierManager | None = None,
    skills_dir: Path | None = None,
) -> SkillForge | None:
    """Wire up SynthesizerAgent + SandboxExecutor + SkillForge.

    Disabled by default. Set WANXIANG_ENABLE_SKILL_FORGE=1 to turn on.
    If the skill_synthesizer YAML is missing or malformed, logs a warning
    and returns None — the server still runs, only synthesis requests
    degrade gracefully.
    """
    enable_flag = os.getenv("WANXIANG_ENABLE_SKILL_FORGE", "").strip().lower()
    if enable_flag not in {"1", "true", "yes", "on"}:
        logger.info(
            "SkillForge disabled (set WANXIANG_ENABLE_SKILL_FORGE=1 to enable)"
        )
        return None
    if not SKILL_SYNTHESIZER_YAML.exists():
        logger.warning(
            "SKILL_FORGE enabled but %s is missing; running without forge",
            SKILL_SYNTHESIZER_YAML,
        )
        return None

    try:
        synth_config = AgentConfig.from_yaml(SKILL_SYNTHESIZER_YAML)
    except Exception:
        logger.exception(
            "Failed to load SkillSynthesizer config from %s", SKILL_SYNTHESIZER_YAML
        )
        return None

    synthesizer = BaseAgent(
        config=synth_config,
        tool_registry=tool_registry,
        llm_mode=llm_mode,
    )
    sandbox = SandboxExecutor(
        timeout_s=float(os.getenv("WANXIANG_SANDBOX_TIMEOUT_S", "30")),
    )
    forge = SkillForge(
        sandbox=sandbox,
        registry=tool_registry,
        synthesizer=synthesizer,
        max_retries=int(os.getenv("WANXIANG_SKILL_FORGE_RETRIES", "3")),
        tier_manager=tier_manager,
        skills_dir=skills_dir,
    )
    logger.info("SkillForge ready: synthesizer=%s", synth_config.name)
    return forge


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


@app.post("/api/projects", response_model=ProjectModel)
async def create_project_endpoint(request: ProjectCreateRequest) -> ProjectModel:
    """Create a project, bootstrap its workspace, return the record."""
    storage = _get_storage()
    try:
        record = create_project(
            storage,
            name=request.name,
            user_goal=request.user_goal,
            workspace_dir="",  # filled in below after slug is known
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Bootstrap the workspace filesystem + venv and update the path in storage.
    try:
        workspace_path = await asyncio.to_thread(bootstrap_workspace, record)
    except Exception as exc:
        logger.exception("Workspace bootstrap failed for project %s", record.project_id)
        raise HTTPException(status_code=500, detail=f"Workspace bootstrap failed: {exc}") from exc

    # Persist the real workspace path (update_project_status won't do it, so touch directly).
    with storage._tx() as c:  # noqa: SLF001 — project-local bookkeeping
        from ..core.storage import _utcnow
        c.execute(
            "UPDATE projects SET workspace_dir = ?, updated_at = ? WHERE project_id = ?",
            (str(workspace_path), _utcnow(), record.project_id),
        )
    refreshed = load_project(storage, record.project_id)
    assert refreshed is not None
    return ProjectModel(**refreshed.to_dict())


@app.get("/api/projects/{project_id}", response_model=ProjectModel)
async def get_project_endpoint(project_id: str) -> ProjectModel:
    storage = _get_storage()
    record = load_project(storage, project_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return ProjectModel(**record.to_dict())


@app.get("/api/projects", response_model=ProjectListResponse)
async def list_projects_endpoint() -> ProjectListResponse:
    storage = _get_storage()
    records = storage.list_projects(limit=100)
    return ProjectListResponse(
        projects=[ProjectModel(**r.to_dict()) for r in records],
        total=len(records),
    )


@app.post("/api/conversations", response_model=ConversationModel)
async def create_conversation_endpoint(
    request: ConversationCreateRequest,
) -> ConversationModel:
    """Start a new conversation with the initial user message."""
    storage = _get_storage()
    if load_project(storage, request.project_id) is None:
        raise HTTPException(
            status_code=404, detail=f"Project not found: {request.project_id}"
        )
    manager = ConversationManager(storage)
    try:
        conv = manager.start(request.project_id, request.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ConversationModel(**conv.to_dict())


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation_endpoint(conversation_id: str) -> ConversationDetailResponse:
    storage = _get_storage()
    manager = ConversationManager(storage)
    conv = manager.get(conversation_id)
    if conv is None:
        raise HTTPException(
            status_code=404, detail=f"Conversation not found: {conversation_id}"
        )
    turns = manager.turns(conversation_id)
    return ConversationDetailResponse(
        conversation=ConversationModel(**conv.to_dict()),
        turns=[ConversationTurnModel(**t.to_dict()) for t in turns],
    )


@app.post(
    "/api/conversations/{conversation_id}/messages",
    response_model=ConversationTurnResponse,
)
async def post_conversation_message(
    conversation_id: str, request: ConversationMessageRequest
) -> ConversationTurnResponse:
    """Append user message, dispatch probe-mode run, append system reply."""
    storage = _get_storage()
    manager = ConversationManager(storage)
    conv = manager.get(conversation_id)
    if conv is None:
        raise HTTPException(
            status_code=404, detail=f"Conversation not found: {conversation_id}"
        )

    # 1. Append user turn (may also transition awaiting_user → open).
    try:
        manager.append_user_turn(conversation_id, request.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 2. Dispatch a probe-mode run with conversation context for dialogue.
    rm = _get_run_manager()
    context = manager.render_context(conversation_id)
    run_id = await rm.start_run(
        request.message,
        probe=True,
        source="conversation",
        conversation_id=conversation_id,
        project_id=conv.project_id,
        conversation_context=context,
    )

    # 3. Wait for completion, extract final agent output.
    await rm.wait_for_run(run_id)
    run_record = storage.get_run(run_id, with_events=True)
    system_text = _extract_final_agent_text(run_record.events if run_record else [])

    # 4. Append system turn — detects NEEDS_CLARIFICATION marker automatically.
    manager.append_system_turn(conversation_id, system_text, run_id=run_id)

    # 5. Return shape for client.
    updated = manager.get(conversation_id)
    assert updated is not None
    return ConversationTurnResponse(
        conversation_id=conversation_id,
        status=updated.status,
        next_speaker=manager.next_speaker(conversation_id),
        system_response=system_text,
        run_id=run_id,
        clarification=ConversationManager.is_clarification(system_text),
    )


def _extract_final_agent_text(events: list[dict[str, Any]]) -> str:
    """Pull content from the LAST agent_completed event."""
    last = ""
    for event in events or []:
        if not isinstance(event, dict):
            continue
        if event.get("type") != "agent_completed":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        content = data.get("content") or data.get("content_preview") or ""
        if isinstance(content, str) and content.strip():
            last = content
    return last


@app.post("/api/runs", response_model=RunResponse)
async def create_run(request: RunRequest) -> RunResponse:
    rm = _get_run_manager()
    try:
        run_id = await rm.start_run(request.task, probe=request.probe)
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


@app.get("/api/skill-forge/status")
async def get_skill_forge_status() -> dict:
    """Report whether SkillForge is wired up this run."""
    if _skill_forge is None:
        return {
            "ready": False,
            "reason": (
                "SkillForge not enabled. Set WANXIANG_ENABLE_SKILL_FORGE=1 "
                "and ensure configs/agents/skill_synthesizer.yaml exists."
            ),
        }
    synthesizer_name = getattr(
        getattr(_skill_forge.synthesizer, "config", None), "name", "unknown"
    )
    return {
        "ready": True,
        "synthesizer": synthesizer_name,
        "max_retries": _skill_forge.max_retries,
        "recent_synthesis_count": len(
            getattr(_get_run_manager().factory, "synthesis_log", [])
        ),
    }


@app.get("/api/tools/audit")
async def get_tool_audit_log(
    limit: int = Query(default=100, ge=1, le=1000),
    tool: str | None = Query(default=None),
) -> dict:
    """Return the tool registry's recent call audit records.

    Newest last. `limit` caps how many to return (most recent N);
    `tool` filters by exact tool_name match before applying the limit.
    """
    rm = _get_run_manager()
    registry = getattr(rm.factory, "tool_registry", None)
    if registry is None:
        return {"records": [], "total_returned": 0}
    records = registry.get_audit_log(limit=limit, tool=tool)
    return {
        "records": records,
        "total_returned": len(records),
    }


@app.get("/api/tier", response_model=TierSummaryResponse)
async def get_tier_summary() -> TierSummaryResponse:
    """Return the current TierManager snapshot — per-tool trust levels and recent changes."""
    rm = _get_run_manager()
    summary = rm.tier_manager.get_tier_summary()
    return TierSummaryResponse(**summary)


@app.get("/api/forge/candidates", response_model=ForgeCandidatesResponse)
async def get_forge_candidates(
    min_failures: int = Query(default=1, ge=1, description="Minimum failure count to surface a candidate."),
) -> ForgeCandidatesResponse:
    """Return synthesis candidates derived from recurring 'Unknown tool:' gaps."""
    rm = _get_run_manager()
    runs = await rm.read_raw_history()

    registry = getattr(rm.factory, "tool_registry", None)
    audit_log = registry.get_audit_log() if registry is not None else []
    tool_groups = registry.get_tool_groups() if registry is not None else {}
    synthesis_log = list(getattr(rm.factory, "synthesis_log", []) or [])
    tier_changes = [ch.to_dict() for ch in rm.tier_manager.get_recent_changes()]

    report = mine_traces(
        runs,
        audit_log=audit_log,
        synthesis_log=synthesis_log,
        tool_groups=tool_groups,
        tier_changes=tier_changes,
    )
    candidates = detect_synthesis_candidates(
        report.capability_gap_patterns,
        runs,
        min_failure_count=min_failures,
    )
    return ForgeCandidatesResponse(
        candidates=[c.to_dict() for c in candidates],
        forge_ready=_skill_forge is not None,
    )


@app.post("/api/forge/trigger", response_model=ForgeTriggerResponse)
async def trigger_forge(request: ForgeTriggerRequest) -> ForgeTriggerResponse:
    """Trigger SkillForge synthesis for a named candidate.

    Requires WANXIANG_ENABLE_SKILL_FORGE=1. The forge runs synchronously —
    expect several seconds of latency. Returns the result immediately.
    """
    if _skill_forge is None:
        raise HTTPException(
            status_code=503,
            detail="SkillForge not enabled. Set WANXIANG_ENABLE_SKILL_FORGE=1.",
        )

    rm = _get_run_manager()
    runs = await rm.read_raw_history()

    requirement = (request.requirement or "").strip()
    if not requirement:
        registry = getattr(rm.factory, "tool_registry", None)
        audit_log = registry.get_audit_log() if registry is not None else []
        tool_groups = registry.get_tool_groups() if registry is not None else {}
        synthesis_log = list(getattr(rm.factory, "synthesis_log", []) or [])
        tier_changes = [ch.to_dict() for ch in rm.tier_manager.get_recent_changes()]

        report = mine_traces(
            runs,
            audit_log=audit_log,
            synthesis_log=synthesis_log,
            tool_groups=tool_groups,
            tier_changes=tier_changes,
        )
        candidates = detect_synthesis_candidates(report.capability_gap_patterns, runs)
        match = next((c for c in candidates if c.tool_name == request.tool_name), None)
        requirement = match.suggested_requirement if match else (
            f"Implement a tool named '{request.tool_name}'."
        )

    result = await _skill_forge.forge(requirement)
    return ForgeTriggerResponse(
        tool_name=request.tool_name,
        success=result.success,
        registered=result.registered,
        attempts=len(result.attempts),
        error=result.error,
    )


@app.get("/api/skills", response_model=SkillListResponse)
async def list_skills_endpoint() -> SkillListResponse:
    """List all synthesized skills — both pending (approved=false) and approved."""
    records = list_skills(SKILLS_DIR)
    return SkillListResponse(
        skills=[SkillRecordModel(**r.to_dict()) for r in records],
        total=len(records),
    )


@app.post("/api/skills/{tool_name}/approve", response_model=SkillApproveResponse)
async def approve_skill_endpoint(tool_name: str) -> SkillApproveResponse:
    """Flip approved=true and register the skill into the live registry."""
    record = approve_skill(SKILLS_DIR, tool_name)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Skill '{tool_name}' not found in {SKILLS_DIR}",
        )

    rm = _get_run_manager()
    registry = getattr(rm.factory, "tool_registry", None)
    already_registered = registry is not None and registry.get(tool_name) is not None
    registered_now = False

    if not already_registered and registry is not None:
        try:
            namespace: dict[str, Any] = {}
            exec(record.handler_code, namespace)  # noqa: S102
            handler_fn = namespace.get("handler") or namespace.get(tool_name)
            if callable(handler_fn):
                from ..core.tools import ToolSpec
                tool_spec = ToolSpec(
                    name=record.tool_name,
                    description=record.description,
                    input_schema=record.input_schema,
                    handler=handler_fn,
                    group="synthesized",
                )
                registry.register(tool_spec)
                rm.tier_manager.initialize_tool(tool_name, record.tier_level)
                registered_now = True
                logger.info("Approved and registered skill '%s'", tool_name)
        except Exception:
            logger.exception("Failed to register skill '%s' after approval", tool_name)

    return SkillApproveResponse(
        tool_name=tool_name,
        approved=True,
        registered=already_registered or registered_now,
        message=(
            "Already registered." if already_registered
            else ("Registered." if registered_now else "Approved; registration failed — check logs.")
        ),
    )


@app.get("/api/trace/mining", response_model=TraceMiningResponse)
async def get_trace_mining(
    after: str | None = Query(default=None, description="ISO-8601 lower bound."),
    before: str | None = Query(default=None, description="ISO-8601 upper bound."),
) -> TraceMiningResponse:
    """Aggregate `runs.jsonl` + tool audit + synthesis log into a report.

    Pure static analysis: no LLM calls, no event replay. Supports
    optional ``?after=`` / ``?before=`` ISO-8601 filters. The report
    shape is stable — the future LLM-based miner agent will consume
    the same JSON.
    """
    rm = _get_run_manager()
    after_dt = _parse_query_datetime("after", after)
    before_dt = _parse_query_datetime("before", before)

    runs = await rm.read_raw_history()

    registry = getattr(rm.factory, "tool_registry", None)
    audit_log = registry.get_audit_log() if registry is not None else []
    tool_groups = registry.get_tool_groups() if registry is not None else {}

    synthesis_log = list(getattr(rm.factory, "synthesis_log", []) or [])

    tier_changes = rm.tier_manager.get_recent_changes()
    report = mine_traces(
        runs,
        audit_log=audit_log,
        synthesis_log=synthesis_log,
        tool_groups=tool_groups,
        after=after_dt,
        before=before_dt,
        tier_changes=[ch.to_dict() for ch in tier_changes],
    )
    return TraceMiningResponse(**report.to_dict())


def _parse_query_datetime(param_name: str, raw: str | None) -> datetime | None:
    if raw is None or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ISO-8601 timestamp for '{param_name}': {raw}",
        ) from exc


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
