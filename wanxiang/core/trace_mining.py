"""Offline trace mining — the observational half of Level 2 self-evolution.

Pure functions + dataclasses. No LLM calls, no I/O beyond what the caller
hands us. Consumes three data sources that the engine already emits:

    1. `runs.jsonl` content (list[dict]) — one record per completed run,
       each with an `events` stream as persisted by `RunManager`.
    2. `ToolRegistry.get_audit_log()` output — per-tool-call records.
    3. `AgentFactory.synthesis_log` — per-synthesis-request outcomes.

Output is a single `TraceMiningReport` that downstream consumers (the
future LLM-based trace miner agent, UI dashboards, or a simple curl)
can interpret without re-walking the raw event stream.

Design notes:
  - Failure clustering is keyword-based, not semantic. We match known
    error signatures we've actually encountered; callers can extend
    the list via `extra_failure_keywords`.
  - Tool usage is bucketed by `group` when the caller supplies a
    name→group map (typically built from `ToolRegistry._tools`). When
    no map is available we fall back to heuristic classification.
  - Reviewer convergence is computed only over runs whose plan says
    `workflow == "review_loop"`; other workflows contribute nothing.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Default failure keywords — each string matches substring-wise against the
# content of ERROR-status messages. Add new ones as new failure modes
# surface; ordering doesn't matter (we count all hits per keyword).
# ---------------------------------------------------------------------------

DEFAULT_FAILURE_KEYWORDS: tuple[str, ...] = (
    "LLM call exceeded",         # hard-timeout firing (TPM hold, slow backend)
    "Exceeded max_tool_rounds",
    "Exceeded max_retries",
    "Tool timed out",
    "Unknown tool:",
    "Invalid arguments:",
    "Claude CLI returned empty",
    "Claude CLI call failed",
    "Not logged in",
    "Native tools require",
    "sandbox did not pass",
    "handler materialization failed",
    "ANTHROPIC_API_KEY",
    "JSONDecodeError",
    "TimeoutError",
    "ConnectionError",
)


# Tool names we can classify without caller help. Anything not here and not
# in the caller-supplied `tool_groups` map falls into "unknown".
_NATIVE_TOOL_NAMES = frozenset(
    {"web_search", "web_search_20250305", "code_execution"}
)
_BUILTIN_TOOL_NAMES = frozenset({"echo", "current_time"})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FailurePattern:
    keyword: str
    count: int
    example: str  # first matching content snippet, truncated to 200 chars

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyword": self.keyword,
            "count": self.count,
            "example": self.example,
        }


@dataclass(slots=True)
class ToolUsageStats:
    tool: str
    group: str  # "builtin" | "native" | "mcp" | "synthesized" | "unknown"
    calls: int
    successes: int
    failures: int
    total_elapsed_ms: int
    avg_elapsed_ms: float
    truncated_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "group": self.group,
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "total_elapsed_ms": self.total_elapsed_ms,
            "avg_elapsed_ms": round(self.avg_elapsed_ms, 2),
            "truncated_calls": self.truncated_calls,
        }


@dataclass(slots=True)
class SynthesisStats:
    total_requests: int
    succeeded: int
    failed: int
    avg_attempts: float
    registered_tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "avg_attempts": round(self.avg_attempts, 2),
            "registered_tools": list(self.registered_tools),
        }


@dataclass(slots=True)
class ReviewerStats:
    # Count of runs with workflow == "review_loop".
    total_review_runs: int
    converged_iteration_1: int
    converged_iteration_2: int
    converged_iteration_3_plus: int
    never_converged: int
    avg_iterations: float
    max_iterations_seen: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_review_runs": self.total_review_runs,
            "converged_iteration_1": self.converged_iteration_1,
            "converged_iteration_2": self.converged_iteration_2,
            "converged_iteration_3_plus": self.converged_iteration_3_plus,
            "never_converged": self.never_converged,
            "avg_iterations": round(self.avg_iterations, 2),
            "max_iterations_seen": self.max_iterations_seen,
        }


@dataclass(slots=True)
class AgentTiming:
    agent: str
    calls: int
    total_elapsed_ms: int
    avg_elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "calls": self.calls,
            "total_elapsed_ms": self.total_elapsed_ms,
            "avg_elapsed_ms": round(self.avg_elapsed_ms, 2),
        }


@dataclass(slots=True)
class TraceMiningReport:
    window: dict[str, Any]
    total_runs: int
    final_status_distribution: dict[str, int]
    workflow_mix: dict[str, int]
    common_failure_patterns: list[FailurePattern]
    tool_usage: dict[str, ToolUsageStats]
    tool_usage_by_group: dict[str, int]
    synthesis_stats: SynthesisStats
    agent_naming: dict[str, int]       # producer (first-in-order) role name counts
    slowest_agents: list[AgentTiming]  # top N by avg_elapsed_ms
    reviewer_convergence: ReviewerStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": dict(self.window),
            "total_runs": self.total_runs,
            "final_status_distribution": dict(self.final_status_distribution),
            "workflow_mix": dict(self.workflow_mix),
            "common_failure_patterns": [
                p.to_dict() for p in self.common_failure_patterns
            ],
            "tool_usage": {
                name: stats.to_dict() for name, stats in self.tool_usage.items()
            },
            "tool_usage_by_group": dict(self.tool_usage_by_group),
            "synthesis_stats": self.synthesis_stats.to_dict(),
            "agent_naming": dict(self.agent_naming),
            "slowest_agents": [a.to_dict() for a in self.slowest_agents],
            "reviewer_convergence": self.reviewer_convergence.to_dict(),
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def mine_traces(
    runs: Iterable[dict[str, Any]],
    *,
    audit_log: list[dict[str, Any]] | None = None,
    synthesis_log: list[dict[str, Any]] | None = None,
    tool_groups: dict[str, str] | None = None,
    after: datetime | None = None,
    before: datetime | None = None,
    extra_failure_keywords: Iterable[str] = (),
    slowest_agents_top_n: int = 5,
    top_failure_patterns: int = 10,
) -> TraceMiningReport:
    """Aggregate run + audit + synthesis data into a `TraceMiningReport`.

    All inputs are already-decoded Python dicts. The caller is expected
    to have read `runs.jsonl` line-by-line (or passed a subset) and to
    have called `ToolRegistry.get_audit_log()` / accessed
    `AgentFactory.synthesis_log` itself. This function performs no I/O.
    """
    filtered_runs = [
        run for run in runs if _run_in_window(run, after=after, before=before)
    ]

    window_info = _window_info(filtered_runs, after=after, before=before)

    final_status_distribution = Counter(
        str(run.get("final_status", "")).lower() or "unknown"
        for run in filtered_runs
    )

    workflow_mix: Counter[str] = Counter()
    agent_naming: Counter[str] = Counter()
    failure_counter: Counter[str] = Counter()
    failure_examples: dict[str, str] = {}
    agent_timing_total: dict[str, int] = defaultdict(int)
    agent_timing_calls: dict[str, int] = defaultdict(int)
    tool_call_agg: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_elapsed_ms": 0,
            "truncated_calls": 0,
        }
    )

    review_iterations: list[tuple[int, str]] = []  # (iteration_count, last_reviewer_status)
    review_run_ids: set[str] = set()

    keywords = tuple(DEFAULT_FAILURE_KEYWORDS) + tuple(extra_failure_keywords)

    for run in filtered_runs:
        events = run.get("events") or []
        if not isinstance(events, list):
            events = []

        plan = _extract_plan(events)
        workflow = str(plan.get("workflow", "")) or "unknown"
        workflow_mix[workflow] += 1

        if workflow == "review_loop":
            producer_name = _extract_first_agent(plan)
            if producer_name:
                agent_naming[producer_name] += 1
            review_run_ids.add(str(run.get("run_id", "")))

        max_iter_in_run = 0
        last_reviewer_status: str | None = None

        for event in events:
            if not isinstance(event, dict):
                continue
            etype = str(event.get("type", ""))
            data = event.get("data") if isinstance(event.get("data"), dict) else {}

            if etype == "agent_completed":
                agent = str(data.get("agent", "")) or "?"
                elapsed = int(data.get("elapsed_ms", 0) or 0)
                agent_timing_total[agent] += elapsed
                agent_timing_calls[agent] += 1

                if str(data.get("status", "")).lower() == "error":
                    _record_failure_patterns(
                        text=str(data.get("content", "")),
                        keywords=keywords,
                        counter=failure_counter,
                        examples=failure_examples,
                    )

            elif etype == "tool_completed":
                tool = str(data.get("tool", "")) or "?"
                bucket = tool_call_agg[tool]
                bucket["calls"] += 1
                bucket["total_elapsed_ms"] += int(data.get("elapsed_ms", 0) or 0)
                if bool(data.get("success", False)):
                    bucket["successes"] += 1
                else:
                    bucket["failures"] += 1

            elif etype == "iteration_completed":
                iter_num = int(data.get("iteration", 0) or 0)
                if iter_num > max_iter_in_run:
                    max_iter_in_run = iter_num
                last_reviewer_status = str(data.get("reviewer_status", ""))

            elif etype == "run_completed":
                # Trace payload inside run_completed may carry error messages
                # the individual agent_completed events missed (e.g. when
                # the engine raised before any agent ran).
                trace = data.get("trace")
                if isinstance(trace, list):
                    for entry in trace:
                        if (
                            isinstance(entry, dict)
                            and str(entry.get("status", "")).lower() == "error"
                        ):
                            _record_failure_patterns(
                                text=str(entry.get("content", "")),
                                keywords=keywords,
                                counter=failure_counter,
                                examples=failure_examples,
                            )

        if workflow == "review_loop" and max_iter_in_run > 0:
            review_iterations.append(
                (max_iter_in_run, (last_reviewer_status or "").lower())
            )

    # ---- tool truncation & classification via audit log -------------------
    if audit_log:
        for record in audit_log:
            if not isinstance(record, dict):
                continue
            ts = record.get("timestamp")
            if ts is not None and not _timestamp_in_window(
                str(ts), after=after, before=before
            ):
                continue
            tool = str(record.get("tool_name", "")) or "?"
            if bool(record.get("truncated", False)):
                tool_call_agg[tool]["truncated_calls"] += 1

    tool_usage = {
        name: _finalize_tool_stats(name, bucket, tool_groups or {})
        for name, bucket in tool_call_agg.items()
    }
    tool_usage_by_group: Counter[str] = Counter()
    for stats in tool_usage.values():
        tool_usage_by_group[stats.group] += stats.calls

    # ---- failure patterns -------------------------------------------------
    patterns = [
        FailurePattern(
            keyword=kw,
            count=count,
            example=failure_examples.get(kw, ""),
        )
        for kw, count in failure_counter.most_common()
    ][:top_failure_patterns]

    # ---- slowest agents ---------------------------------------------------
    agent_timings = [
        AgentTiming(
            agent=name,
            calls=agent_timing_calls[name],
            total_elapsed_ms=agent_timing_total[name],
            avg_elapsed_ms=(
                agent_timing_total[name] / agent_timing_calls[name]
                if agent_timing_calls[name]
                else 0.0
            ),
        )
        for name in agent_timing_total
    ]
    agent_timings.sort(key=lambda a: a.avg_elapsed_ms, reverse=True)
    slowest = agent_timings[: max(0, slowest_agents_top_n)]

    # ---- synthesis stats --------------------------------------------------
    synthesis_stats = _summarize_synthesis(synthesis_log or [])

    # ---- reviewer convergence --------------------------------------------
    reviewer = _summarize_reviewer(review_iterations)

    return TraceMiningReport(
        window=window_info,
        total_runs=len(filtered_runs),
        final_status_distribution=dict(final_status_distribution),
        workflow_mix=dict(workflow_mix),
        common_failure_patterns=patterns,
        tool_usage=tool_usage,
        tool_usage_by_group=dict(tool_usage_by_group),
        synthesis_stats=synthesis_stats,
        agent_naming=dict(agent_naming),
        slowest_agents=slowest,
        reviewer_convergence=reviewer,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_in_window(
    run: dict[str, Any],
    *,
    after: datetime | None,
    before: datetime | None,
) -> bool:
    if after is None and before is None:
        return True
    # Prefer completed_at; fall back to started_at.
    ts_raw = str(run.get("completed_at") or run.get("started_at") or "")
    return _timestamp_in_window(ts_raw, after=after, before=before)


def _timestamp_in_window(
    ts_raw: str, *, after: datetime | None, before: datetime | None
) -> bool:
    ts = _parse_iso(ts_raw)
    if ts is None:
        # If we can't parse, include by default — better to overcount than
        # to silently drop data the operator wanted to see.
        return True
    if after is not None and ts < after:
        return False
    if before is not None and ts > before:
        return False
    return True


def _parse_iso(ts_raw: str) -> datetime | None:
    if not ts_raw:
        return None
    try:
        # Python 3.11+ handles the trailing 'Z' in fromisoformat; earlier
        # versions need the manual swap.
        cleaned = ts_raw.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _window_info(
    runs: list[dict[str, Any]],
    *,
    after: datetime | None,
    before: datetime | None,
) -> dict[str, Any]:
    starts = [str(run.get("started_at", "")) for run in runs if run.get("started_at")]
    ends = [
        str(run.get("completed_at", ""))
        for run in runs
        if run.get("completed_at")
    ]
    return {
        "after": after.isoformat() if after else None,
        "before": before.isoformat() if before else None,
        "first_run_started_at": min(starts) if starts else None,
        "last_run_completed_at": max(ends) if ends else None,
    }


def _extract_plan(events: list[Any]) -> dict[str, Any]:
    for event in events:
        if not isinstance(event, dict):
            continue
        if str(event.get("type", "")) != "run_started":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        plan = data.get("plan")
        if isinstance(plan, dict):
            return plan
    return {}


def _extract_first_agent(plan: dict[str, Any]) -> str | None:
    order = plan.get("execution_order")
    if isinstance(order, list) and order:
        first = order[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    agents = plan.get("agents")
    if isinstance(agents, list) and agents:
        first = agents[0]
        if isinstance(first, dict):
            name = first.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return None


def _record_failure_patterns(
    *,
    text: str,
    keywords: tuple[str, ...],
    counter: Counter[str],
    examples: dict[str, str],
) -> None:
    if not text:
        return
    for keyword in keywords:
        if keyword in text:
            counter[keyword] += 1
            if keyword not in examples:
                snippet = " ".join(text.split())
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."
                examples[keyword] = snippet


def _classify_tool(tool: str, tool_groups: dict[str, str]) -> str:
    # Explicit registration wins — `tool in tool_groups` says "this tool
    # is registered locally in the ToolRegistry", regardless of whether
    # its group label is empty (the convention for builtin tools).
    # Before this check the test was `if explicit:` which treated an
    # empty-string group as "not classified" and fell through to the
    # _NATIVE_TOOL_NAMES heuristic. That misfired after Phase 7 when
    # web_search was moved from native-only into the builtin registry:
    # the registered builtin tool kept showing up as `native` in
    # mining reports.
    if tool in tool_groups:
        lower = (tool_groups[tool] or "").lower()
        if lower in {"", "builtin"}:
            return "builtin"
        if lower == "synthesized":
            return "synthesized"
        return "mcp"  # any non-empty, non-synthesized group comes from an MCP server
    # Not in the registry — fall back to the name-based heuristic for
    # Claude-native server-side tools (e.g. a raw `web_search` without
    # a local registry entry) and a few well-known builtins.
    if tool in _NATIVE_TOOL_NAMES:
        return "native"
    if tool in _BUILTIN_TOOL_NAMES:
        return "builtin"
    return "unknown"


def _finalize_tool_stats(
    name: str,
    bucket: dict[str, int],
    tool_groups: dict[str, str],
) -> ToolUsageStats:
    calls = bucket["calls"]
    avg = bucket["total_elapsed_ms"] / calls if calls else 0.0
    return ToolUsageStats(
        tool=name,
        group=_classify_tool(name, tool_groups),
        calls=calls,
        successes=bucket["successes"],
        failures=bucket["failures"],
        total_elapsed_ms=bucket["total_elapsed_ms"],
        avg_elapsed_ms=avg,
        truncated_calls=bucket["truncated_calls"],
    )


def _summarize_synthesis(entries: list[dict[str, Any]]) -> SynthesisStats:
    total = len(entries)
    succeeded = sum(1 for e in entries if bool(e.get("success", False)))
    failed = total - succeeded
    attempts = [int(e.get("attempts", 0) or 0) for e in entries]
    avg_attempts = sum(attempts) / len(attempts) if attempts else 0.0
    registered = [
        str(e.get("tool_name"))
        for e in entries
        if e.get("tool_name") and bool(e.get("registered", False))
    ]
    return SynthesisStats(
        total_requests=total,
        succeeded=succeeded,
        failed=failed,
        avg_attempts=avg_attempts,
        registered_tools=registered,
    )


def _summarize_reviewer(
    iterations: list[tuple[int, str]],
) -> ReviewerStats:
    total = len(iterations)
    if total == 0:
        return ReviewerStats(
            total_review_runs=0,
            converged_iteration_1=0,
            converged_iteration_2=0,
            converged_iteration_3_plus=0,
            never_converged=0,
            avg_iterations=0.0,
            max_iterations_seen=0,
        )

    c1 = c2 = c3p = never = 0
    for count, last_status in iterations:
        if last_status != "success":
            never += 1
            continue
        if count <= 1:
            c1 += 1
        elif count == 2:
            c2 += 1
        else:
            c3p += 1

    avg = sum(c for c, _ in iterations) / total
    max_iter = max(c for c, _ in iterations)
    return ReviewerStats(
        total_review_runs=total,
        converged_iteration_1=c1,
        converged_iteration_2=c2,
        converged_iteration_3_plus=c3p,
        never_converged=never,
        avg_iterations=avg,
        max_iterations_seen=max_iter,
    )
