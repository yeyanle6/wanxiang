from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from wanxiang.core.trace_mining import (
    DEFAULT_FAILURE_KEYWORDS,
    FailurePattern,
    ReviewerStats,
    SynthesisStats,
    TraceMiningReport,
    mine_traces,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "trace_mining"


def _load_runs() -> list[dict]:
    lines = FIXTURE_DIR.joinpath("runs.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _load_audit() -> list[dict]:
    return json.loads(FIXTURE_DIR.joinpath("audit_log.json").read_text(encoding="utf-8"))


def _load_synthesis() -> list[dict]:
    return json.loads(
        FIXTURE_DIR.joinpath("synthesis_log.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def runs() -> list[dict]:
    return _load_runs()


@pytest.fixture
def audit_log() -> list[dict]:
    return _load_audit()


@pytest.fixture
def synthesis_log() -> list[dict]:
    return _load_synthesis()


# ---------------------------------------------------------------------------
# Basic aggregation
# ---------------------------------------------------------------------------


def test_total_runs_and_status_distribution(runs):
    report = mine_traces(runs)
    assert report.total_runs == 5
    assert report.final_status_distribution == {"success": 3, "error": 2}


def test_workflow_mix(runs):
    report = mine_traces(runs)
    # 2 review_loop, 2 pipeline, 1 parallel
    assert report.workflow_mix == {
        "review_loop": 2,
        "pipeline": 2,
        "parallel": 1,
    }


def test_window_info_spans_fixture_range(runs):
    report = mine_traces(runs)
    assert report.window["first_run_started_at"] == "2026-04-10T10:00:00+00:00"
    assert report.window["last_run_completed_at"] == "2026-04-14T14:01:30+00:00"
    assert report.window["after"] is None
    assert report.window["before"] is None


# ---------------------------------------------------------------------------
# Failure patterns (keyword clustering, not semantic)
# ---------------------------------------------------------------------------


def test_common_failure_patterns_picks_up_known_signatures(runs):
    report = mine_traces(runs)
    keywords_found = {p.keyword for p in report.common_failure_patterns}
    # Both of these appear literally in the fixture error messages.
    assert "Exceeded max_tool_rounds" in keywords_found
    assert "Claude CLI call failed" in keywords_found
    assert "Not logged in" in keywords_found

    # Patterns are sorted by count descending.
    counts = [p.count for p in report.common_failure_patterns]
    assert counts == sorted(counts, reverse=True)


def test_failure_pattern_carries_example_snippet(runs):
    report = mine_traces(runs)
    by_kw = {p.keyword: p for p in report.common_failure_patterns}
    assert "Exceeded max_tool_rounds" in by_kw
    assert "Exceeded max_tool_rounds" in by_kw["Exceeded max_tool_rounds"].example


def test_extra_failure_keywords_are_honored(runs):
    report = mine_traces(runs, extra_failure_keywords=["run_failed"])
    # The `run_failed` intent appears in the CLI-error trace entry, but
    # we only match against `content`, so this keyword should NOT show up.
    keywords = {p.keyword for p in report.common_failure_patterns}
    assert "run_failed" not in keywords

    report2 = mine_traces(runs, extra_failure_keywords=["claude auth login"])
    keywords2 = {p.keyword for p in report2.common_failure_patterns}
    assert "claude auth login" in keywords2


# ---------------------------------------------------------------------------
# Tool usage (grouped)
# ---------------------------------------------------------------------------


def test_tool_usage_counts_calls_across_runs(runs, audit_log):
    report = mine_traces(runs, audit_log=audit_log)
    usage = report.tool_usage

    # web_search: 1 call, synthesized gets 3, read_text_file gets 2.
    assert usage["web_search"].calls == 1
    assert usage["chinese_numeral_to_int"].calls == 3
    assert usage["read_text_file"].calls == 2

    # Elapsed averaging.
    assert usage["chinese_numeral_to_int"].avg_elapsed_ms == pytest.approx(
        (5 + 3 + 2) / 3
    )


def test_tool_usage_picks_up_truncation_from_audit(runs, audit_log):
    report = mine_traces(runs, audit_log=audit_log)
    assert report.tool_usage["web_search"].truncated_calls == 1
    assert report.tool_usage["read_text_file"].truncated_calls == 0


def test_tool_classification_distinguishes_native_mcp_synthesized(runs, audit_log):
    tool_groups = {
        "read_text_file": "filesystem",       # MCP server
        "chinese_numeral_to_int": "synthesized",
        # web_search is not in the map — relies on native fallback.
    }
    report = mine_traces(runs, audit_log=audit_log, tool_groups=tool_groups)
    assert report.tool_usage["web_search"].group == "native"
    assert report.tool_usage["read_text_file"].group == "mcp"
    assert report.tool_usage["chinese_numeral_to_int"].group == "synthesized"

    # By-group totals aggregate across all tools within a group.
    by_group = report.tool_usage_by_group
    assert by_group["native"] == 1
    assert by_group["mcp"] == 2
    assert by_group["synthesized"] == 3


def test_unknown_tool_without_classification_is_flagged(runs):
    report = mine_traces(runs)
    # Without tool_groups passed, `read_text_file` and
    # `chinese_numeral_to_int` can't be classified — they land in 'unknown'.
    assert report.tool_usage["read_text_file"].group == "unknown"
    assert report.tool_usage["chinese_numeral_to_int"].group == "unknown"
    # But `web_search` is in the built-in native list.
    assert report.tool_usage["web_search"].group == "native"


# ---------------------------------------------------------------------------
# Synthesis stats
# ---------------------------------------------------------------------------


def test_synthesis_stats_counts_successes_and_failures(runs, synthesis_log):
    report = mine_traces(runs, synthesis_log=synthesis_log)
    s = report.synthesis_stats
    assert isinstance(s, SynthesisStats)
    assert s.total_requests == 3
    assert s.succeeded == 2
    assert s.failed == 1
    assert s.avg_attempts == pytest.approx((1 + 2 + 3) / 3)
    assert set(s.registered_tools) == {"chinese_numeral_to_int", "fib"}


def test_synthesis_stats_without_log_is_zero(runs):
    report = mine_traces(runs)
    assert report.synthesis_stats.total_requests == 0
    assert report.synthesis_stats.registered_tools == []


# ---------------------------------------------------------------------------
# Reviewer convergence
# ---------------------------------------------------------------------------


def test_reviewer_convergence_splits_by_iteration_count(runs):
    report = mine_traces(runs)
    rc = report.reviewer_convergence
    assert isinstance(rc, ReviewerStats)
    # Two review_loop runs: run-review-2turn converged at iter 2,
    # run-analyst-review converged at iter 1.
    assert rc.total_review_runs == 2
    assert rc.converged_iteration_1 == 1
    assert rc.converged_iteration_2 == 1
    assert rc.converged_iteration_3_plus == 0
    assert rc.never_converged == 0
    assert rc.avg_iterations == pytest.approx(1.5)
    assert rc.max_iterations_seen == 2


def test_reviewer_convergence_counts_non_success_as_never():
    run = {
        "run_id": "stuck",
        "started_at": "2026-04-15T00:00:00+00:00",
        "completed_at": "2026-04-15T00:05:00+00:00",
        "final_status": "needs_revision",
        "events": [
            {
                "type": "run_started",
                "data": {
                    "plan": {
                        "workflow": "review_loop",
                        "execution_order": ["writer", "reviewer"],
                        "agents": [],
                    }
                },
            },
            {
                "type": "iteration_completed",
                "data": {"iteration": 1, "reviewer_status": "needs_revision"},
            },
            {
                "type": "iteration_completed",
                "data": {"iteration": 2, "reviewer_status": "needs_revision"},
            },
            {
                "type": "iteration_completed",
                "data": {"iteration": 3, "reviewer_status": "needs_revision"},
            },
        ],
    }
    report = mine_traces([run])
    rc = report.reviewer_convergence
    assert rc.total_review_runs == 1
    assert rc.never_converged == 1
    assert rc.converged_iteration_1 == 0
    assert rc.max_iterations_seen == 3


# ---------------------------------------------------------------------------
# Agent naming (producer role names in review_loop)
# ---------------------------------------------------------------------------


def test_agent_naming_captures_producer_variants(runs):
    report = mine_traces(runs)
    # fixture review_loop producers: "writer" and "analyst"
    assert report.agent_naming == {"writer": 1, "analyst": 1}


# ---------------------------------------------------------------------------
# Slowest agents
# ---------------------------------------------------------------------------


def test_slowest_agents_sorted_by_avg_elapsed_ms(runs):
    report = mine_traces(runs, slowest_agents_top_n=3)
    names = [a.agent for a in report.slowest_agents]
    # researcher_b has the largest avg (120000 ms over 1 call).
    assert names[0] == "researcher_b"
    # All entries are >= the next one's avg.
    avgs = [a.avg_elapsed_ms for a in report.slowest_agents]
    assert avgs == sorted(avgs, reverse=True)
    assert len(report.slowest_agents) <= 3


# ---------------------------------------------------------------------------
# Window filtering
# ---------------------------------------------------------------------------


def test_after_filter_drops_older_runs(runs):
    cutoff = datetime(2026, 4, 13, tzinfo=timezone.utc)
    report = mine_traces(runs, after=cutoff)
    # Only run-tool-rounds (04-13) and run-analyst-review (04-14) remain.
    assert report.total_runs == 2
    assert report.window["after"] == cutoff.isoformat()


def test_before_filter_drops_newer_runs(runs):
    cutoff = datetime(2026, 4, 12, tzinfo=timezone.utc)
    report = mine_traces(runs, before=cutoff)
    # Only run-review-2turn (04-10) and run-cli-error (04-11) fit.
    assert report.total_runs == 2


def test_audit_log_also_filtered_by_window(runs, audit_log):
    cutoff = datetime(2026, 4, 14, tzinfo=timezone.utc)
    report = mine_traces(runs, audit_log=audit_log, after=cutoff)
    # Only the 04-14 read_text_file audit record survives; the 04-12 and
    # 04-13 ones should not bump truncated_calls or tool_usage buckets
    # beyond what the events already contribute.
    # Events: no tool events fall inside the window for this fixture
    # (run-analyst-review's tool_completed timestamp is 04-14T14:00:10).
    # So read_text_file's call count comes entirely from that run.
    assert report.tool_usage["read_text_file"].calls == 1


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_report_to_dict_is_json_serializable(runs, audit_log, synthesis_log):
    report = mine_traces(runs, audit_log=audit_log, synthesis_log=synthesis_log)
    assert isinstance(report, TraceMiningReport)
    payload = report.to_dict()
    # Round-trip through JSON to confirm every nested structure is
    # JSON-safe (no datetime / enum / dataclass leaks).
    dumped = json.dumps(payload)
    restored = json.loads(dumped)
    assert restored["total_runs"] == 5
    assert "tool_usage" in restored
    assert "reviewer_convergence" in restored


# ---------------------------------------------------------------------------
# Empty input edge case
# ---------------------------------------------------------------------------


def test_mine_traces_handles_empty_inputs():
    report = mine_traces([])
    assert report.total_runs == 0
    assert report.final_status_distribution == {}
    assert report.workflow_mix == {}
    assert report.common_failure_patterns == []
    assert report.tool_usage == {}
    assert report.synthesis_stats.total_requests == 0
    assert report.reviewer_convergence.total_review_runs == 0
    assert report.window["first_run_started_at"] is None


def test_default_failure_keywords_nonempty():
    # Defensive check — if someone wipes the list we want a red test.
    assert len(DEFAULT_FAILURE_KEYWORDS) >= 5
    assert "Exceeded max_tool_rounds" in DEFAULT_FAILURE_KEYWORDS
