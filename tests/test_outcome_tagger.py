"""Tests for outcome_tagger — classifying completed runs into outcome labels."""
from __future__ import annotations

from wanxiang.core.outcome_tagger import (
    OUTCOME_CAPABILITY_GAP,
    OUTCOME_INFRA_ERROR,
    OUTCOME_PARTIAL,
    OUTCOME_SUCCESS,
    OUTCOME_TIMEOUT,
    OUTCOME_UNKNOWN,
    tag_run,
)


class TestSuccess:
    def test_pure_success(self):
        events = [
            {"type": "run_started", "data": {}},
            {"type": "tool_completed", "data": {"success": True}},
            {"type": "run_completed", "data": {"status": "success"}},
        ]
        assert tag_run(events, "success") == OUTCOME_SUCCESS

    def test_empty_events_with_success_status(self):
        assert tag_run([], "success") == OUTCOME_SUCCESS


class TestInfraError:
    def test_api_key_missing(self):
        events = [
            {
                "type": "agent_completed",
                "data": {"status": "error", "message": "ANTHROPIC_API_KEY not configured"},
            }
        ]
        assert tag_run(events, "error") == OUTCOME_INFRA_ERROR

    def test_connection_error(self):
        events = [{"type": "tool_completed", "data": {"success": False, "content": "ConnectionError: remote unreachable"}}]
        assert tag_run(events, "error") == OUTCOME_INFRA_ERROR

    def test_max_retries_exceeded(self):
        events = [{"type": "agent_completed", "data": {"status": "error", "error": "Exceeded max_retries after 3 attempts"}}]
        assert tag_run(events, "error") == OUTCOME_INFRA_ERROR


class TestCapabilityGap:
    def test_unknown_tool(self):
        events = [
            {
                "type": "tool_completed",
                "data": {"success": False, "content": "Unknown tool: calc_roi — not in registry"},
            }
        ]
        assert tag_run(events, "error") == OUTCOME_CAPABILITY_GAP

    def test_invalid_arguments(self):
        events = [{"type": "tool_completed", "data": {"success": False, "content": "Invalid arguments: missing field 'x'"}}]
        assert tag_run(events, "error") == OUTCOME_CAPABILITY_GAP

    def test_max_tool_rounds_exceeded(self):
        events = [{"type": "agent_completed", "data": {"status": "error", "message": "Exceeded max_tool_rounds=15"}}]
        assert tag_run(events, "error") == OUTCOME_CAPABILITY_GAP


class TestTimeout:
    def test_tool_timed_out(self):
        events = [{"type": "tool_completed", "data": {"success": False, "content": "Tool timed out after 30s"}}]
        assert tag_run(events, "error") == OUTCOME_TIMEOUT

    def test_llm_call_exceeded(self):
        # This specific phrase is BOTH an infra keyword and a timeout keyword;
        # timeout wins because it's checked first.
        events = [{"type": "agent_completed", "data": {"status": "error", "error": "LLM call exceeded timeout of 60s"}}]
        assert tag_run(events, "error") == OUTCOME_TIMEOUT

    def test_generic_timeout_error(self):
        events = [{"type": "tool_completed", "data": {"success": False, "error": "TimeoutError during handler exec"}}]
        assert tag_run(events, "error") == OUTCOME_TIMEOUT


class TestPartial:
    def test_success_status_but_tool_error_somewhere(self):
        # Run recovered after a failed tool call — final_status is success but error signal exists.
        events = [
            {"type": "tool_completed", "data": {"success": False, "content": "weird glitch, retried"}},
            {"type": "run_completed", "data": {"status": "success"}},
        ]
        # Falls through infra/gap/timeout checks (generic text), final_status=success + error
        # present → partial.
        assert tag_run(events, "success") == OUTCOME_PARTIAL

    def test_needs_revision_final_status_maps_to_partial(self):
        # review_loop exhausted max_iterations with reviewer still unhappy.
        events = [
            {"type": "agent_completed", "data": {"agent": "writer", "status": "success"}},
            {"type": "agent_completed", "data": {"agent": "reviewer", "status": "needs_revision"}},
        ]
        assert tag_run(events, "needs_revision") == OUTCOME_PARTIAL

    def test_needs_revision_with_no_events(self):
        assert tag_run([], "needs_revision") == OUTCOME_PARTIAL


class TestUnknown:
    def test_no_success_no_errors(self):
        events = [{"type": "run_completed", "data": {}}]
        assert tag_run(events, None) == OUTCOME_UNKNOWN

    def test_malformed_events_do_not_crash(self):
        events = [None, "not-a-dict", {"type": "x"}, {"type": "y", "data": "not-a-dict"}]  # type: ignore
        assert tag_run(events, "success") == OUTCOME_SUCCESS


class TestDecisionOrder:
    """First-match-wins order: timeout → infra → gap → success."""

    def test_timeout_beats_infra(self):
        events = [
            {"type": "tool_completed", "data": {"success": False, "content": "Tool timed out"}},
            {"type": "tool_completed", "data": {"success": False, "content": "ConnectionError"}},
        ]
        assert tag_run(events, "error") == OUTCOME_TIMEOUT

    def test_infra_beats_gap(self):
        events = [
            {"type": "tool_completed", "data": {"success": False, "content": "ANTHROPIC_API_KEY missing"}},
            {"type": "tool_completed", "data": {"success": False, "content": "Unknown tool: foo"}},
        ]
        assert tag_run(events, "error") == OUTCOME_INFRA_ERROR
