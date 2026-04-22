"""Tests for grader — programmatic verification of agent text output."""
from __future__ import annotations

import pytest

from wanxiang.core.grader import Grade, grade_run


def _final_text_event(content: str) -> dict:
    return {"type": "agent_completed", "data": {"agent": "responder", "content": content}}


class TestKeywordMatching:
    def test_all_keywords_match_passes(self):
        events = [_final_text_event("The answer is 42 and seven roses.")]
        g = grade_run(events, "success", ["42", "roses"])
        assert g.passed is True
        assert set(g.matched_keywords) == {"42", "roses"}
        assert g.missing_keywords == []

    def test_missing_keyword_fails(self):
        events = [_final_text_event("42 only, no flowers.")]
        g = grade_run(events, "success", ["42", "roses"])
        assert g.passed is False
        assert "roses" in g.missing_keywords
        assert "42" in g.matched_keywords

    def test_case_insensitive(self):
        events = [_final_text_event("Hello WORLD")]
        g = grade_run(events, "success", ["hello", "world"])
        assert g.passed is True

    def test_empty_output_fails_when_keywords_expected(self):
        events = [_final_text_event("")]
        g = grade_run(events, "success", ["something"])
        assert g.passed is False
        assert "no final agent output" in g.reason

    def test_whitespace_keywords_are_stripped(self):
        events = [_final_text_event("contains x")]
        g = grade_run(events, "success", ["  ", "x", "   "])
        assert g.passed is True


class TestNoKeywords:
    def test_success_outcome_passes_without_keywords(self):
        g = grade_run([], "success", None)
        assert g.passed is True
        assert "outcome=success" in g.reason

    def test_partial_outcome_passes_without_keywords(self):
        g = grade_run([], "partial", [])
        assert g.passed is True

    def test_unknown_outcome_fails_without_keywords(self):
        g = grade_run([], "unknown", None)
        assert g.passed is False


class TestInvalidatingOutcomes:
    def test_infra_error_never_passes(self):
        events = [_final_text_event("the answer is 42")]
        # Even with keywords matching, infra errors invalidate the run.
        g = grade_run(events, "infra_error", ["42"])
        assert g.passed is False
        assert "infra_error" in g.reason

    def test_timeout_never_passes(self):
        events = [_final_text_event("done")]
        g = grade_run(events, "timeout", None)
        assert g.passed is False
        assert "timeout" in g.reason


class TestFinalTextExtraction:
    def test_last_agent_completed_is_used(self):
        # Review loop: writer → reviewer → writer(revised) → reviewer.
        events = [
            {"type": "agent_completed", "data": {"agent": "writer", "content": "draft v1"}},
            {"type": "agent_completed", "data": {"agent": "reviewer", "content": "needs revision"}},
            {"type": "agent_completed", "data": {"agent": "writer", "content": "final with keyword"}},
        ]
        g = grade_run(events, "success", ["keyword"])
        assert g.passed is True

    def test_content_preview_fallback(self):
        events = [{"type": "agent_completed", "data": {"agent": "x", "content_preview": "short"}}]
        g = grade_run(events, "success", ["short"])
        assert g.passed is True

    def test_non_agent_events_ignored(self):
        events = [
            {"type": "tool_completed", "data": {"content": "should be ignored"}},
            _final_text_event("real output with keyword"),
        ]
        g = grade_run(events, "success", ["keyword"])
        assert g.passed is True

    def test_empty_events(self):
        g = grade_run([], "success", ["x"])
        assert g.passed is False


class TestGradeSerialization:
    def test_to_dict_shape(self):
        g = Grade(passed=True, matched_keywords=["a"], missing_keywords=[], reason="ok")
        d = g.to_dict()
        assert d == {"passed": True, "matched_keywords": ["a"], "missing_keywords": [], "reason": "ok"}
