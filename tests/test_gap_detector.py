"""Tests for Phase 6.4: gap detector — bridges capability_gap_patterns to
synthesis candidates.

Covers: unknown-tool extraction, sample-argument harvesting from run history,
candidate sorting/filtering, requirement string generation, edge cases.
"""
from __future__ import annotations

from wanxiang.core.gap_detector import (
    SynthesisCandidate,
    detect_synthesis_candidates,
    _build_requirement,
    _extract_tool_names,
)
from wanxiang.core.trace_mining import FailurePattern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gap(keyword: str, count: int, example: str = "") -> FailurePattern:
    return FailurePattern(
        keyword=keyword, count=count, example=example, category="capability_gap"
    )


def _run_with_tool_started(run_id: str, tool: str, args: dict) -> dict:
    return {
        "run_id": run_id,
        "events": [
            {
                "type": "tool_started",
                "data": {"agent": "researcher", "tool": tool, "arguments": args},
            }
        ],
    }


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------


class TestDetectSynthesisCandidates:
    def test_no_patterns_returns_empty(self):
        assert detect_synthesis_candidates([], []) == []

    def test_non_unknown_tool_patterns_ignored(self):
        patterns = [_gap("Exceeded max_tool_rounds", 5)]
        assert detect_synthesis_candidates(patterns, []) == []

    def test_single_unknown_tool_from_example(self):
        patterns = [
            _gap("Unknown tool:", 3, "Unknown tool: search_arxiv — not found in registry")
        ]
        candidates = detect_synthesis_candidates(patterns, [])
        assert len(candidates) == 1
        assert candidates[0].tool_name == "search_arxiv"
        assert candidates[0].failure_count == 3

    def test_tool_name_parsed_from_keyword_if_example_empty(self):
        # If example is empty, fall back to keyword itself
        patterns = [_gap("Unknown tool: fetch_price", 2, "")]
        candidates = detect_synthesis_candidates(patterns, [])
        assert len(candidates) == 1
        assert candidates[0].tool_name == "fetch_price"

    def test_min_failure_count_filters(self):
        patterns = [
            _gap("Unknown tool:", 1, "Unknown tool: rare_tool"),
            _gap("Unknown tool:", 5, "Unknown tool: common_tool"),
        ]
        candidates = detect_synthesis_candidates(patterns, [], min_failure_count=3)
        assert len(candidates) == 1
        assert candidates[0].tool_name == "common_tool"

    def test_candidates_sorted_by_failure_count_desc(self):
        patterns = [
            _gap("Unknown tool:", 2, "Unknown tool: tool_b"),
            _gap("Unknown tool:", 7, "Unknown tool: tool_a"),
            _gap("Unknown tool:", 4, "Unknown tool: tool_c"),
        ]
        candidates = detect_synthesis_candidates(patterns, [])
        counts = [c.failure_count for c in candidates]
        assert counts == sorted(counts, reverse=True)
        assert candidates[0].tool_name == "tool_a"

    def test_multiple_patterns_same_tool_accumulate(self):
        # Two FailurePattern entries both pointing to the same tool
        # (e.g. from different runs/windows) — counts should add up.
        patterns = [
            _gap("Unknown tool:", 3, "Unknown tool: my_tool"),
            _gap("Unknown tool:", 2, "Unknown tool: my_tool"),
        ]
        candidates = detect_synthesis_candidates(patterns, [])
        assert len(candidates) == 1
        assert candidates[0].failure_count == 5


# ---------------------------------------------------------------------------
# Sample argument harvesting
# ---------------------------------------------------------------------------


class TestSampleArgumentHarvesting:
    def test_tool_started_args_collected(self):
        patterns = [_gap("Unknown tool:", 2, "Unknown tool: calc_roi")]
        runs = [
            _run_with_tool_started("r1", "calc_roi", {"revenue": 100, "cost": 80}),
        ]
        candidates = detect_synthesis_candidates(patterns, runs)
        assert len(candidates[0].sample_arguments) == 1
        assert candidates[0].sample_arguments[0]["revenue"] == 100

    def test_max_3_sample_args(self):
        patterns = [_gap("Unknown tool:", 5, "Unknown tool: calc_roi")]
        runs = [
            _run_with_tool_started(f"r{i}", "calc_roi", {"x": i})
            for i in range(5)
        ]
        candidates = detect_synthesis_candidates(patterns, runs)
        assert len(candidates[0].sample_arguments) <= 3

    def test_args_from_different_tools_not_mixed(self):
        patterns = [
            _gap("Unknown tool:", 2, "Unknown tool: tool_x"),
            _gap("Unknown tool:", 2, "Unknown tool: tool_y"),
        ]
        runs = [
            _run_with_tool_started("r1", "tool_x", {"param": "alpha"}),
            _run_with_tool_started("r2", "tool_y", {"param": "beta"}),
        ]
        candidates = detect_synthesis_candidates(patterns, runs)
        by_name = {c.tool_name: c for c in candidates}
        assert by_name["tool_x"].sample_arguments[0]["param"] == "alpha"
        assert by_name["tool_y"].sample_arguments[0]["param"] == "beta"

    def test_non_dict_args_skipped(self):
        patterns = [_gap("Unknown tool:", 1, "Unknown tool: my_tool")]
        runs = [
            {
                "run_id": "r1",
                "events": [
                    {
                        "type": "tool_started",
                        "data": {"agent": "a", "tool": "my_tool", "arguments": "not a dict"},
                    }
                ],
            }
        ]
        candidates = detect_synthesis_candidates(patterns, runs)
        assert candidates[0].sample_arguments == []


# ---------------------------------------------------------------------------
# Requirement string
# ---------------------------------------------------------------------------


class TestBuildRequirement:
    def test_contains_tool_name(self):
        req = _build_requirement("my_tool", 3, [])
        assert "my_tool" in req

    def test_contains_failure_count(self):
        req = _build_requirement("my_tool", 7, [])
        assert "7" in req

    def test_with_sample_args_mentions_arguments(self):
        req = _build_requirement("my_tool", 2, [{"key": "val"}])
        assert "Sample call arguments" in req or "arguments" in req.lower()

    def test_without_sample_args_mentions_name(self):
        req = _build_requirement("my_tool", 1, [])
        assert "tool name" in req.lower() or "infer" in req.lower()


# ---------------------------------------------------------------------------
# SynthesisCandidate serialization
# ---------------------------------------------------------------------------


class TestSynthesisCandidateSerialization:
    def test_to_dict_contains_required_keys(self):
        c = SynthesisCandidate(
            tool_name="echo_v2",
            failure_count=4,
            sample_arguments=[{"x": 1}],
            suggested_requirement="Implement echo_v2.",
        )
        d = c.to_dict()
        assert d["tool_name"] == "echo_v2"
        assert d["failure_count"] == 4
        assert d["sample_arguments"] == [{"x": 1}]
        assert "suggested_requirement" in d


# ---------------------------------------------------------------------------
# Extract tool names helper
# ---------------------------------------------------------------------------


class TestExtractToolNames:
    def test_extracts_from_example(self):
        pattern = _gap("Unknown tool:", 1, "call to Unknown tool: arxiv_search failed")
        names = _extract_tool_names(pattern)
        assert names == ["arxiv_search"]

    def test_extracts_from_keyword_if_example_empty(self):
        pattern = _gap("Unknown tool: my_tool", 1, "")
        names = _extract_tool_names(pattern)
        assert names == ["my_tool"]

    def test_no_match_returns_empty(self):
        pattern = _gap("Exceeded max_tool_rounds", 2, "some error text")
        names = _extract_tool_names(pattern)
        assert names == []

    def test_deduplicates_names(self):
        pattern = _gap(
            "Unknown tool:", 3,
            "Unknown tool: dup_tool ... Unknown tool: dup_tool again"
        )
        names = _extract_tool_names(pattern)
        assert names.count("dup_tool") == 1
