"""Gap detector — bridges capability_gap_patterns to SkillForge synthesis candidates.

Pure analysis: no I/O, no LLM calls. Takes mining output and raw run history,
returns structured synthesis candidates.

Primary signal: "Unknown tool: <name>" patterns. Each unique missing tool name
becomes one SynthesisCandidate. Sample arguments are harvested from tool_started
events in the run history so the synthesizer has an interface hint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .trace_mining import FailurePattern


_UNKNOWN_TOOL_RE = re.compile(r"Unknown tool:\s+(\S+)")
_MAX_SAMPLE_ARGS = 3


@dataclass(slots=True)
class SynthesisCandidate:
    tool_name: str
    failure_count: int
    sample_arguments: list[dict[str, Any]]
    suggested_requirement: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "failure_count": self.failure_count,
            "sample_arguments": list(self.sample_arguments),
            "suggested_requirement": self.suggested_requirement,
        }


def detect_synthesis_candidates(
    capability_gap_patterns: list[FailurePattern],
    runs: list[dict[str, Any]],
    *,
    min_failure_count: int = 1,
) -> list[SynthesisCandidate]:
    """Return synthesis candidates derived from missing-tool failures.

    Each unique tool name that appeared in "Unknown tool: X" errors becomes
    one candidate. Results are sorted by failure_count descending.
    Tools with fewer than `min_failure_count` occurrences are dropped.
    """
    tool_counts: dict[str, int] = {}
    for pattern in capability_gap_patterns:
        if "Unknown tool:" not in pattern.keyword:
            continue
        for name in _extract_tool_names(pattern):
            tool_counts[name] = tool_counts.get(name, 0) + pattern.count

    if not tool_counts:
        return []

    tool_args: dict[str, list[dict[str, Any]]] = {n: [] for n in tool_counts}
    for run in runs:
        for event in run.get("events") or []:
            if not isinstance(event, dict):
                continue
            if str(event.get("type", "")) != "tool_started":
                continue
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            tool = str(data.get("tool", ""))
            if tool in tool_args and len(tool_args[tool]) < _MAX_SAMPLE_ARGS:
                args = data.get("arguments")
                if isinstance(args, dict):
                    tool_args[tool].append(dict(args))

    candidates = []
    for tool_name, count in tool_counts.items():
        if count < min_failure_count:
            continue
        sample_args = tool_args.get(tool_name, [])
        candidates.append(
            SynthesisCandidate(
                tool_name=tool_name,
                failure_count=count,
                sample_arguments=sample_args,
                suggested_requirement=_build_requirement(tool_name, count, sample_args),
            )
        )

    candidates.sort(key=lambda c: c.failure_count, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_tool_names(pattern: FailurePattern) -> list[str]:
    """Parse tool names out of an 'Unknown tool:' pattern's example and keyword."""
    for text in (pattern.example, pattern.keyword):
        found = _UNKNOWN_TOOL_RE.findall(text)
        if found:
            return list(dict.fromkeys(found))
    return []


def _build_requirement(
    tool_name: str,
    failure_count: int,
    sample_args: list[dict[str, Any]],
) -> str:
    parts = [
        f"Implement a tool named '{tool_name}'.",
        f"It was called {failure_count} time(s) but was not found in the registry.",
    ]
    if sample_args:
        args_repr = "; ".join(str(a) for a in sample_args[:2])
        parts.append(
            f"Sample call arguments: {args_repr}. "
            "Infer the input/output contract from these arguments."
        )
    else:
        parts.append("No sample arguments available — infer intent from the tool name.")
    return " ".join(parts)
