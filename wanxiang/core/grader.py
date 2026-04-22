"""grader — external programmatic verification of agent text output.

outcome_tagger asks "did this run complete mechanically?" (success/error/timeout).
grader asks "did the agent actually solve the task?" — orthogonal concern.

For a curriculum task with expected_outcome_keywords, a run passes if:
  - outcome != infra_error / timeout (those invalidate the attempt entirely)
  - the final agent_completed.content contains every expected keyword

Later the grader can grow: regex matches, JSON schema checks, running
the agent's code-like output, LLM-as-judge. Start with the keyword
check — it's what seed_tasks.yaml already encodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


INVALIDATING_OUTCOMES = frozenset({"infra_error", "timeout"})


@dataclass
class Grade:
    passed: bool
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "matched_keywords": list(self.matched_keywords),
            "missing_keywords": list(self.missing_keywords),
            "reason": self.reason,
        }


def grade_run(
    events: Iterable[dict[str, Any]],
    outcome: str | None,
    expected_keywords: list[str] | None,
) -> Grade:
    """Return Grade for a completed run.

    Design:
      - Runs with outcome in INVALIDATING_OUTCOMES cannot pass — the
        attempt was never real (infra crashed, timed out). Keyword
        checks against a partial/empty output would be misleading.
      - When no expected_keywords are provided, grade on outcome alone:
        'success' or 'partial' → pass, else → fail.
      - With expected_keywords, ALL must appear (case-insensitive) in
        the final agent_completed.content. Missing any → fail.
    """
    outcome_label = (outcome or "").strip().lower()

    if outcome_label in INVALIDATING_OUTCOMES:
        return Grade(
            passed=False,
            missing_keywords=list(expected_keywords or []),
            reason=f"outcome '{outcome_label}' invalidates the attempt",
        )

    cleaned_keywords = [
        str(kw).strip() for kw in (expected_keywords or []) if str(kw).strip()
    ]

    if not cleaned_keywords:
        if outcome_label in ("success", "partial"):
            return Grade(passed=True, reason=f"no keywords; outcome={outcome_label}")
        return Grade(passed=False, reason=f"no keywords and outcome={outcome_label}")

    final_text = _extract_final_agent_text(events)
    if not final_text:
        return Grade(
            passed=False,
            missing_keywords=cleaned_keywords,
            reason="no final agent output in events",
        )

    text_lower = final_text.lower()
    matched: list[str] = []
    missing: list[str] = []
    for kw in cleaned_keywords:
        if kw.lower() in text_lower:
            matched.append(kw)
        else:
            missing.append(kw)

    passed = len(missing) == 0
    return Grade(
        passed=passed,
        matched_keywords=matched,
        missing_keywords=missing,
        reason=(
            f"all {len(cleaned_keywords)} keyword(s) matched"
            if passed
            else f"{len(missing)}/{len(cleaned_keywords)} keyword(s) missing"
        ),
    )


def _extract_final_agent_text(events: Iterable[dict[str, Any]]) -> str:
    """Return the content of the last agent_completed event.

    Review-loop runs emit multiple agent_completed events (writer,
    reviewer, writer, ...). For grading purposes we want the LAST
    agent's output — that's the answer actually delivered to the user.
    """
    last = ""
    for event in events:
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
