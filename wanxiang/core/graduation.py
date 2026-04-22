"""graduation — pure-data judge of "is the system ready to move up a level?"

Reads storage only. No LLM calls, no side effects beyond reading the DB.
Autoschool (when implemented) calls evaluate() after each tick and
promotes the growth_state.current_level when the criterion fires.

Criteria per level transition:
  L0 → L1: last N graded L0 runs, pass rate ≥ threshold
  L1 → L2: last N graded L1 runs, pass rate ≥ threshold + skill library ≥ M
  L2 → L3: last N graded L2 runs, pass rate ≥ threshold + all 3 workflows exercised
  L3 → L4: manual (human gate — system cannot auto-promote itself to real service)

Thresholds are defaults; callers can override. The point of pure-data
graduation is that "how do we know we're ready?" stays quantitative and
auditable, not an LLM judgment call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .storage import Storage


@dataclass
class GraduationResult:
    current_level: int
    eligible: bool
    next_level: int | None
    criterion: str
    observed: dict[str, float | int] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "current_level": self.current_level,
            "eligible": self.eligible,
            "next_level": self.next_level,
            "criterion": self.criterion,
            "observed": dict(self.observed),
            "reason": self.reason,
        }


# Defaults — see project_growth_plan.md for rationale.
DEFAULT_WINDOW = 20
DEFAULT_PASS_THRESHOLD = 15        # 15/20 for L0-L1
DEFAULT_L2_PASS_THRESHOLD = 12     # 12/20 for L2
MIN_SKILL_LIBRARY_FOR_L2 = 5


def evaluate(
    storage: Storage,
    current_level: int,
    *,
    window: int = DEFAULT_WINDOW,
    l0_threshold: int = DEFAULT_PASS_THRESHOLD,
    l1_threshold: int = DEFAULT_PASS_THRESHOLD,
    l2_threshold: int = DEFAULT_L2_PASS_THRESHOLD,
    min_skills_for_l2: int = MIN_SKILL_LIBRARY_FOR_L2,
) -> GraduationResult:
    """Decide whether the system can promote from current_level.

    Returns GraduationResult with eligible=True/False + observed metrics.
    Caller (autoschool) applies the promotion by writing to growth_state.
    """
    if current_level >= 3:
        return GraduationResult(
            current_level=current_level,
            eligible=False,
            next_level=None,
            criterion="manual",
            reason="L3+ requires human approval; auto-graduation disabled",
        )

    graded_runs = [
        r for r in storage.list_runs(limit=10_000, level=current_level)
        if r.graded_pass is not None
    ]
    sample = graded_runs[:window]
    passed = sum(1 for r in sample if r.graded_pass)

    threshold_map = {0: l0_threshold, 1: l1_threshold, 2: l2_threshold}
    required = threshold_map[current_level]

    observed: dict[str, float | int] = {
        "graded_sample_size": len(sample),
        "passed": passed,
        "pass_rate": (passed / len(sample)) if sample else 0.0,
        "window": window,
        "required_passes": required,
    }
    criterion = f"L{current_level}→L{current_level + 1}: {required}/{window} graded runs pass"

    if len(sample) < window:
        return GraduationResult(
            current_level=current_level,
            eligible=False,
            next_level=current_level + 1,
            criterion=criterion,
            observed=observed,
            reason=f"only {len(sample)}/{window} graded runs at level {current_level}",
        )

    if passed < required:
        return GraduationResult(
            current_level=current_level,
            eligible=False,
            next_level=current_level + 1,
            criterion=criterion,
            observed=observed,
            reason=f"pass rate {passed}/{len(sample)} below threshold {required}",
        )

    # L1 → L2 extra gate: skill library size.
    if current_level == 1:
        skill_count = _count_synthesized_tools(storage)
        observed["synthesized_tools"] = skill_count
        if skill_count < min_skills_for_l2:
            return GraduationResult(
                current_level=current_level,
                eligible=False,
                next_level=2,
                criterion=criterion + f" + skills ≥ {min_skills_for_l2}",
                observed=observed,
                reason=f"skill library has {skill_count} tools, need ≥ {min_skills_for_l2}",
            )

    return GraduationResult(
        current_level=current_level,
        eligible=True,
        next_level=current_level + 1,
        criterion=criterion,
        observed=observed,
        reason=f"pass rate {passed}/{len(sample)} meets threshold {required}",
    )


def _count_synthesized_tools(storage: Storage) -> int:
    """Synthesized tools are registered from runs where forge succeeded.

    Phase 2 uses the proxy of counting distinct synthesized-source runs
    in the DB. A cleaner count (reading skills/ directory or querying
    the TierManager) can replace this later without changing the
    graduation API.
    """
    # Proxy: count runs whose source is 'autoschool' and outcome is 'success'.
    # The real skill registry isn't queryable from storage; when autoschool
    # lands we'll route skills/ file count through here instead.
    return sum(
        1
        for r in storage.list_runs(limit=10_000, source="autoschool")
        if r.outcome == "success"
    )
