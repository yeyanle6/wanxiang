"""Tool trust tier system — runtime confidence grading for tools.

Every tool starts at Level 0. As it accumulates successful calls across
independent runs, it can be promoted up to Level 3. A sliding window
of recent results triggers automatic demotion when failure rate exceeds
a threshold — tiers go down as easily as they go up.

Tier levels:
  0  sandbox-passed or never-used (no real-world evidence yet)
  1  first real-task SUCCESS (minimum viable trust)
  2  ≥3 distinct runs with success + window success rate ≥ 90%
  3  used as a dependency by a non-primary agent in a pipeline

Demotion:
  If the sliding window (default: last 10 calls) has ≥ threshold
  failures (default: 3), drop one level. Can cascade on repeated
  failures.

This module is pure data + logic, no I/O, no LLM. Integration point
is RunManager — it calls `record_result` after each `tool_completed`
event, outside the immutable-core ToolRegistry.execute pipeline.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TierChange:
    tool_name: str
    old_level: int
    new_level: int
    reason: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "old_level": self.old_level,
            "new_level": self.new_level,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ToolTier:
    tool_name: str
    level: int = 0
    total_calls: int = 0
    recent_results: deque[bool] = field(default_factory=lambda: deque(maxlen=10))
    distinct_run_ids: set[str] = field(default_factory=set)
    successful_run_ids: set[str] = field(default_factory=set)
    last_updated: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    tier_history: list[TierChange] = field(default_factory=list)

    @property
    def window_size(self) -> int:
        return self.recent_results.maxlen or 10

    @property
    def recent_failures(self) -> int:
        return sum(1 for r in self.recent_results if not r)

    @property
    def recent_successes(self) -> int:
        return sum(1 for r in self.recent_results if r)

    @property
    def window_success_rate(self) -> float:
        if not self.recent_results:
            return 0.0
        return self.recent_successes / len(self.recent_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "level": self.level,
            "total_calls": self.total_calls,
            "window_size": self.window_size,
            "recent_failures": self.recent_failures,
            "recent_successes": self.recent_successes,
            "window_success_rate": round(self.window_success_rate, 3),
            "distinct_runs": len(self.distinct_run_ids),
            "successful_runs": len(self.successful_run_ids),
            "last_updated": self.last_updated.isoformat(),
            "tier_history": [ch.to_dict() for ch in self.tier_history],
        }


# ---------------------------------------------------------------------------
# TierManager
# ---------------------------------------------------------------------------


class TierManager:
    """Tracks per-tool trust tiers with sliding-window demotion.

    Call `record_result` after each tool execution. The manager
    handles promotion / demotion logic and keeps an audit trail
    of tier transitions.
    """

    def __init__(
        self,
        *,
        window_size: int = 10,
        downgrade_threshold: int = 3,
        upgrade_2_min_runs: int = 3,
        upgrade_2_min_success_rate: float = 0.9,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if downgrade_threshold < 1:
            raise ValueError("downgrade_threshold must be >= 1")
        self._tiers: dict[str, ToolTier] = {}
        self._changes: list[TierChange] = []
        self.window_size = window_size
        self.downgrade_threshold = downgrade_threshold
        self.upgrade_2_min_runs = upgrade_2_min_runs
        self.upgrade_2_min_success_rate = upgrade_2_min_success_rate

    def _get_or_create(self, tool_name: str) -> ToolTier:
        if tool_name not in self._tiers:
            self._tiers[tool_name] = ToolTier(
                tool_name=tool_name,
                recent_results=deque(maxlen=self.window_size),
            )
        return self._tiers[tool_name]

    def record_result(
        self,
        tool_name: str,
        success: bool,
        run_id: str,
    ) -> TierChange | None:
        """Record a tool execution result and return a TierChange if the
        tier changed, or None if it stayed the same.
        """
        tier = self._get_or_create(tool_name)
        tier.total_calls += 1
        tier.recent_results.append(success)
        tier.distinct_run_ids.add(run_id)
        if success:
            tier.successful_run_ids.add(run_id)
        tier.last_updated = datetime.now(timezone.utc)

        old_level = tier.level

        # Check demotion first — failures take priority.
        self._check_downgrade(tier)

        # Then check promotion.
        if tier.level == old_level:
            self._check_upgrade(tier)

        if tier.level != old_level:
            change = TierChange(
                tool_name=tool_name,
                old_level=old_level,
                new_level=tier.level,
                reason=self._describe_change(tier, old_level),
                timestamp=tier.last_updated,
            )
            tier.tier_history.append(change)
            self._changes.append(change)
            return change

        return None

    def record_dependency_use(self, tool_name: str, run_id: str) -> TierChange | None:
        """Record that this tool was used as a dependency by a non-primary
        agent. This is the signal for Level 2 → 3 promotion.
        """
        tier = self._get_or_create(tool_name)
        if tier.level < 2:
            return None

        old_level = tier.level
        tier.level = 3
        tier.last_updated = datetime.now(timezone.utc)

        if tier.level != old_level:
            change = TierChange(
                tool_name=tool_name,
                old_level=old_level,
                new_level=3,
                reason=(
                    f"used as dependency in run {run_id} "
                    f"by non-primary agent (promoted to trusted)"
                ),
                timestamp=tier.last_updated,
            )
            tier.tier_history.append(change)
            self._changes.append(change)
            return change
        return None

    def initialize_tool(self, tool_name: str, level: int = 0) -> None:
        """Pre-seed a tier entry (e.g. after SkillForge registration)."""
        tier = self._get_or_create(tool_name)
        tier.level = max(0, min(3, level))
        tier.last_updated = datetime.now(timezone.utc)

    def get_tier(self, tool_name: str) -> ToolTier | None:
        return self._tiers.get(tool_name)

    def get_all_tiers(self) -> dict[str, ToolTier]:
        return dict(self._tiers)

    def get_recent_changes(self, limit: int = 50) -> list[TierChange]:
        return list(self._changes[-limit:])

    def get_tier_summary(self) -> dict[str, Any]:
        """Snapshot suitable for JSON serialization (mining / API)."""
        by_level: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        for tier in self._tiers.values():
            by_level[tier.level] = by_level.get(tier.level, 0) + 1
        return {
            "total_tools_tracked": len(self._tiers),
            "by_level": by_level,
            "recent_changes": [ch.to_dict() for ch in self._changes[-20:]],
            "tools": {
                name: tier.to_dict() for name, tier in self._tiers.items()
            },
        }

    # -------------------------------------------------------------------
    # Internal logic
    # -------------------------------------------------------------------

    def _check_downgrade(self, tier: ToolTier) -> None:
        if tier.level <= 0:
            return
        if tier.recent_failures >= self.downgrade_threshold:
            tier.level = max(0, tier.level - 1)

    def _check_upgrade(self, tier: ToolTier) -> None:
        if tier.level == 0:
            self._try_promote_0_to_1(tier)
        elif tier.level == 1:
            self._try_promote_1_to_2(tier)
        # 2 → 3 is only via record_dependency_use (external signal).

    def _try_promote_0_to_1(self, tier: ToolTier) -> None:
        if tier.total_calls >= 1 and tier.recent_results and tier.recent_results[-1]:
            tier.level = 1

    def _try_promote_1_to_2(self, tier: ToolTier) -> None:
        if len(tier.successful_run_ids) < self.upgrade_2_min_runs:
            return
        if len(tier.recent_results) < self.upgrade_2_min_runs:
            return
        if tier.window_success_rate >= self.upgrade_2_min_success_rate:
            tier.level = 2

    def _describe_change(self, tier: ToolTier, old_level: int) -> str:
        if tier.level < old_level:
            return (
                f"downgraded: {tier.recent_failures} failures in last "
                f"{len(tier.recent_results)} calls "
                f"(threshold={self.downgrade_threshold})"
            )
        if tier.level == 1:
            return "first successful real-task call"
        if tier.level == 2:
            return (
                f"promoted: {len(tier.successful_run_ids)} distinct successful runs, "
                f"window success rate {tier.window_success_rate:.0%}"
            )
        if tier.level == 3:
            return "used as dependency by non-primary agent"
        return f"tier changed {old_level} → {tier.level}"
