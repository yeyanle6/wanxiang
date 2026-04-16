"""Tests for the tool trust tier system (wanxiang.core.tier).

Covers: promotion 0→1→2→3, demotion via sliding window, edge cases
(repeated failures, mixed results, dependency-use signal), serialization,
and configuration validation.
"""
from __future__ import annotations

import json

import pytest

from wanxiang.core.tier import TierChange, TierManager, ToolTier


# ---------------------------------------------------------------------------
# Promotion: 0 → 1 (first real-task SUCCESS)
# ---------------------------------------------------------------------------


class TestPromote0To1:
    def test_first_success_promotes_to_1(self):
        mgr = TierManager()
        change = mgr.record_result("my_tool", success=True, run_id="run-1")
        assert change is not None
        assert change.old_level == 0
        assert change.new_level == 1
        assert "first successful" in change.reason

    def test_first_failure_stays_at_0(self):
        mgr = TierManager()
        change = mgr.record_result("my_tool", success=False, run_id="run-1")
        assert change is None
        assert mgr.get_tier("my_tool").level == 0

    def test_failure_then_success_promotes(self):
        mgr = TierManager()
        mgr.record_result("t", success=False, run_id="run-1")
        change = mgr.record_result("t", success=True, run_id="run-2")
        assert change is not None
        assert change.new_level == 1


# ---------------------------------------------------------------------------
# Promotion: 1 → 2 (≥3 distinct runs + 90% window success rate)
# ---------------------------------------------------------------------------


class TestPromote1To2:
    def test_three_distinct_runs_promotes_to_2(self):
        mgr = TierManager()
        mgr.record_result("t", success=True, run_id="r1")  # 0 → 1
        mgr.record_result("t", success=True, run_id="r2")
        change = mgr.record_result("t", success=True, run_id="r3")
        assert change is not None
        assert change.new_level == 2
        assert "distinct successful runs" in change.reason

    def test_same_run_id_does_not_count_as_distinct(self):
        mgr = TierManager()
        mgr.record_result("t", success=True, run_id="r1")  # 0 → 1
        mgr.record_result("t", success=True, run_id="r1")
        change = mgr.record_result("t", success=True, run_id="r1")
        assert change is None
        assert mgr.get_tier("t").level == 1

    def test_low_success_rate_blocks_promotion(self):
        mgr = TierManager(window_size=10)
        mgr.record_result("t", success=True, run_id="r1")  # 0 → 1
        # 2 failures in a window of 4 = 50% success rate < 90%
        mgr.record_result("t", success=False, run_id="r2")
        mgr.record_result("t", success=False, run_id="r3")
        change = mgr.record_result("t", success=True, run_id="r4")
        assert change is None
        assert mgr.get_tier("t").level == 1

    def test_enough_successes_after_early_failures_can_promote(self):
        mgr = TierManager(window_size=10)
        mgr.record_result("t", success=True, run_id="r1")  # 0 → 1
        mgr.record_result("t", success=False, run_id="r2")
        # Fill window with successes to push rate above 90%
        for i in range(3, 12):
            mgr.record_result("t", success=True, run_id=f"r{i}")
        tier = mgr.get_tier("t")
        assert tier.level == 2
        assert tier.window_success_rate >= 0.9


# ---------------------------------------------------------------------------
# Promotion: 2 → 3 (dependency use signal)
# ---------------------------------------------------------------------------


class TestPromote2To3:
    def _make_level_2(self, mgr: TierManager, name: str = "t") -> None:
        mgr.record_result(name, success=True, run_id="r1")
        mgr.record_result(name, success=True, run_id="r2")
        mgr.record_result(name, success=True, run_id="r3")
        assert mgr.get_tier(name).level == 2

    def test_dependency_use_promotes_to_3(self):
        mgr = TierManager()
        self._make_level_2(mgr)
        change = mgr.record_dependency_use("t", run_id="r4")
        assert change is not None
        assert change.new_level == 3
        assert "dependency" in change.reason

    def test_dependency_use_on_level_1_is_ignored(self):
        mgr = TierManager()
        mgr.record_result("t", success=True, run_id="r1")  # 0 → 1
        change = mgr.record_dependency_use("t", run_id="r2")
        assert change is None
        assert mgr.get_tier("t").level == 1

    def test_dependency_use_on_level_0_is_ignored(self):
        mgr = TierManager()
        change = mgr.record_dependency_use("t", run_id="r1")
        assert change is None

    def test_repeated_dependency_use_on_level_3_is_noop(self):
        mgr = TierManager()
        self._make_level_2(mgr)
        mgr.record_dependency_use("t", run_id="r4")
        change = mgr.record_dependency_use("t", run_id="r5")
        assert change is None
        assert mgr.get_tier("t").level == 3


# ---------------------------------------------------------------------------
# Demotion: sliding window failure threshold
# ---------------------------------------------------------------------------


class TestDemotion:
    def test_three_failures_in_window_demotes(self):
        mgr = TierManager(window_size=10, downgrade_threshold=3)
        # Promote to level 1
        mgr.record_result("t", success=True, run_id="r1")
        assert mgr.get_tier("t").level == 1
        # 3 failures should demote
        mgr.record_result("t", success=False, run_id="r2")
        mgr.record_result("t", success=False, run_id="r3")
        change = mgr.record_result("t", success=False, run_id="r4")
        assert change is not None
        assert change.old_level == 1
        assert change.new_level == 0
        assert "downgraded" in change.reason

    def test_two_failures_does_not_demote(self):
        mgr = TierManager(window_size=10, downgrade_threshold=3)
        mgr.record_result("t", success=True, run_id="r1")  # 0 → 1
        mgr.record_result("t", success=False, run_id="r2")
        change = mgr.record_result("t", success=False, run_id="r3")
        assert change is None
        assert mgr.get_tier("t").level == 1

    def test_level_0_cannot_demote_further(self):
        mgr = TierManager(window_size=5, downgrade_threshold=2)
        for i in range(5):
            mgr.record_result("t", success=False, run_id=f"r{i}")
        assert mgr.get_tier("t").level == 0

    def test_demotion_from_level_2(self):
        mgr = TierManager(window_size=10, downgrade_threshold=3)
        # Promote to level 2
        for i in range(1, 4):
            mgr.record_result("t", success=True, run_id=f"r{i}")
        assert mgr.get_tier("t").level == 2
        # 3 failures should demote to 1
        for i in range(4, 7):
            mgr.record_result("t", success=False, run_id=f"r{i}")
        assert mgr.get_tier("t").level == 1

    def test_demotion_priority_over_promotion(self):
        """If window has enough failures AND enough distinct runs,
        demotion fires first — failures take priority."""
        mgr = TierManager(window_size=10, downgrade_threshold=3)
        mgr.record_result("t", success=True, run_id="r1")  # 0 → 1
        mgr.record_result("t", success=True, run_id="r2")
        # 3 failures from 3 distinct new runs
        mgr.record_result("t", success=False, run_id="r3")
        mgr.record_result("t", success=False, run_id="r4")
        change = mgr.record_result("t", success=False, run_id="r5")
        # Despite 5 distinct runs, demotion wins
        assert change is not None
        assert change.new_level == 0

    def test_sliding_window_forgets_old_results(self):
        mgr = TierManager(window_size=5, downgrade_threshold=3)
        mgr.record_result("t", success=True, run_id="r1")  # 0 → 1
        # 2 failures
        mgr.record_result("t", success=False, run_id="r2")
        mgr.record_result("t", success=False, run_id="r3")
        # 3 successes push the old failures out of the window
        for i in range(4, 7):
            mgr.record_result("t", success=True, run_id=f"r{i}")
        # Now add 2 more failures — only 2 in window, below threshold
        mgr.record_result("t", success=False, run_id="r7")
        change = mgr.record_result("t", success=False, run_id="r8")
        # Should NOT demote (only 2 failures in last 5)
        assert mgr.get_tier("t").level >= 1


# ---------------------------------------------------------------------------
# Tier history and audit trail
# ---------------------------------------------------------------------------


class TestTierHistory:
    def test_changes_are_recorded_on_tier_object(self):
        mgr = TierManager()
        mgr.record_result("t", success=True, run_id="r1")
        tier = mgr.get_tier("t")
        assert len(tier.tier_history) == 1
        assert tier.tier_history[0].new_level == 1

    def test_get_recent_changes_returns_all_tools(self):
        mgr = TierManager()
        mgr.record_result("a", success=True, run_id="r1")
        mgr.record_result("b", success=True, run_id="r2")
        changes = mgr.get_recent_changes()
        assert len(changes) == 2
        names = {ch.tool_name for ch in changes}
        assert names == {"a", "b"}

    def test_get_recent_changes_respects_limit(self):
        mgr = TierManager()
        for i in range(10):
            mgr.record_result(f"t{i}", success=True, run_id=f"r{i}")
        assert len(mgr.get_recent_changes(limit=3)) == 3


# ---------------------------------------------------------------------------
# Initialize and get
# ---------------------------------------------------------------------------


class TestInitializeAndGet:
    def test_initialize_sets_level(self):
        mgr = TierManager()
        mgr.initialize_tool("synth_tool", level=0)
        tier = mgr.get_tier("synth_tool")
        assert tier is not None
        assert tier.level == 0
        assert tier.total_calls == 0

    def test_initialize_clamps_level(self):
        mgr = TierManager()
        mgr.initialize_tool("t", level=99)
        assert mgr.get_tier("t").level == 3
        mgr.initialize_tool("u", level=-5)
        assert mgr.get_tier("u").level == 0

    def test_get_tier_returns_none_for_unknown(self):
        mgr = TierManager()
        assert mgr.get_tier("nope") is None

    def test_get_all_tiers(self):
        mgr = TierManager()
        mgr.record_result("a", success=True, run_id="r1")
        mgr.record_result("b", success=True, run_id="r2")
        all_tiers = mgr.get_all_tiers()
        assert set(all_tiers.keys()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_tool_tier_to_dict_is_json_safe(self):
        mgr = TierManager()
        mgr.record_result("t", success=True, run_id="r1")
        mgr.record_result("t", success=True, run_id="r2")
        mgr.record_result("t", success=True, run_id="r3")
        tier = mgr.get_tier("t")
        payload = tier.to_dict()
        dumped = json.dumps(payload)
        restored = json.loads(dumped)
        assert restored["level"] == 2
        assert restored["total_calls"] == 3
        assert restored["distinct_runs"] == 3
        assert len(restored["tier_history"]) >= 1

    def test_tier_summary_is_json_safe(self):
        mgr = TierManager()
        mgr.record_result("a", success=True, run_id="r1")
        mgr.record_result("b", success=False, run_id="r2")
        summary = mgr.get_tier_summary()
        dumped = json.dumps(summary)
        restored = json.loads(dumped)
        assert restored["total_tools_tracked"] == 2
        assert restored["by_level"]["1"] == 1 or restored["by_level"][1] == 1

    def test_tier_change_to_dict(self):
        mgr = TierManager()
        change = mgr.record_result("t", success=True, run_id="r1")
        payload = change.to_dict()
        assert payload["tool_name"] == "t"
        assert payload["old_level"] == 0
        assert payload["new_level"] == 1
        assert "timestamp" in payload


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            TierManager(window_size=0)

    def test_invalid_downgrade_threshold(self):
        with pytest.raises(ValueError, match="downgrade_threshold"):
            TierManager(downgrade_threshold=0)

    def test_custom_thresholds(self):
        mgr = TierManager(
            window_size=5,
            downgrade_threshold=2,
            upgrade_2_min_runs=2,
            upgrade_2_min_success_rate=0.8,
        )
        mgr.record_result("t", success=True, run_id="r1")
        change = mgr.record_result("t", success=True, run_id="r2")
        assert change is not None
        assert change.new_level == 2
