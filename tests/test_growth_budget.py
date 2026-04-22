"""Tests for growth_budget — daily token budget gate."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from wanxiang.core.growth_budget import (
    DEFAULT_DAILY_TOKENS,
    DEFAULT_ESTIMATED_CALL_COST,
    GrowthBudget,
)
from wanxiang.core.storage import Storage


@pytest.fixture
def storage(tmp_path):
    s = Storage(tmp_path / "budget.db")
    yield s
    s.close()


# ---- Defaults + env override --------------------------------------------


class TestConstruction:
    def test_default_estimate(self, storage):
        b = GrowthBudget(storage)
        assert b.default_estimate == DEFAULT_ESTIMATED_CALL_COST

    def test_explicit_estimate_overrides_default(self, storage):
        b = GrowthBudget(storage, default_estimate=1234)
        assert b.default_estimate == 1234

    def test_env_daily_budget_writes_to_storage(self, storage, monkeypatch):
        monkeypatch.setenv("WANXIANG_BUDGET_DAILY_TOKENS", "42000")
        GrowthBudget(storage)
        assert storage.read_growth_state()["budget_daily_tokens"] == 42000

    def test_env_daily_budget_zero_ignored(self, storage, monkeypatch):
        monkeypatch.setenv("WANXIANG_BUDGET_DAILY_TOKENS", "0")
        GrowthBudget(storage)
        # Zero was ignored; defaults stay.
        assert storage.read_growth_state()["budget_daily_tokens"] == DEFAULT_DAILY_TOKENS

    def test_env_non_integer_ignored(self, storage, monkeypatch):
        monkeypatch.setenv("WANXIANG_BUDGET_DAILY_TOKENS", "not-a-number")
        GrowthBudget(storage)
        assert storage.read_growth_state()["budget_daily_tokens"] == DEFAULT_DAILY_TOKENS


# ---- Gate checks --------------------------------------------------------


class TestCanAfford:
    def test_fresh_budget_can_afford_default(self, storage):
        b = GrowthBudget(storage)
        assert b.can_afford() is True

    def test_explicit_estimate(self, storage):
        b = GrowthBudget(storage)
        # 100k budget, estimate under budget → OK.
        assert b.can_afford(50_000) is True
        assert b.can_afford(100_001) is False  # over budget

    def test_after_usage_less_room(self, storage):
        b = GrowthBudget(storage)
        b.record_usage(40_000, 40_000, "api")  # 80k used, 20k left
        assert b.can_afford(15_000) is True
        assert b.can_afford(25_000) is False

    def test_exactly_at_budget_denied(self, storage):
        # can_afford is strict ≤, but at budget means 0 remaining.
        # Default estimate would exceed what's left.
        b = GrowthBudget(storage)
        b.record_usage(100_000, 0, "api")
        assert b.can_afford() is False


class TestIsExhausted:
    def test_fresh_not_exhausted(self, storage):
        assert GrowthBudget(storage).is_exhausted() is False

    def test_used_reaches_budget_exhausted(self, storage):
        b = GrowthBudget(storage)
        b.record_usage(100_000, 0, "api")
        assert b.is_exhausted() is True

    def test_overshoot_exhausted(self, storage):
        b = GrowthBudget(storage)
        b.record_usage(120_000, 0, "api")
        assert b.is_exhausted() is True

    def test_remaining_goes_to_zero_not_negative(self, storage):
        b = GrowthBudget(storage)
        b.record_usage(120_000, 0, "api")
        assert b.remaining() == 0


# ---- record_usage -------------------------------------------------------


class TestRecordUsage:
    def test_records_sum(self, storage):
        b = GrowthBudget(storage)
        b.record_usage(100, 200, "api")
        assert storage.read_growth_state()["tokens_used_today"] == 300

    def test_accumulates(self, storage):
        b = GrowthBudget(storage)
        b.record_usage(50, 50, "api")
        b.record_usage(30, 20, "cli")
        assert storage.read_growth_state()["tokens_used_today"] == 150

    def test_negative_input_clamped(self, storage):
        b = GrowthBudget(storage)
        b.record_usage(-100, 50, "api")
        # Only the positive output counted.
        assert storage.read_growth_state()["tokens_used_today"] == 50

    def test_zero_usage_skipped(self, storage):
        b = GrowthBudget(storage)
        b.record_usage(0, 0, "api")
        assert storage.read_growth_state()["tokens_used_today"] == 0

    def test_records_over_budget(self, storage):
        # Recording MUST persist even when over-budget; the gate is in
        # can_afford, not here.
        b = GrowthBudget(storage)
        b.record_usage(200_000, 0, "api")
        assert storage.read_growth_state()["tokens_used_today"] == 200_000
        assert b.is_exhausted() is True


# ---- Daily rollover -----------------------------------------------------


class TestRefreshIfNewDay:
    def test_same_day_noop(self, storage):
        b = GrowthBudget(storage)
        b.record_usage(5000, 0, "api")
        b.refresh_if_new_day()  # first call sets last_reset to today
        used_before = storage.read_growth_state()["tokens_used_today"]

        # Second call in the same day must not reset.
        result = b.refresh_if_new_day()
        assert result is False
        assert storage.read_growth_state()["tokens_used_today"] == used_before

    def test_new_day_resets(self, storage):
        b = GrowthBudget(storage)
        b.record_usage(5000, 0, "api")
        # First tick "yesterday".
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        b.refresh_if_new_day(now=yesterday)
        b.record_usage(3000, 0, "api")

        # Move clock forward a day — should reset.
        today = yesterday + timedelta(days=1)
        reset = b.refresh_if_new_day(now=today)
        assert reset is True
        assert storage.read_growth_state()["tokens_used_today"] == 0

    def test_explicit_now_used(self, storage):
        b = GrowthBudget(storage)
        frozen = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)
        b.refresh_if_new_day(now=frozen)
        assert storage.read_growth_state()["last_reset"] == "2026-04-22"


# ---- Snapshot -----------------------------------------------------------


class TestSnapshot:
    def test_snapshot_shape(self, storage):
        b = GrowthBudget(storage)
        b.record_usage(1000, 2000, "api")
        s = b.snapshot()
        assert s["budget_daily_tokens"] == DEFAULT_DAILY_TOKENS
        assert s["tokens_used_today"] == 3000
        assert s["remaining"] == DEFAULT_DAILY_TOKENS - 3000
        assert s["exhausted"] is False
        assert "default_estimate" in s


# ---- Integration with LLMClient UsageRecorder shape --------------------


class TestUsageRecorderCompatibility:
    def test_record_usage_matches_usage_recorder_signature(self, storage):
        b = GrowthBudget(storage)
        # LLMClient calls: recorder(input_tokens, output_tokens, mode)
        recorder = b.record_usage
        recorder(100, 50, "api")
        recorder(200, 100, "cli")
        assert storage.read_growth_state()["tokens_used_today"] == 450
