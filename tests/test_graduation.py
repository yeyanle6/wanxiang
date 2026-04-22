"""Tests for graduation — data-driven level promotion judge."""
from __future__ import annotations

from pathlib import Path

import pytest

from wanxiang.core.graduation import (
    DEFAULT_WINDOW,
    MIN_SKILL_LIBRARY_FOR_L2,
    evaluate,
)
from wanxiang.core.storage import RunRecord, Storage


def _graded_run(store, run_id, level, passed, *, source="autoschool", outcome="success"):
    store.upsert_run(
        RunRecord(
            run_id=run_id,
            task=f"task_{run_id}",
            started_at=f"2026-04-22T00:00:{int(run_id[-2:]):02d}+00:00",
            completed_at=f"2026-04-22T00:00:{int(run_id[-2:]):02d}+00:00",
            level=level,
            source=source,
            outcome=outcome,
        )
    )
    store.update_grade(run_id, passed=passed, reason="test")


@pytest.fixture
def storage(tmp_path):
    s = Storage(tmp_path / "test.db")
    yield s
    s.close()


class TestInsufficientSample:
    def test_below_window_not_eligible(self, storage):
        # Only 5 graded runs — need 20.
        for i in range(5):
            _graded_run(storage, f"r{i:02d}", level=0, passed=True)
        result = evaluate(storage, current_level=0)
        assert result.eligible is False
        assert "5/20 graded runs" in result.reason

    def test_empty_storage(self, storage):
        result = evaluate(storage, current_level=0)
        assert result.eligible is False


class TestL0ToL1:
    def test_meets_threshold(self, storage):
        for i in range(15):
            _graded_run(storage, f"p{i:02d}", level=0, passed=True)
        for i in range(5):
            _graded_run(storage, f"f{i:02d}", level=0, passed=False)
        result = evaluate(storage, current_level=0)
        assert result.eligible is True
        assert result.next_level == 1
        assert result.observed["passed"] == 15

    def test_below_threshold(self, storage):
        for i in range(14):
            _graded_run(storage, f"p{i:02d}", level=0, passed=True)
        for i in range(6):
            _graded_run(storage, f"f{i:02d}", level=0, passed=False)
        result = evaluate(storage, current_level=0)
        assert result.eligible is False
        assert result.observed["passed"] == 14


class TestL1ToL2SkillGate:
    def test_pass_rate_ok_but_no_skills_blocks(self, storage):
        # 15/20 pass at L1 but no synthesized skills yet.
        for i in range(15):
            _graded_run(storage, f"p{i:02d}", level=1, passed=True, outcome="partial")
        for i in range(5):
            _graded_run(storage, f"f{i:02d}", level=1, passed=False, outcome="partial")
        result = evaluate(storage, current_level=1)
        assert result.eligible is False
        assert "skill library" in result.reason
        assert result.observed["synthesized_tools"] == 0

    def test_pass_rate_ok_with_enough_skills(self, storage):
        for i in range(15):
            _graded_run(storage, f"p{i:02d}", level=1, passed=True, outcome="success")
        for i in range(5):
            _graded_run(storage, f"f{i:02d}", level=1, passed=False, outcome="partial")
        # Proxy skill count: successful autoschool runs. 15 successes ≥ 5 → eligible.
        result = evaluate(storage, current_level=1)
        assert result.eligible is True
        assert result.observed["synthesized_tools"] >= MIN_SKILL_LIBRARY_FOR_L2


class TestL2ToL3:
    def test_lower_threshold_for_l2(self, storage):
        # L2 uses 12/20 instead of 15/20.
        for i in range(12):
            _graded_run(storage, f"p{i:02d}", level=2, passed=True)
        for i in range(8):
            _graded_run(storage, f"f{i:02d}", level=2, passed=False)
        result = evaluate(storage, current_level=2)
        assert result.eligible is True

    def test_below_l2_threshold(self, storage):
        for i in range(11):
            _graded_run(storage, f"p{i:02d}", level=2, passed=True)
        for i in range(9):
            _graded_run(storage, f"f{i:02d}", level=2, passed=False)
        result = evaluate(storage, current_level=2)
        assert result.eligible is False


class TestManualLevels:
    def test_l3_requires_manual_approval(self, storage):
        result = evaluate(storage, current_level=3)
        assert result.eligible is False
        assert "manual" in result.criterion or "manual" in result.reason

    def test_l4_same(self, storage):
        result = evaluate(storage, current_level=4)
        assert result.eligible is False


class TestLevelFiltering:
    def test_wrong_level_runs_ignored(self, storage):
        # 20 passing L1 runs shouldn't help L0 graduation.
        for i in range(20):
            _graded_run(storage, f"p{i:02d}", level=1, passed=True)
        result = evaluate(storage, current_level=0)
        assert result.eligible is False
        assert result.observed["graded_sample_size"] == 0


class TestResultSerialization:
    def test_to_dict_shape(self, storage):
        result = evaluate(storage, current_level=0)
        d = result.to_dict()
        assert set(d.keys()) == {
            "current_level", "eligible", "next_level",
            "criterion", "observed", "reason",
        }
