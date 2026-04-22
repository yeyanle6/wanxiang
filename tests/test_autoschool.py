"""Tests for Autoschool — the autonomous growth loop.

Mocks RunManager to avoid real LLM calls. Focus on dispatch logic,
budget gating, concurrency cap, and graduation-on-promotion.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from wanxiang.core.autoschool import Autoschool
from wanxiang.core.growth_budget import GrowthBudget
from wanxiang.core.storage import RunRecord, Storage


@pytest.fixture
def storage(tmp_path):
    s = Storage(tmp_path / "autoschool.db")
    yield s
    s.close()


@pytest.fixture
def budget(storage):
    return GrowthBudget(storage)


def _make_run_manager() -> MagicMock:
    rm = MagicMock()
    rm.start_run = AsyncMock(side_effect=lambda *a, **kw: f"run-{id(a)}")
    rm.wait_for_run = AsyncMock()
    return rm


def _autoschool(storage, run_manager, budget, **overrides) -> Autoschool:
    kwargs = dict(
        storage=storage,
        run_manager=run_manager,
        budget=budget,
        tick_interval_s=60.0,
        max_concurrent=1,
        graduation_check_every=5,
    )
    kwargs.update(overrides)
    return Autoschool(**kwargs)


# ---- Dispatch basics -----------------------------------------------------


class TestDispatch:
    @pytest.mark.asyncio
    async def test_dispatches_current_level_task(self, storage, budget):
        storage.enqueue_task(level=0, task="hello", source="seed")
        rm = _make_run_manager()
        rm.start_run.return_value = "r1"
        school = _autoschool(storage, rm, budget)

        await school.tick()
        # Wait for tracker to complete (wait_for_run is mocked as no-op).
        await asyncio.gather(*school._trackers.values(), return_exceptions=True)

        rm.start_run.assert_awaited_once()
        call = rm.start_run.await_args
        assert call.args[0] == "hello"
        assert call.kwargs == {
            "probe": True,
            "level": 0,
            "source": "autoschool",
            "expected_keywords": None,
        }

    @pytest.mark.asyncio
    async def test_keywords_threaded_through(self, storage, budget):
        storage.enqueue_task(
            level=0, task="t", source="seed",
            expected_outcome_keywords=["a", "b"],
        )
        rm = _make_run_manager()
        rm.start_run.return_value = "r1"
        school = _autoschool(storage, rm, budget)

        await school.tick()
        await asyncio.gather(*school._trackers.values(), return_exceptions=True)

        assert rm.start_run.await_args.kwargs["expected_keywords"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_falls_back_to_any_level(self, storage, budget):
        # current_level=0 but only L1 tasks exist — should still dispatch.
        storage.enqueue_task(level=1, task="l1 only", source="seed")
        rm = _make_run_manager()
        rm.start_run.return_value = "r1"
        school = _autoschool(storage, rm, budget)

        await school.tick()
        await asyncio.gather(*school._trackers.values(), return_exceptions=True)

        rm.start_run.assert_awaited_once()
        assert rm.start_run.await_args.kwargs["level"] == 1

    @pytest.mark.asyncio
    async def test_empty_queue_noop(self, storage, budget):
        rm = _make_run_manager()
        school = _autoschool(storage, rm, budget)
        await school.tick()
        rm.start_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_task_marked_done_after_run(self, storage, budget):
        tid = storage.enqueue_task(level=0, task="hello", source="seed")
        rm = _make_run_manager()
        rm.start_run.return_value = "r1"
        school = _autoschool(storage, rm, budget)

        await school.tick()
        await asyncio.gather(*school._trackers.values(), return_exceptions=True)

        assert storage.pending_tasks() == []  # nothing pending; done row exists


# ---- Budget gate ---------------------------------------------------------


class TestBudgetGate:
    @pytest.mark.asyncio
    async def test_exhausted_budget_skips_dispatch(self, storage, budget):
        # Stamp last_reset=today so the tick's refresh_if_new_day is a no-op.
        budget.refresh_if_new_day()
        storage.enqueue_task(level=0, task="t", source="seed")
        budget.record_usage(200_000, 0, "api")  # way over 100k

        rm = _make_run_manager()
        school = _autoschool(storage, rm, budget)
        await school.tick()
        rm.start_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cannot_afford_skips_dispatch(self, storage, budget):
        budget.refresh_if_new_day()
        storage.enqueue_task(level=0, task="t", source="seed")
        # Use 99k of 100k; default estimate is 5k → cannot afford.
        budget.record_usage(99_000, 0, "api")

        rm = _make_run_manager()
        school = _autoschool(storage, rm, budget)
        await school.tick()
        rm.start_run.assert_not_awaited()


# ---- Concurrency cap -----------------------------------------------------


class TestConcurrencyCap:
    @pytest.mark.asyncio
    async def test_cap_prevents_dispatch_when_full(self, storage, budget):
        storage.enqueue_task(level=0, task="a", source="seed")
        storage.enqueue_task(level=0, task="b", source="seed")

        # Wait_for_run blocks until we release it.
        release = asyncio.Event()

        async def slow_wait(run_id):
            await release.wait()

        rm = _make_run_manager()
        rm.start_run.side_effect = ["r1", "r2"]
        rm.wait_for_run.side_effect = slow_wait

        school = _autoschool(storage, rm, budget, max_concurrent=1)

        await school.tick()  # dispatches 'a'
        await school.tick()  # cap full → no dispatch
        assert rm.start_run.await_count == 1

        release.set()
        # Let tracker finish.
        for t in list(school._trackers.values()):
            try:
                await asyncio.wait_for(t, timeout=2.0)
            except asyncio.TimeoutError:
                t.cancel()

        # Now a new tick should dispatch 'b'.
        await school.tick()
        assert rm.start_run.await_count == 2


# ---- Graduation trigger --------------------------------------------------


class TestGraduation:
    @pytest.mark.asyncio
    async def test_promotes_on_eligible_graduation(self, storage, budget):
        # Seed 15 passing L0 runs + 5 failing to hit 15/20 threshold.
        for i in range(15):
            storage.upsert_run(
                RunRecord(
                    run_id=f"pass{i:02d}",
                    task="t",
                    started_at="2026-04-22T00:00:00+00:00",
                    level=0,
                    source="autoschool",
                    outcome="success",
                )
            )
            storage.update_grade(f"pass{i:02d}", passed=True)
        for i in range(5):
            storage.upsert_run(
                RunRecord(
                    run_id=f"fail{i:02d}",
                    task="t",
                    started_at="2026-04-22T00:00:00+00:00",
                    level=0,
                    source="autoschool",
                    outcome="timeout",
                )
            )
            storage.update_grade(f"fail{i:02d}", passed=False)

        # One queued task so dispatch runs; graduation check every 1 tick.
        storage.enqueue_task(level=0, task="t", source="seed")
        rm = _make_run_manager()
        school = _autoschool(storage, rm, budget, graduation_check_every=1)

        assert storage.read_growth_state()["current_level"] == 0
        await school.tick()
        assert storage.read_growth_state()["current_level"] == 1

    @pytest.mark.asyncio
    async def test_no_promotion_when_ineligible(self, storage, budget):
        # Only 5 graded runs; below 20-run window.
        for i in range(5):
            storage.upsert_run(
                RunRecord(
                    run_id=f"r{i}", task="t",
                    started_at="2026-04-22T00:00:00+00:00",
                    level=0, source="autoschool", outcome="success",
                )
            )
            storage.update_grade(f"r{i}", passed=True)

        storage.enqueue_task(level=0, task="t", source="seed")
        rm = _make_run_manager()
        school = _autoschool(storage, rm, budget, graduation_check_every=1)

        await school.tick()
        assert storage.read_growth_state()["current_level"] == 0


# ---- Lifecycle -----------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_stop(self, storage, budget):
        rm = _make_run_manager()
        school = _autoschool(storage, rm, budget, tick_interval_s=1.0)

        await school.start()
        assert school.running is True

        await asyncio.sleep(0.05)  # give loop time to schedule one iteration

        await school.stop(timeout=2.0)
        assert school.running is False

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self, storage, budget):
        rm = _make_run_manager()
        school = _autoschool(storage, rm, budget, tick_interval_s=1.0)
        await school.start()
        await school.start()  # should be idempotent
        await school.stop(timeout=2.0)

    @pytest.mark.asyncio
    async def test_stop_cancels_trackers(self, storage, budget):
        storage.enqueue_task(level=0, task="t", source="seed")

        forever = asyncio.Event()

        async def block_forever(run_id):
            await forever.wait()

        rm = _make_run_manager()
        rm.start_run.return_value = "r1"
        rm.wait_for_run.side_effect = block_forever

        school = _autoschool(storage, rm, budget, tick_interval_s=10.0)
        await school.tick()
        assert len(school._trackers) == 1

        await school.stop(timeout=1.0)
        assert len(school._trackers) == 0


# ---- Error isolation -----------------------------------------------------


class TestErrorIsolation:
    @pytest.mark.asyncio
    async def test_start_run_failure_marks_complete(self, storage, budget):
        tid = storage.enqueue_task(level=0, task="t", source="seed")
        rm = _make_run_manager()
        rm.start_run.side_effect = RuntimeError("boom")

        school = _autoschool(storage, rm, budget)
        await school.tick()  # must not raise

        # Task was marked done with run_id=None so it doesn't stay stuck.
        assert storage.pending_tasks() == []
