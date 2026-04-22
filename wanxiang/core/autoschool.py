"""autoschool — the autonomous growth loop.

Phase 5 of Week 2. Runs as a single background asyncio.Task that ticks
every N seconds, claims curriculum tasks, dispatches them through
RunManager (probe=True by default), and promotes the growth_state level
when graduation says so.

Design:
  - Single instance per server. Concurrency cap is Autoschool-internal.
  - Tick cadence default 60s; shorter hurts TPM budget, longer slows
    growth pace. Autoschool does NOT retry — if a tick fails it logs
    and waits for the next one.
  - Fire-and-forget dispatch: start_run returns quickly, a background
    tracker task awaits run completion and marks the curriculum row
    'done'. On autoschool.stop(), trackers are cancelled (run itself
    keeps going if already past start_run).
  - Budget exhaustion is a soft gate — the tick returns early, no
    dispatch happens. No retry loop, no escalation.
  - Graduation check is cheap (SQL query + threshold math); runs every
    graduation_check_every ticks (default 5) to avoid hammering storage
    without real-time precision requirements.

Environment:
  WANXIANG_AUTOSCHOOL_ENABLED=1         turn on (default off)
  WANXIANG_AUTOSCHOOL_TICK_S=60         tick cadence seconds
  WANXIANG_AUTOSCHOOL_MAX_CONCURRENT=1  simultaneous dispatched runs
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from . import graduation

if TYPE_CHECKING:
    from .growth_budget import GrowthBudget
    from .storage import Storage
    from ..server.runner import RunManager


DEFAULT_TICK_INTERVAL_S = 60.0
DEFAULT_MAX_CONCURRENT = 1
DEFAULT_GRADUATION_CHECK_EVERY = 5


class Autoschool:
    def __init__(
        self,
        *,
        storage: Storage,
        run_manager: RunManager,
        budget: GrowthBudget,
        tick_interval_s: float = DEFAULT_TICK_INTERVAL_S,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        graduation_check_every: int = DEFAULT_GRADUATION_CHECK_EVERY,
    ) -> None:
        self.storage = storage
        self.run_manager = run_manager
        self.budget = budget
        self.tick_interval_s = max(1.0, float(tick_interval_s))
        self.max_concurrent = max(1, int(max_concurrent))
        self.graduation_check_every = max(1, int(graduation_check_every))
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._trackers: dict[int, asyncio.Task] = {}
        self._tick_count = 0
        self.logger = logging.getLogger("wanxiang.autoschool")

    # ---- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run_loop(), name="autoschool-loop")
        self.logger.info(
            "Autoschool started: tick=%.1fs, max_concurrent=%d",
            self.tick_interval_s,
            self.max_concurrent,
        )

    async def stop(self, *, timeout: float = 10.0) -> None:
        """Stop the loop and cancel any in-flight tracker tasks.

        Safe to call from any state — loop not started, loop running,
        loop already finished. Trackers created by bare tick() calls
        (tests, REPL) are cancelled here too; not just those from the
        loop.
        """
        self._stop_event.set()
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
        for tracker in list(self._trackers.values()):
            tracker.cancel()
            try:
                await tracker
            except (asyncio.CancelledError, Exception):
                pass
        self._trackers.clear()
        self._task = None
        self.logger.info("Autoschool stopped")

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ---- Main loop ----------------------------------------------------------

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("autoschool tick failed; continuing")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.tick_interval_s
                )
                return  # stop_event fired
            except asyncio.TimeoutError:
                continue  # normal inter-tick wait

    async def tick(self) -> None:
        """Publicly callable single tick — used in tests."""
        await self._tick()

    async def _tick(self) -> None:
        self._tick_count += 1
        self.budget.refresh_if_new_day()
        self._sweep_finished_trackers()

        if self.budget.is_exhausted():
            self.logger.info(
                "Budget exhausted (tokens_used_today=%d); skipping tick",
                self.budget.snapshot().get("tokens_used_today", 0),
            )
            return

        if len(self._trackers) >= self.max_concurrent:
            return

        if not self.budget.can_afford():
            self.logger.info("Budget cannot afford default call cost; skipping tick")
            return

        state = self.storage.read_growth_state()
        current_level = int(state.get("current_level", 0))
        task = self._claim_next_for_level(current_level)
        if task is None:
            return

        await self._dispatch(task)

        if self._tick_count % self.graduation_check_every == 0:
            self._maybe_promote(current_level)

    # ---- Task dispatch ------------------------------------------------------

    def _claim_next_for_level(self, current_level: int) -> dict[str, Any] | None:
        """Prefer current_level; fall back to any pending task if none left."""
        claimed = self.storage.claim_next_task(level=current_level)
        if claimed is not None:
            return claimed
        return self.storage.claim_next_task()

    async def _dispatch(self, task: dict[str, Any]) -> None:
        task_id = int(task["id"])
        task_text = str(task["task"])
        level = int(task.get("level", 0))
        keywords = task.get("expected_outcome_keywords") or None
        try:
            run_id = await self.run_manager.start_run(
                task_text,
                probe=True,
                level=level,
                source="autoschool",
                expected_keywords=keywords,
            )
        except Exception:
            self.logger.exception("Dispatch failed for task %d; marking complete", task_id)
            self.storage.complete_task(task_id, run_id=None)
            return

        tracker = asyncio.create_task(
            self._track(task_id, run_id), name=f"autoschool-track-{task_id}"
        )
        self._trackers[task_id] = tracker
        self.logger.info(
            "Dispatched L%d task %d as run %s", level, task_id, run_id[:8]
        )

    async def _track(self, task_id: int, run_id: str) -> None:
        try:
            await self.run_manager.wait_for_run(run_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("Track failed waiting for run %s", run_id)
        self.storage.complete_task(task_id, run_id=run_id)

    def _sweep_finished_trackers(self) -> None:
        done = [tid for tid, t in self._trackers.items() if t.done()]
        for tid in done:
            del self._trackers[tid]

    # ---- Graduation ---------------------------------------------------------

    def _maybe_promote(self, current_level: int) -> None:
        result = graduation.evaluate(self.storage, current_level)
        if not result.eligible or result.next_level is None:
            return
        self.storage.update_growth_state(current_level=result.next_level)
        self.logger.info(
            "Graduated L%d → L%d: %s",
            current_level,
            result.next_level,
            result.reason,
        )
