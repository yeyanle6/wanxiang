"""growth_budget — daily token budget gate for autoschool.

Thin domain layer on top of storage.growth_state. Stays out of storage
because "budget exhausted → switch to offline mode" is a behavior
concern, not a persistence concern.

Autoschool calls:
  budget.refresh_if_new_day()        # once per tick, no-op if same UTC day
  budget.can_afford(estimated=5000)  # gate before claiming next task
  budget.record_usage(i, o, mode)    # wired as LLMClient.usage_recorder
  budget.is_exhausted()              # bail-out signal

Design choices:
  - Budget lives in storage.growth_state.tokens_used_today. Resetting
    on UTC 00:00 boundary is detected via last_reset != today (UTC).
  - No async / no lock here: storage already serializes writes. Budget
    is a single logical owner per server instance.
  - record_usage always persists even when over-budget — the LLM call
    already happened; refusing to record would just hide the overshoot.
    The next can_afford() call will see the debt and refuse.
  - estimated-tokens-per-call defaults to 5000. Callers can override at
    can_afford time if they have a better estimate.

Environment:
  WANXIANG_BUDGET_DAILY_TOKENS   override daily budget (int)
  WANXIANG_ESTIMATED_CALL_COST   override default can_afford estimate
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .storage import Storage

logger = logging.getLogger("wanxiang.growth_budget")

DEFAULT_DAILY_TOKENS = 100_000
DEFAULT_ESTIMATED_CALL_COST = 5_000


class GrowthBudget:
    def __init__(
        self,
        storage: Storage,
        *,
        default_estimate: int | None = None,
    ) -> None:
        self.storage = storage
        self.default_estimate = int(
            default_estimate
            if default_estimate is not None
            else os.getenv("WANXIANG_ESTIMATED_CALL_COST", DEFAULT_ESTIMATED_CALL_COST)
        )
        env_daily = os.getenv("WANXIANG_BUDGET_DAILY_TOKENS")
        if env_daily:
            try:
                env_value = int(env_daily)
                if env_value > 0:
                    # Sync storage so env override wins over stored default.
                    state = self.storage.read_growth_state()
                    if int(state.get("budget_daily_tokens", 0)) != env_value:
                        self.storage.update_growth_state(budget_daily_tokens=env_value)
            except ValueError:
                logger.warning(
                    "Ignoring non-integer WANXIANG_BUDGET_DAILY_TOKENS: %r", env_daily
                )

    # ---- Daily rollover -----------------------------------------------------

    def refresh_if_new_day(self, *, now: datetime | None = None) -> bool:
        """Zero tokens_used_today if we've crossed a UTC day boundary.

        Returns True if a reset happened. Idempotent within a day.
        """
        today = (now or datetime.now(timezone.utc)).date().isoformat()
        state = self.storage.read_growth_state()
        last_reset = state.get("last_reset") or ""
        if last_reset == today:
            return False
        self.storage.reset_daily_budget(today_utc=today)
        logger.info("Daily budget reset for %s", today)
        return True

    # ---- Gate checks --------------------------------------------------------

    def can_afford(self, estimated: int | None = None) -> bool:
        """True iff tokens_used_today + estimated ≤ budget_daily_tokens."""
        estimate = int(estimated if estimated is not None else self.default_estimate)
        state = self.storage.read_growth_state()
        used = int(state.get("tokens_used_today", 0))
        budget = int(state.get("budget_daily_tokens", DEFAULT_DAILY_TOKENS))
        return used + estimate <= budget

    def is_exhausted(self) -> bool:
        """True iff used tokens already meet or exceed the daily budget."""
        state = self.storage.read_growth_state()
        used = int(state.get("tokens_used_today", 0))
        budget = int(state.get("budget_daily_tokens", DEFAULT_DAILY_TOKENS))
        return used >= budget

    def remaining(self) -> int:
        state = self.storage.read_growth_state()
        return max(
            0,
            int(state.get("budget_daily_tokens", DEFAULT_DAILY_TOKENS))
            - int(state.get("tokens_used_today", 0)),
        )

    # ---- Bookkeeping --------------------------------------------------------

    def record_usage(
        self, input_tokens: int, output_tokens: int, mode: str = ""
    ) -> None:
        """Add input+output tokens to today's counter.

        Compatible with LLMClient.UsageRecorder shape. Negative or zero
        deltas are ignored. Records over-budget too — hiding that would
        defeat the point of the budget gate.
        """
        total = max(0, int(input_tokens)) + max(0, int(output_tokens))
        if total <= 0:
            return
        self.storage.increment_tokens_used(total)
        logger.debug(
            "Budget +%d tokens (mode=%s); remaining=%d", total, mode, self.remaining()
        )

    # ---- Snapshot -----------------------------------------------------------

    def snapshot(self) -> dict:
        state = self.storage.read_growth_state()
        budget = int(state.get("budget_daily_tokens", DEFAULT_DAILY_TOKENS))
        used = int(state.get("tokens_used_today", 0))
        return {
            "budget_daily_tokens": budget,
            "tokens_used_today": used,
            "remaining": max(0, budget - used),
            "last_reset": state.get("last_reset"),
            "exhausted": used >= budget,
            "default_estimate": self.default_estimate,
        }
