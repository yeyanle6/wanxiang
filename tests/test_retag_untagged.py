"""Test the _retag_untagged_runs lifespan helper — fills outcome on runs
that landed without a label (e.g., bootstrapped from jsonl)."""
from __future__ import annotations

from pathlib import Path

import pytest

from wanxiang.core.storage import RunRecord, Storage
from wanxiang.server.app import _retag_untagged_runs


def _untagged_run(run_id: str, final_status: str, events: list[dict]) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        task="test",
        started_at="2026-04-21T00:00:00+00:00",
        completed_at="2026-04-21T00:00:01+00:00",
        final_status=final_status,
        outcome=None,  # key — untagged
        events=events,
    )


class TestRetagUntagged:
    def test_noop_when_all_tagged(self, tmp_path):
        store = Storage(tmp_path / "test.db")
        try:
            record = _untagged_run("r1", "success", [])
            record.outcome = "success"
            store.upsert_run(record)

            count = _retag_untagged_runs(store)
            assert count == 0
        finally:
            store.close()

    def test_tags_success_run(self, tmp_path):
        store = Storage(tmp_path / "test.db")
        try:
            events = [{"type": "run_completed", "data": {"status": "success"}}]
            store.upsert_run(_untagged_run("r1", "success", events))

            count = _retag_untagged_runs(store)
            assert count == 1
            assert store.get_run("r1").outcome == "success"
        finally:
            store.close()

    def test_tags_needs_revision_as_partial(self, tmp_path):
        # This is exactly the bug that motivated the fix.
        store = Storage(tmp_path / "test.db")
        try:
            events = [{"type": "agent_completed", "data": {"status": "needs_revision"}}]
            store.upsert_run(_untagged_run("r1", "needs_revision", events))

            _retag_untagged_runs(store)
            assert store.get_run("r1").outcome == "partial"
        finally:
            store.close()

    def test_tags_error_runs(self, tmp_path):
        store = Storage(tmp_path / "test.db")
        try:
            events = [{"type": "tool_completed", "data": {"success": False, "content": "ConnectionError"}}]
            store.upsert_run(_untagged_run("r1", "error", events))

            _retag_untagged_runs(store)
            assert store.get_run("r1").outcome == "infra_error"
        finally:
            store.close()

    def test_only_touches_untagged(self, tmp_path):
        store = Storage(tmp_path / "test.db")
        try:
            events = [{"type": "run_completed", "data": {"status": "success"}}]
            # One already tagged (intentionally 'wrong' to catch overwrites).
            r1 = _untagged_run("r1", "success", events)
            r1.outcome = "unknown"
            store.upsert_run(r1)
            # One untagged.
            store.upsert_run(_untagged_run("r2", "success", events))

            count = _retag_untagged_runs(store)
            assert count == 1
            # The pre-tagged one is left alone, even if its label is stale.
            assert store.get_run("r1").outcome == "unknown"
            assert store.get_run("r2").outcome == "success"
        finally:
            store.close()

    def test_handles_empty_db(self, tmp_path):
        store = Storage(tmp_path / "test.db")
        try:
            assert _retag_untagged_runs(store) == 0
        finally:
            store.close()
