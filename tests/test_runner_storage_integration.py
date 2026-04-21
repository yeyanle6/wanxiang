"""Tests for RunManager ↔ Storage dual-write integration.

Verifies that:
  - Storage writes happen only when storage is configured.
  - Outcome tagger runs on every persist and the tag lands in SQLite.
  - Jsonl write path is still correct (backward compat).
  - level / source from start_run() propagate into the SQLite record.
  - SQLite write failures do not break jsonl persistence (graceful degradation).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from wanxiang.core.outcome_tagger import OUTCOME_INFRA_ERROR, OUTCOME_SUCCESS
from wanxiang.core.storage import RunRecord, Storage
from wanxiang.server.runner import RunManager, _RunState


@pytest.fixture
def tmp_storage(tmp_path):
    store = Storage(tmp_path / "test.db")
    yield store
    store.close()


def _fake_state(events, *, final_status="success", level=None, source="user"):
    state = _RunState(
        user_task="test task",
        started_at="2026-04-21T00:00:00+00:00",
        completed_at="2026-04-21T00:00:01+00:00",
        final_status=final_status,
        events=events,
        level=level,
        source=source,
    )
    return state


def _base_record(state, run_id="r1"):
    return {
        "run_id": run_id,
        "task": state.user_task,
        "started_at": state.started_at,
        "completed_at": state.completed_at,
        "final_status": state.final_status,
        "events": state.events,
    }


class TestDualWriteEnabled:
    def test_run_lands_in_sqlite_when_storage_provided(self, tmp_storage, tmp_path):
        rm = RunManager(storage=tmp_storage)
        rm._history_path = tmp_path / "runs.jsonl"

        events = [
            {"type": "run_started", "timestamp": "t", "data": {}},
            {"type": "run_completed", "timestamp": "t", "data": {"status": "success"}},
        ]
        state = _fake_state(events)
        rm._runs["r1"] = state
        record = _base_record(state)

        rm._write_storage_record(record, None, "user")

        got = tmp_storage.get_run("r1")
        assert got is not None
        assert got.outcome == OUTCOME_SUCCESS
        assert got.event_count == 2
        assert got.source == "user"

    def test_outcome_tagged_on_error(self, tmp_storage, tmp_path):
        rm = RunManager(storage=tmp_storage)
        rm._history_path = tmp_path / "runs.jsonl"

        events = [
            {
                "type": "tool_completed",
                "timestamp": "t",
                "data": {"success": False, "content": "ConnectionError: unreachable"},
            }
        ]
        state = _fake_state(events, final_status="error")
        rm._runs["r1"] = state

        rm._write_storage_record(_base_record(state), None, "user")

        got = tmp_storage.get_run("r1")
        assert got.outcome == OUTCOME_INFRA_ERROR

    def test_level_and_source_propagate(self, tmp_storage, tmp_path):
        rm = RunManager(storage=tmp_storage)
        rm._history_path = tmp_path / "runs.jsonl"

        state = _fake_state([], level=1, source="autoschool")
        rm._runs["r1"] = state
        rm._write_storage_record(_base_record(state), 1, "autoschool")

        got = tmp_storage.get_run("r1")
        assert got.level == 1
        assert got.source == "autoschool"


class TestDualWriteDisabled:
    def test_no_storage_means_only_jsonl(self, tmp_path):
        rm = RunManager()  # storage=None
        rm._history_path = tmp_path / "runs.jsonl"

        state = _fake_state([])
        record = _base_record(state)
        rm._append_history_record(record)

        # jsonl exists
        lines = rm._history_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["run_id"] == "r1"


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_storage_exception_does_not_block_jsonl(self, tmp_path):
        import asyncio

        # Fake Storage that raises on upsert_run.
        bad_storage = MagicMock()
        bad_storage.upsert_run.side_effect = RuntimeError("disk full")
        bad_storage.update_outcome = MagicMock()

        rm = RunManager(storage=bad_storage)
        rm._history_path = tmp_path / "runs.jsonl"

        events = [{"type": "run_completed", "timestamp": "t", "data": {"status": "success"}}]
        state = _fake_state(events)
        rm._runs["r1"] = state

        # _persist_run must complete without raising even if SQLite blows up.
        await rm._persist_run(
            "r1",
            {
                "run_id": "r1",
                "task": state.user_task,
                "started_at": state.started_at,
                "completed_at": state.completed_at,
                "final_status": state.final_status,
            },
        )

        # jsonl write still happened
        assert rm._history_path.exists()
        lines = rm._history_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        # state marked persisted so we don't double-write on retry
        assert state.persisted is True


class TestStartRunWithLevelSource:
    @pytest.mark.asyncio
    async def test_start_run_accepts_level_and_source(self, tmp_path):
        from unittest.mock import AsyncMock

        rm = RunManager()
        rm._history_path = tmp_path / "runs.jsonl"

        # Short-circuit _execute_run so we don't actually call an LLM.
        rm._execute_run = AsyncMock(return_value=None)

        run_id = await rm.start_run("test", level=1, source="autoschool")
        state = rm._runs[run_id]
        assert state.level == 1
        assert state.source == "autoschool"

    @pytest.mark.asyncio
    async def test_start_run_defaults(self, tmp_path):
        from unittest.mock import AsyncMock

        rm = RunManager()
        rm._history_path = tmp_path / "runs.jsonl"
        rm._execute_run = AsyncMock(return_value=None)

        run_id = await rm.start_run("test")
        state = rm._runs[run_id]
        assert state.level is None
        assert state.source == "user"


class TestBootstrapIdempotence:
    def test_importing_twice_does_not_duplicate(self, tmp_path):
        jsonl = tmp_path / "runs.jsonl"
        record = {
            "run_id": "imported_r1",
            "task": "t",
            "started_at": "2026-04-20T00:00:00+00:00",
            "final_status": "success",
            "events": [],
        }
        jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

        store = Storage(tmp_path / "test.db")
        try:
            store.import_jsonl(jsonl)
            store.import_jsonl(jsonl)
            assert len(store.list_runs(limit=100)) == 1
        finally:
            store.close()
