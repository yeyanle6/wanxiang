"""Tests for wanxiang.core.storage — SQLite-backed run / event / growth / curriculum store."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wanxiang.core.storage import RunRecord, Storage


def _sample_events() -> list[dict]:
    return [
        {"type": "run_started", "timestamp": "2026-04-21T00:00:00+00:00", "data": {"run_id": "r1"}},
        {
            "type": "tool_completed",
            "timestamp": "2026-04-21T00:00:01+00:00",
            "data": {"tool": "search", "success": True},
        },
        {
            "type": "run_completed",
            "timestamp": "2026-04-21T00:00:02+00:00",
            "data": {"status": "success"},
        },
    ]


def _make_store(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "wanxiang.db")


# ---- Schema + lifecycle --------------------------------------------------


class TestSchema:
    def test_fresh_db_has_growth_state_row(self, tmp_path):
        store = _make_store(tmp_path)
        state = store.read_growth_state()
        assert state["id"] == 1
        assert state["current_level"] == 0
        assert state["budget_daily_tokens"] == 100000

    def test_close_is_safe(self, tmp_path):
        store = _make_store(tmp_path)
        store.close()

    def test_schema_version_recorded(self, tmp_path):
        store = _make_store(tmp_path)
        with store._lock:
            row = store._conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
        assert row is not None
        assert row["value"] == "1"


# ---- Runs ----------------------------------------------------------------


class TestRuns:
    def test_upsert_and_get(self, tmp_path):
        store = _make_store(tmp_path)
        record = RunRecord(
            run_id="r1",
            task="test task",
            started_at="2026-04-21T00:00:00+00:00",
            final_status="success",
            events=_sample_events(),
        )
        store.upsert_run(record)

        got = store.get_run("r1")
        assert got is not None
        assert got.task == "test task"
        assert got.event_count == 3
        assert len(got.events or []) == 3

    def test_upsert_is_idempotent_on_run_id(self, tmp_path):
        store = _make_store(tmp_path)
        r1 = RunRecord(run_id="r1", task="v1", started_at="2026-04-21T00:00:00+00:00")
        r2 = RunRecord(run_id="r1", task="v2", started_at="2026-04-21T00:00:00+00:00")
        store.upsert_run(r1)
        store.upsert_run(r2)
        got = store.get_run("r1")
        assert got.task == "v2"

        # events should be replaced, not appended
        r3 = RunRecord(
            run_id="r1",
            task="v2",
            started_at="2026-04-21T00:00:00+00:00",
            events=_sample_events(),
        )
        store.upsert_run(r3)
        assert len(store.get_run("r1").events) == 3

    def test_get_missing_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_run("nope") is None

    def test_list_runs_sorted_desc(self, tmp_path):
        store = _make_store(tmp_path)
        for i, ts in enumerate(["2026-04-20", "2026-04-21", "2026-04-19"]):
            store.upsert_run(
                RunRecord(run_id=f"r{i}", task=f"t{i}", started_at=f"{ts}T00:00:00+00:00")
            )
        results = store.list_runs(limit=10)
        dates = [r.started_at[:10] for r in results]
        assert dates == ["2026-04-21", "2026-04-20", "2026-04-19"]

    def test_list_runs_filter_by_outcome(self, tmp_path):
        store = _make_store(tmp_path)
        store.upsert_run(RunRecord(run_id="r1", task="t", started_at="2026-04-21T00:00:00+00:00"))
        store.upsert_run(RunRecord(run_id="r2", task="t", started_at="2026-04-21T00:00:01+00:00"))
        store.update_outcome("r1", "success")
        store.update_outcome("r2", "infra_error")

        successes = store.list_runs(outcome="success")
        assert len(successes) == 1 and successes[0].run_id == "r1"

    def test_list_runs_filter_by_source_and_level(self, tmp_path):
        store = _make_store(tmp_path)
        store.upsert_run(RunRecord(run_id="r1", task="t", started_at="2026-04-21T00:00:00+00:00", source="user"))
        store.upsert_run(RunRecord(run_id="r2", task="t", started_at="2026-04-21T00:00:01+00:00", source="autoschool", level=1))
        store.upsert_run(RunRecord(run_id="r3", task="t", started_at="2026-04-21T00:00:02+00:00", source="autoschool", level=2))

        assert len(store.list_runs(source="autoschool")) == 2
        assert len(store.list_runs(level=1)) == 1

    def test_update_outcome_sets_tagged_at(self, tmp_path):
        store = _make_store(tmp_path)
        store.upsert_run(RunRecord(run_id="r1", task="t", started_at="2026-04-21T00:00:00+00:00"))
        store.update_outcome("r1", "success")
        got = store.get_run("r1")
        assert got.outcome == "success"
        assert got.tagged_at is not None


# ---- Events + FTS --------------------------------------------------------


class TestEventsAndSearch:
    def test_events_preserve_order(self, tmp_path):
        store = _make_store(tmp_path)
        events = _sample_events()
        store.upsert_run(
            RunRecord(run_id="r1", task="t", started_at="2026-04-21T00:00:00+00:00", events=events)
        )
        loaded = store.get_run("r1").events
        assert [e["type"] for e in loaded] == ["run_started", "tool_completed", "run_completed"]

    def test_search_events_fts(self, tmp_path):
        store = _make_store(tmp_path)
        store.upsert_run(
            RunRecord(
                run_id="r1",
                task="t",
                started_at="2026-04-21T00:00:00+00:00",
                events=[
                    {"type": "tool_completed", "timestamp": "t", "data": {"tool": "web_search", "content": "unique_needle_xyz"}}
                ],
            )
        )
        hits = store.search_events("unique_needle_xyz")
        assert len(hits) == 1
        assert hits[0]["run_id"] == "r1"

    def test_search_events_no_match_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.search_events("nothing_here") == []


# ---- Growth state --------------------------------------------------------


class TestGrowthState:
    def test_update_fields(self, tmp_path):
        store = _make_store(tmp_path)
        store.update_growth_state(current_level=1, total_tasks_run=5)
        state = store.read_growth_state()
        assert state["current_level"] == 1
        assert state["total_tasks_run"] == 5

    def test_increment_tokens_used(self, tmp_path):
        store = _make_store(tmp_path)
        store.increment_tokens_used(1000)
        store.increment_tokens_used(500)
        assert store.read_growth_state()["tokens_used_today"] == 1500

    def test_increment_ignores_non_positive(self, tmp_path):
        store = _make_store(tmp_path)
        store.increment_tokens_used(0)
        store.increment_tokens_used(-100)
        assert store.read_growth_state()["tokens_used_today"] == 0

    def test_reset_daily_budget(self, tmp_path):
        store = _make_store(tmp_path)
        store.increment_tokens_used(5000)
        store.reset_daily_budget(today_utc="2026-04-21")
        state = store.read_growth_state()
        assert state["tokens_used_today"] == 0
        assert state["last_reset"] == "2026-04-21"


# ---- Curriculum queue ----------------------------------------------------


class TestCurriculumQueue:
    def test_enqueue_and_claim(self, tmp_path):
        store = _make_store(tmp_path)
        tid = store.enqueue_task(level=1, task="echo hello", source="seed")
        assert tid > 0
        claimed = store.claim_next_task()
        assert claimed["id"] == tid
        assert claimed["status"] == "running"

    def test_claim_respects_level_order(self, tmp_path):
        store = _make_store(tmp_path)
        store.enqueue_task(level=2, task="l2", source="seed")
        store.enqueue_task(level=0, task="l0", source="seed")
        store.enqueue_task(level=1, task="l1", source="seed")
        # Claim 3 times — should come out in level order (0, 1, 2).
        first = store.claim_next_task()
        second = store.claim_next_task()
        third = store.claim_next_task()
        assert first["level"] == 0
        assert second["level"] == 1
        assert third["level"] == 2

    def test_claim_respects_level_filter(self, tmp_path):
        store = _make_store(tmp_path)
        store.enqueue_task(level=0, task="l0", source="seed")
        store.enqueue_task(level=1, task="l1", source="seed")
        claimed = store.claim_next_task(level=1)
        assert claimed["level"] == 1

    def test_claim_none_when_empty(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.claim_next_task() is None

    def test_complete_task_sets_status_done(self, tmp_path):
        store = _make_store(tmp_path)
        tid = store.enqueue_task(level=0, task="t", source="seed")
        store.claim_next_task()
        store.complete_task(tid, run_id="r123")
        # Now pending should be empty
        assert store.pending_tasks() == []

    def test_pending_tasks_filter_by_level(self, tmp_path):
        store = _make_store(tmp_path)
        store.enqueue_task(level=0, task="l0", source="seed")
        store.enqueue_task(level=1, task="l1", source="seed")
        assert len(store.pending_tasks(level=0)) == 1
        assert len(store.pending_tasks()) == 2

    def test_keywords_serialized_roundtrip(self, tmp_path):
        store = _make_store(tmp_path)
        store.enqueue_task(
            level=1, task="t", source="seed",
            expected_outcome_keywords=["alpha", "beta"],
        )
        claimed = store.claim_next_task()
        assert claimed["expected_outcome_keywords"] == ["alpha", "beta"]


# ---- Import from jsonl ---------------------------------------------------


class TestImportJsonl:
    def test_imports_valid_records(self, tmp_path):
        jsonl = tmp_path / "runs.jsonl"
        record = {
            "run_id": "imported_1",
            "task": "imported task",
            "started_at": "2026-04-20T00:00:00+00:00",
            "completed_at": "2026-04-20T00:00:01+00:00",
            "final_status": "success",
            "events": _sample_events(),
        }
        jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

        store = _make_store(tmp_path)
        count = store.import_jsonl(jsonl)
        assert count == 1
        got = store.get_run("imported_1")
        assert got is not None
        assert len(got.events) == 3

    def test_import_is_idempotent(self, tmp_path):
        jsonl = tmp_path / "runs.jsonl"
        record = {
            "run_id": "r1",
            "task": "t",
            "started_at": "2026-04-20T00:00:00+00:00",
            "events": [],
        }
        jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")
        store = _make_store(tmp_path)
        store.import_jsonl(jsonl)
        store.import_jsonl(jsonl)  # Second call must not duplicate.
        assert len(store.list_runs()) == 1

    def test_import_skips_bad_lines(self, tmp_path):
        jsonl = tmp_path / "runs.jsonl"
        good = json.dumps({"run_id": "ok", "task": "t", "started_at": "2026-04-20T00:00:00+00:00"})
        jsonl.write_text("NOT JSON\n" + good + "\n\n", encoding="utf-8")
        store = _make_store(tmp_path)
        count = store.import_jsonl(jsonl)
        assert count == 1

    def test_import_nonexistent_file_returns_zero(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.import_jsonl(tmp_path / "missing.jsonl") == 0
