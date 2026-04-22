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
        assert row["value"] == "3"

    def test_schema1_db_migrates_to_schema2(self, tmp_path):
        # Simulate a DB created at schema 1 (no grade columns), then reopen
        # with the current code and verify the columns are added + data
        # preserved.
        import sqlite3 as sq

        db_path = tmp_path / "legacy.db"
        c = sq.connect(db_path)
        c.execute(
            """CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, task TEXT NOT NULL,
                started_at TEXT NOT NULL, completed_at TEXT,
                final_status TEXT, outcome TEXT, level INTEGER,
                source TEXT NOT NULL DEFAULT 'user',
                total_tokens INTEGER NOT NULL DEFAULT 0,
                event_count INTEGER NOT NULL DEFAULT 0,
                tagged_at TEXT
            )"""
        )
        c.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        c.execute("INSERT INTO meta VALUES ('schema_version', '1')")
        c.execute(
            "INSERT INTO runs (run_id, task, started_at, source) VALUES (?, ?, ?, 'user')",
            ("legacy_run", "legacy task", "2026-04-22T00:00:00+00:00"),
        )
        c.commit()
        c.close()

        # Reopen with current Storage — migration runs.
        store = Storage(db_path)
        try:
            cols = {
                r["name"]
                for r in store._conn.execute("PRAGMA table_info(runs)").fetchall()
            }
            assert {"graded_pass", "grade_reason", "graded_at"}.issubset(cols)
            # Legacy row still reachable after migration.
            got = store.get_run("legacy_run")
            assert got is not None
            assert got.graded_pass is None
        finally:
            store.close()


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


# ---- Projects / Conversations (schema v3) -------------------------------


class TestProjects:
    def test_create_and_get(self, tmp_path):
        store = _make_store(tmp_path)
        rec = store.create_project(
            project_id="p1",
            name="md2html",
            slug="md2html",
            user_goal="convert markdown to html",
            workspace_dir=str(tmp_path / "projects" / "md2html"),
        )
        assert rec.status == "elicit"
        got = store.get_run  # sanity: methods exist
        p = store.get_project("p1")
        assert p is not None
        assert p.name == "md2html"
        assert p.user_goal == "convert markdown to html"

    def test_get_by_slug(self, tmp_path):
        store = _make_store(tmp_path)
        store.create_project(
            project_id="p1", name="X", slug="x", user_goal="g", workspace_dir="/tmp/x"
        )
        p = store.get_project_by_slug("x")
        assert p is not None and p.project_id == "p1"

    def test_duplicate_slug_rejected(self, tmp_path):
        store = _make_store(tmp_path)
        store.create_project(
            project_id="p1", name="A", slug="dup", user_goal="g", workspace_dir="/tmp/a"
        )
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            store.create_project(
                project_id="p2", name="B", slug="dup", user_goal="g", workspace_dir="/tmp/b"
            )

    def test_update_status(self, tmp_path):
        store = _make_store(tmp_path)
        store.create_project(
            project_id="p1", name="X", slug="x", user_goal="g", workspace_dir="/tmp/x"
        )
        store.update_project_status("p1", status="planning")
        assert store.get_project("p1").status == "planning"

    def test_update_status_touches_updated_at(self, tmp_path):
        store = _make_store(tmp_path)
        rec = store.create_project(
            project_id="p1", name="X", slug="x", user_goal="g", workspace_dir="/tmp/x"
        )
        original_updated = rec.updated_at
        # add a tiny gap so timestamps differ
        import time
        time.sleep(0.01)
        store.update_project_status("p1", status="done")
        assert store.get_project("p1").updated_at != original_updated

    def test_blocked_on_set_and_cleared(self, tmp_path):
        store = _make_store(tmp_path)
        store.create_project(
            project_id="p1", name="X", slug="x", user_goal="g", workspace_dir="/tmp/x"
        )
        store.update_project_status("p1", status="blocked", blocked_on="need GitHub token")
        assert store.get_project("p1").blocked_on == "need GitHub token"
        store.update_project_status("p1", status="implementing", blocked_on="")
        assert store.get_project("p1").blocked_on is None

    def test_list_projects_filter_by_status(self, tmp_path):
        store = _make_store(tmp_path)
        for i, status in enumerate(["elicit", "planning", "elicit"]):
            store.create_project(
                project_id=f"p{i}", name=f"N{i}", slug=f"s{i}",
                user_goal="g", workspace_dir=f"/tmp/s{i}",
            )
            if status != "elicit":
                store.update_project_status(f"p{i}", status=status)
        elicit_projects = store.list_projects(status="elicit")
        assert len(elicit_projects) == 2


class TestConversations:
    def _with_project(self, store) -> str:
        store.create_project(
            project_id="p1", name="X", slug="x",
            user_goal="g", workspace_dir="/tmp/x",
        )
        return "p1"

    def test_create_and_get(self, tmp_path):
        store = _make_store(tmp_path)
        self._with_project(store)
        conv = store.create_conversation(conversation_id="c1", project_id="p1")
        assert conv.status == "open"
        got = store.get_conversation("c1")
        assert got is not None
        assert got.project_id == "p1"

    def test_foreign_key_on_project(self, tmp_path):
        store = _make_store(tmp_path)
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            store.create_conversation(
                conversation_id="c1", project_id="missing"
            )

    def test_append_turns_and_list(self, tmp_path):
        store = _make_store(tmp_path)
        self._with_project(store)
        store.create_conversation(conversation_id="c1", project_id="p1")
        store.append_conversation_turn(
            conversation_id="c1", speaker="user", content="hi"
        )
        store.append_conversation_turn(
            conversation_id="c1", speaker="system", content="hello",
        )
        turns = store.list_conversation_turns("c1")
        assert [(t.seq, t.speaker) for t in turns] == [(0, "user"), (1, "system")]

    def test_turn_seq_autoincrements_per_conversation(self, tmp_path):
        # Two conversations get independent seq counters.
        store = _make_store(tmp_path)
        self._with_project(store)
        store.create_conversation(conversation_id="c1", project_id="p1")
        store.create_conversation(conversation_id="c2", project_id="p1")
        store.append_conversation_turn(conversation_id="c1", speaker="user", content="a")
        store.append_conversation_turn(conversation_id="c2", speaker="user", content="b")
        store.append_conversation_turn(conversation_id="c1", speaker="system", content="a2")
        assert [t.seq for t in store.list_conversation_turns("c1")] == [0, 1]
        assert [t.seq for t in store.list_conversation_turns("c2")] == [0]

    def test_update_status(self, tmp_path):
        store = _make_store(tmp_path)
        self._with_project(store)
        store.create_conversation(conversation_id="c1", project_id="p1")
        store.update_conversation_status("c1", "awaiting_user")
        assert store.get_conversation("c1").status == "awaiting_user"

    def test_turn_carries_run_id(self, tmp_path):
        store = _make_store(tmp_path)
        self._with_project(store)
        store.create_conversation(conversation_id="c1", project_id="p1")
        turn = store.append_conversation_turn(
            conversation_id="c1",
            speaker="system",
            content="result",
            run_id="run-123",
        )
        assert turn.run_id == "run-123"

    def test_list_conversations_by_project(self, tmp_path):
        store = _make_store(tmp_path)
        self._with_project(store)
        store.create_conversation(conversation_id="c1", project_id="p1")
        store.create_conversation(conversation_id="c2", project_id="p1")
        found = store.list_conversations(project_id="p1")
        ids = {c.conversation_id for c in found}
        assert ids == {"c1", "c2"}

    def test_cascade_delete_on_project(self, tmp_path):
        store = _make_store(tmp_path)
        self._with_project(store)
        store.create_conversation(conversation_id="c1", project_id="p1")
        store.append_conversation_turn(
            conversation_id="c1", speaker="user", content="x"
        )
        # Cascade delete the project; conversations + turns follow.
        with store._tx() as c:
            c.execute("DELETE FROM projects WHERE project_id = 'p1'")
        assert store.get_conversation("c1") is None
        assert store.list_conversation_turns("c1") == []


class TestSchemaV3Migration:
    def test_v2_db_migrates_to_v3_without_data_loss(self, tmp_path):
        # Legacy v2 DB: runs table + meta schema_version=2, no project tables.
        import sqlite3 as sq
        db_path = tmp_path / "legacy.db"
        c = sq.connect(db_path)
        c.execute(
            """CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, task TEXT NOT NULL,
                started_at TEXT NOT NULL, completed_at TEXT,
                final_status TEXT, outcome TEXT, level INTEGER,
                source TEXT NOT NULL DEFAULT 'user',
                total_tokens INTEGER NOT NULL DEFAULT 0,
                event_count INTEGER NOT NULL DEFAULT 0,
                tagged_at TEXT, graded_pass INTEGER, grade_reason TEXT,
                graded_at TEXT
            )"""
        )
        c.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        c.execute("INSERT INTO meta VALUES ('schema_version', '2')")
        c.execute(
            "INSERT INTO runs (run_id, task, started_at, source) VALUES (?, ?, ?, 'user')",
            ("legacy", "t", "2026-04-22T00:00:00+00:00"),
        )
        c.commit()
        c.close()

        store = Storage(db_path)
        try:
            # Project/Conversation tables must exist after migration.
            cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(projects)").fetchall()}
            assert "project_id" in cols
            assert "workspace_dir" in cols
            # Legacy data preserved.
            legacy = store.get_run("legacy")
            assert legacy is not None and legacy.task == "t"
            # Can create project in migrated DB.
            store.create_project(
                project_id="new", name="N", slug="n",
                user_goal="g", workspace_dir="/tmp/n",
            )
            assert store.get_project("new") is not None
        finally:
            store.close()
