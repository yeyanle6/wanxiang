"""SQLite storage for runs, events, growth state, and curriculum queue.

Designed for the autonomous growth mode: every run the autoschool produces
must be queryable (by outcome, level, source) and searchable (FTS5 over
event data). Parallel writes to runs.jsonl are handled by the caller —
this module is the SQLite side only.

Tables:
  runs              one row per completed run (summary + outcome tag)
  events            one row per event; joined to runs via run_id
  events_fts        FTS5 shadow of events.data for search
  growth_state      singleton row tracking curriculum level + budget
  curriculum_queue  pending / in-flight / done tasks the autoschool consumes

All writes are idempotent on run_id (INSERT OR REPLACE).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


logger = logging.getLogger("wanxiang.storage")

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    task          TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    completed_at  TEXT,
    final_status  TEXT,
    outcome       TEXT,
    level         INTEGER,
    source        TEXT NOT NULL DEFAULT 'user',
    total_tokens  INTEGER NOT NULL DEFAULT 0,
    event_count   INTEGER NOT NULL DEFAULT 0,
    tagged_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_runs_outcome    ON runs(outcome);
CREATE INDEX IF NOT EXISTS idx_runs_source     ON runs(source);
CREATE INDEX IF NOT EXISTS idx_runs_level      ON runs(level);

CREATE TABLE IF NOT EXISTS events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    event_type  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    data        TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_type   ON events(event_type);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    run_id     UNINDEXED,
    event_type UNINDEXED,
    data
);

CREATE TABLE IF NOT EXISTS growth_state (
    id                    INTEGER PRIMARY KEY CHECK (id = 1),
    current_level         INTEGER NOT NULL DEFAULT 0,
    started_at            TEXT NOT NULL,
    total_tasks_run       INTEGER NOT NULL DEFAULT 0,
    consecutive_successes INTEGER NOT NULL DEFAULT 0,
    budget_daily_tokens   INTEGER NOT NULL DEFAULT 100000,
    tokens_used_today     INTEGER NOT NULL DEFAULT 0,
    last_reset            TEXT,
    last_updated          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS curriculum_queue (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    level                     INTEGER NOT NULL,
    task                      TEXT NOT NULL,
    source                    TEXT NOT NULL,
    expected_outcome_keywords TEXT,
    status                    TEXT NOT NULL DEFAULT 'pending',
    claimed_at                TEXT,
    completed_at              TEXT,
    run_id                    TEXT
);

CREATE INDEX IF NOT EXISTS idx_curriculum_status ON curriculum_queue(status, level);
"""


@dataclass
class RunRecord:
    run_id: str
    task: str
    started_at: str
    completed_at: str | None = None
    final_status: str | None = None
    outcome: str | None = None
    level: int | None = None
    source: str = "user"
    total_tokens: int = 0
    event_count: int = 0
    tagged_at: str | None = None
    events: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "final_status": self.final_status,
            "outcome": self.outcome,
            "level": self.level,
            "source": self.source,
            "total_tokens": self.total_tokens,
            "event_count": self.event_count,
            "tagged_at": self.tagged_at,
            "events": list(self.events or []),
        }


class Storage:
    """Thin wrapper over sqlite3 with a per-instance connection lock.

    SQLite with WAL + NORMAL synchronous is safe for the access pattern
    here (single writer process, many readers). The threading.Lock keeps
    multi-thread use (asyncio.to_thread calls) correct.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._init_schema()

    # ---- Schema / lifecycle --------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            now = _utcnow()
            self._conn.execute(
                """
                INSERT OR IGNORE INTO growth_state
                    (id, current_level, started_at, last_updated)
                VALUES (1, 0, ?, ?)
                """,
                (now, now),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # ---- Runs ---------------------------------------------------------------

    def upsert_run(self, record: RunRecord) -> None:
        events = record.events or []
        with self._tx() as c:
            c.execute(
                """
                INSERT INTO runs
                    (run_id, task, started_at, completed_at, final_status,
                     outcome, level, source, total_tokens, event_count, tagged_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    task=excluded.task,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at,
                    final_status=excluded.final_status,
                    outcome=excluded.outcome,
                    level=excluded.level,
                    source=excluded.source,
                    total_tokens=excluded.total_tokens,
                    event_count=excluded.event_count,
                    tagged_at=excluded.tagged_at
                """,
                (
                    record.run_id,
                    record.task,
                    record.started_at,
                    record.completed_at,
                    record.final_status,
                    record.outcome,
                    record.level,
                    record.source,
                    record.total_tokens,
                    record.event_count or len(events),
                    record.tagged_at,
                ),
            )
            c.execute("DELETE FROM events WHERE run_id = ?", (record.run_id,))
            c.execute("DELETE FROM events_fts WHERE run_id = ?", (record.run_id,))
            for seq, event in enumerate(events):
                etype = str(event.get("type", ""))
                ts = str(event.get("timestamp", ""))
                data_blob = json.dumps(event.get("data") or {}, ensure_ascii=False)
                c.execute(
                    """
                    INSERT INTO events (run_id, seq, event_type, timestamp, data)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (record.run_id, seq, etype, ts, data_blob),
                )
                c.execute(
                    "INSERT INTO events_fts (run_id, event_type, data) VALUES (?, ?, ?)",
                    (record.run_id, etype, data_blob),
                )

    def get_run(self, run_id: str, *, with_events: bool = True) -> RunRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        record = _row_to_run(row)
        if with_events:
            record.events = self._load_events(run_id)
        return record

    def list_runs(
        self,
        *,
        limit: int = 50,
        outcome: str | None = None,
        source: str | None = None,
        level: int | None = None,
    ) -> list[RunRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if outcome is not None:
            clauses.append("outcome = ?")
            params.append(outcome)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if level is not None:
            clauses.append("level = ?")
            params.append(level)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(max(1, int(limit)))
        sql = f"SELECT * FROM runs {where} ORDER BY started_at DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_run(r) for r in rows]

    def update_outcome(self, run_id: str, outcome: str) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE runs SET outcome = ?, tagged_at = ? WHERE run_id = ?",
                (outcome, _utcnow(), run_id),
            )

    def search_events(
        self, query: str, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT run_id, event_type, data
                FROM events_fts
                WHERE events_fts MATCH ?
                LIMIT ?
                """,
                (query, max(1, int(limit))),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                parsed = json.loads(row["data"])
            except (TypeError, json.JSONDecodeError):
                parsed = {}
            out.append(
                {"run_id": row["run_id"], "type": row["event_type"], "data": parsed}
            )
        return out

    def _load_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT event_type, timestamp, data FROM events "
                "WHERE run_id = ? ORDER BY seq ASC",
                (run_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                data = json.loads(row["data"])
            except (TypeError, json.JSONDecodeError):
                data = {}
            out.append({"type": row["event_type"], "timestamp": row["timestamp"], "data": data})
        return out

    # ---- Growth state -------------------------------------------------------

    def read_growth_state(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM growth_state WHERE id = 1"
            ).fetchone()
        return dict(row) if row else {}

    def update_growth_state(self, **fields: Any) -> None:
        if not fields:
            return
        fields["last_updated"] = _utcnow()
        set_clause = ", ".join(f"{k} = ?" for k in fields.keys())
        params = list(fields.values())
        with self._tx() as c:
            c.execute(
                f"UPDATE growth_state SET {set_clause} WHERE id = 1", params
            )

    def increment_tokens_used(self, delta: int) -> None:
        if delta <= 0:
            return
        with self._tx() as c:
            c.execute(
                "UPDATE growth_state SET tokens_used_today = tokens_used_today + ?, "
                "last_updated = ? WHERE id = 1",
                (int(delta), _utcnow()),
            )

    def reset_daily_budget(self, *, today_utc: str | None = None) -> None:
        today = today_utc or datetime.now(timezone.utc).date().isoformat()
        with self._tx() as c:
            c.execute(
                "UPDATE growth_state SET tokens_used_today = 0, last_reset = ?, "
                "last_updated = ? WHERE id = 1",
                (today, _utcnow()),
            )

    # ---- Curriculum queue ---------------------------------------------------

    def enqueue_task(
        self,
        *,
        level: int,
        task: str,
        source: str,
        expected_outcome_keywords: list[str] | None = None,
    ) -> int:
        keywords_json = (
            json.dumps(expected_outcome_keywords, ensure_ascii=False)
            if expected_outcome_keywords
            else None
        )
        with self._tx() as c:
            cursor = c.execute(
                """
                INSERT INTO curriculum_queue
                    (level, task, source, expected_outcome_keywords, status)
                VALUES (?, ?, ?, ?, 'pending')
                """,
                (level, task, source, keywords_json),
            )
            return int(cursor.lastrowid)

    def claim_next_task(self, *, level: int | None = None) -> dict[str, Any] | None:
        sql = "SELECT * FROM curriculum_queue WHERE status = 'pending'"
        params: list[Any] = []
        if level is not None:
            sql += " AND level = ?"
            params.append(level)
        sql += " ORDER BY level ASC, id ASC LIMIT 1"
        with self._tx() as c:
            row = c.execute(sql, params).fetchone()
            if row is None:
                return None
            c.execute(
                "UPDATE curriculum_queue SET status = 'running', claimed_at = ? WHERE id = ?",
                (_utcnow(), row["id"]),
            )
        task_dict = dict(row)
        task_dict["status"] = "running"
        task_dict["claimed_at"] = _utcnow()
        if task_dict.get("expected_outcome_keywords"):
            try:
                task_dict["expected_outcome_keywords"] = json.loads(
                    task_dict["expected_outcome_keywords"]
                )
            except (TypeError, json.JSONDecodeError):
                task_dict["expected_outcome_keywords"] = []
        return task_dict

    def complete_task(self, task_id: int, *, run_id: str | None = None) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE curriculum_queue SET status = 'done', completed_at = ?, run_id = ? "
                "WHERE id = ?",
                (_utcnow(), run_id, task_id),
            )

    def pending_tasks(self, *, level: int | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM curriculum_queue WHERE status = 'pending'"
        params: list[Any] = []
        if level is not None:
            sql += " AND level = ?"
            params.append(level)
        sql += " ORDER BY level ASC, id ASC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ---- Import / migration -------------------------------------------------

    def import_jsonl(self, jsonl_path: Path) -> int:
        """Import existing runs.jsonl records. Idempotent on run_id."""
        if not jsonl_path.exists():
            return 0
        count = 0
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(parsed, dict):
                    continue
                run_id = str(parsed.get("run_id", "")).strip()
                if not run_id:
                    continue
                events = parsed.get("events") if isinstance(parsed.get("events"), list) else []
                record = RunRecord(
                    run_id=run_id,
                    task=str(parsed.get("task", "")),
                    started_at=str(parsed.get("started_at", "")),
                    completed_at=str(parsed.get("completed_at") or "") or None,
                    final_status=str(parsed.get("final_status") or "") or None,
                    source="user",
                    event_count=len(events),
                    events=events,
                )
                self.upsert_run(record)
                count += 1
        return count


# ---- Helpers --------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_run(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        task=row["task"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        final_status=row["final_status"],
        outcome=row["outcome"],
        level=row["level"],
        source=row["source"],
        total_tokens=row["total_tokens"] or 0,
        event_count=row["event_count"] or 0,
        tagged_at=row["tagged_at"],
    )
