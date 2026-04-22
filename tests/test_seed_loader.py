"""Tests for seed_loader — YAML parsing + idempotent enqueue."""
from __future__ import annotations

from pathlib import Path

import pytest

from wanxiang.core.seed_loader import enqueue_seed_tasks, load_seed_tasks
from wanxiang.core.storage import Storage


def _write_yaml(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


MINIMAL_YAML = """\
seed_tasks:
  - id: l0_01
    level: 0
    task: "列出你当前可以使用的所有工具。"
    expected_outcome_keywords: ["工具"]
  - id: l1_01
    level: 1
    task: "把字符串 'Hello' 反转。"
    expected_outcome_keywords: ["olleH"]
"""


class TestLoadSeedTasks:
    def test_parses_valid_yaml(self, tmp_path):
        p = tmp_path / "seed.yaml"
        _write_yaml(p, MINIMAL_YAML)
        tasks = load_seed_tasks(p)
        assert len(tasks) == 2
        assert tasks[0]["source_id"] == "l0_01"
        assert tasks[0]["level"] == 0
        assert "工具" in tasks[0]["expected_outcome_keywords"]

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_seed_tasks(tmp_path / "nope.yaml") == []

    def test_malformed_yaml_returns_empty(self, tmp_path):
        p = tmp_path / "bad.yaml"
        _write_yaml(p, "not: valid: yaml: everywhere:\n  - [")
        assert load_seed_tasks(p) == []

    def test_missing_seed_tasks_key(self, tmp_path):
        p = tmp_path / "empty.yaml"
        _write_yaml(p, "other_key: foo\n")
        assert load_seed_tasks(p) == []

    def test_skips_bad_entries(self, tmp_path):
        p = tmp_path / "mixed.yaml"
        _write_yaml(
            p,
            """\
seed_tasks:
  - id: ok_01
    level: 0
    task: "valid task"
  - not_a_dict
  - id: ""
    level: 0
    task: "missing id"
  - id: bad_level
    level: "not-an-int"
    task: "bad level"
  - id: missing_task
    level: 0
    task: ""
""",
        )
        tasks = load_seed_tasks(p)
        assert len(tasks) == 1
        assert tasks[0]["source_id"] == "ok_01"

    def test_real_seed_file_loads(self):
        """Smoke test against the actual shipped seed YAML."""
        root = Path(__file__).resolve().parents[1]
        tasks = load_seed_tasks(root / "configs" / "seed_tasks.yaml")
        assert len(tasks) >= 30
        levels = {t["level"] for t in tasks}
        assert levels == {0, 1, 2}


class TestEnqueueSeedTasks:
    def test_first_run_enqueues_all(self, tmp_path):
        store = Storage(tmp_path / "test.db")
        try:
            p = tmp_path / "seed.yaml"
            _write_yaml(p, MINIMAL_YAML)
            loaded = enqueue_seed_tasks(store, p)
            assert loaded == 2
            assert len(store.pending_tasks()) == 2
        finally:
            store.close()

    def test_second_run_is_noop(self, tmp_path):
        store = Storage(tmp_path / "test.db")
        try:
            p = tmp_path / "seed.yaml"
            _write_yaml(p, MINIMAL_YAML)
            enqueue_seed_tasks(store, p)
            loaded_again = enqueue_seed_tasks(store, p)
            assert loaded_again == 0
            assert len(store.pending_tasks()) == 2
        finally:
            store.close()

    def test_partial_additions_only_insert_new(self, tmp_path):
        store = Storage(tmp_path / "test.db")
        try:
            p = tmp_path / "seed.yaml"
            _write_yaml(
                p,
                """\
seed_tasks:
  - id: a
    level: 0
    task: "first"
""",
            )
            enqueue_seed_tasks(store, p)
            _write_yaml(
                p,
                """\
seed_tasks:
  - id: a
    level: 0
    task: "first"
  - id: b
    level: 0
    task: "second"
""",
            )
            loaded = enqueue_seed_tasks(store, p)
            assert loaded == 1
        finally:
            store.close()

    def test_dedup_matches_on_task_text(self, tmp_path):
        # Same text but different id should still dedupe — we match on
        # (task, source) rather than id.
        store = Storage(tmp_path / "test.db")
        try:
            p = tmp_path / "seed.yaml"
            _write_yaml(p, 'seed_tasks:\n  - id: a\n    level: 0\n    task: "same text"\n')
            enqueue_seed_tasks(store, p)
            _write_yaml(p, 'seed_tasks:\n  - id: DIFFERENT_ID\n    level: 0\n    task: "same text"\n')
            assert enqueue_seed_tasks(store, p) == 0
        finally:
            store.close()
