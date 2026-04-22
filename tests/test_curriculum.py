"""Tests for curriculum.py — L0/L1 task generators + commit helper."""
from __future__ import annotations

import pytest

from wanxiang.core.curriculum import commit_tasks, generate_l0_tasks, generate_l1_tasks
from wanxiang.core.storage import Storage
from wanxiang.core.tools import ToolRegistry, ToolSpec


def _make_registry(*names: str, groups: dict[str, str] | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    groups = groups or {}
    for name in names:
        registry.register(
            ToolSpec(
                name=name,
                description=f"desc for {name}",
                input_schema={"type": "object"},
                handler=lambda: "ok",
                group=groups.get(name, ""),
            )
        )
    return registry


# ---------------------------------------------------------------------------
# L0 generator
# ---------------------------------------------------------------------------


class TestGenerateL0Tasks:
    def test_empty_registry_yields_no_tasks(self):
        registry = ToolRegistry()
        assert generate_l0_tasks(registry) == []

    def test_two_tasks_per_tool(self):
        registry = _make_registry("read_file", "write_file")
        tasks = generate_l0_tasks(registry)
        assert len(tasks) == 4

    def test_source_ids_deterministic_and_unique(self):
        registry = _make_registry("tool_a", "tool_b")
        tasks = generate_l0_tasks(registry)
        ids = [t["source_id"] for t in tasks]
        assert len(ids) == len(set(ids))
        # Re-running must produce identical ids (important for dedup).
        tasks2 = generate_l0_tasks(registry)
        assert [t["source_id"] for t in tasks2] == ids

    def test_all_tasks_are_level_zero(self):
        registry = _make_registry("x")
        tasks = generate_l0_tasks(registry)
        assert all(t["level"] == 0 for t in tasks)

    def test_tool_name_in_keywords_and_text(self):
        registry = _make_registry("my_special_tool")
        tasks = generate_l0_tasks(registry)
        for t in tasks:
            assert "my_special_tool" in t["task"]
            assert "my_special_tool" in t["expected_outcome_keywords"]

    def test_synthesized_tools_are_skipped(self):
        # Synthesized tools are targets of L1+, not L0 probes.
        registry = _make_registry(
            "builtin_a",
            "forged_b",
            groups={"forged_b": "synthesized"},
        )
        tasks = generate_l0_tasks(registry)
        names_in_tasks = {n for t in tasks for n in t["expected_outcome_keywords"]}
        assert "builtin_a" in names_in_tasks
        assert "forged_b" not in names_in_tasks

    def test_mcp_group_tools_included(self):
        # MCP tools have non-empty non-'synthesized' group — must include.
        registry = _make_registry(
            "fs_read",
            groups={"fs_read": "filesystem"},
        )
        tasks = generate_l0_tasks(registry)
        assert len(tasks) == 2


# ---------------------------------------------------------------------------
# L1 stub
# ---------------------------------------------------------------------------


class TestGenerateL1Tasks:
    def test_returns_empty_list_until_implemented(self):
        assert generate_l1_tasks() == []
        assert generate_l1_tasks(count=10) == []


# ---------------------------------------------------------------------------
# Commit helper
# ---------------------------------------------------------------------------


class TestCommitTasks:
    def test_commits_all_new_tasks(self, tmp_path):
        store = Storage(tmp_path / "test.db")
        try:
            registry = _make_registry("a", "b")
            tasks = generate_l0_tasks(registry)
            loaded = commit_tasks(store, tasks, source="l0_generator")
            assert loaded == 4
            assert len(store.pending_tasks()) == 4
        finally:
            store.close()

    def test_rerunning_is_noop(self, tmp_path):
        store = Storage(tmp_path / "test.db")
        try:
            registry = _make_registry("a")
            tasks = generate_l0_tasks(registry)
            commit_tasks(store, tasks, source="l0_generator")
            loaded = commit_tasks(store, tasks, source="l0_generator")
            assert loaded == 0
            assert len(store.pending_tasks()) == 2
        finally:
            store.close()

    def test_different_source_dedups_separately(self, tmp_path):
        # A task enqueued under source='seed' doesn't block the same text
        # under source='generated' — dedup key is (task, source).
        store = Storage(tmp_path / "test.db")
        try:
            tasks = [{"level": 0, "task": "same text", "expected_outcome_keywords": []}]
            commit_tasks(store, tasks, source="seed")
            loaded = commit_tasks(store, tasks, source="generated")
            assert loaded == 1
            assert len(store.pending_tasks()) == 2
        finally:
            store.close()

    def test_source_label_is_stored(self, tmp_path):
        store = Storage(tmp_path / "test.db")
        try:
            tasks = [{"level": 1, "task": "x", "expected_outcome_keywords": []}]
            commit_tasks(store, tasks, source="l1_generator")
            pending = store.pending_tasks()
            assert pending[0]["source"] == "l1_generator"
        finally:
            store.close()

    def test_empty_batch_returns_zero(self, tmp_path):
        store = Storage(tmp_path / "test.db")
        try:
            assert commit_tasks(store, [], source="noop") == 0
        finally:
            store.close()
