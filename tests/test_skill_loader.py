"""Tests for skill_loader — list_skills, load_approved_skills, approve_skill."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wanxiang.core.skill_loader import (
    SkillRecord,
    approve_skill,
    list_skills,
    load_approved_skills,
)
from wanxiang.core.tier import TierManager
from wanxiang.core.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HANDLER = "def handler(value: str) -> str:\n    return f'echo:{value}'\n"
_SCHEMA = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}


def _write_manifest(
    skills_dir: Path,
    tool_name: str,
    *,
    approved: bool = False,
    tier_level: int = 0,
    handler_code: str = _HANDLER,
) -> None:
    manifest = {
        "tool_name": tool_name,
        "description": f"Test tool {tool_name}",
        "input_schema": _SCHEMA,
        "handler_code": handler_code,
        "tier_level": tier_level,
        "approved": approved,
        "created_at": "2026-04-21T00:00:00+00:00",
    }
    (skills_dir / f"{tool_name}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# list_skills
# ---------------------------------------------------------------------------


class TestListSkills:
    def test_nonexistent_dir_returns_empty(self, tmp_path):
        assert list_skills(tmp_path / "nope") == []

    def test_empty_dir_returns_empty(self, tmp_path):
        assert list_skills(tmp_path) == []

    def test_returns_all_records(self, tmp_path):
        _write_manifest(tmp_path, "tool_a", approved=True)
        _write_manifest(tmp_path, "tool_b", approved=False)
        records = list_skills(tmp_path)
        names = [r.tool_name for r in records]
        assert "tool_a" in names
        assert "tool_b" in names
        assert len(records) == 2

    def test_sorted_by_tool_name(self, tmp_path):
        _write_manifest(tmp_path, "zz_tool")
        _write_manifest(tmp_path, "aa_tool")
        records = list_skills(tmp_path)
        assert records[0].tool_name == "aa_tool"
        assert records[1].tool_name == "zz_tool"

    def test_approved_flag_preserved(self, tmp_path):
        _write_manifest(tmp_path, "pending", approved=False)
        _write_manifest(tmp_path, "live", approved=True)
        by_name = {r.tool_name: r for r in list_skills(tmp_path)}
        assert not by_name["pending"].approved
        assert by_name["live"].approved

    def test_bad_json_skipped(self, tmp_path):
        (tmp_path / "bad.json").write_text("NOT JSON", encoding="utf-8")
        _write_manifest(tmp_path, "good")
        records = list_skills(tmp_path)
        assert len(records) == 1
        assert records[0].tool_name == "good"

    def test_missing_tool_name_skipped(self, tmp_path):
        (tmp_path / "empty.json").write_text(
            json.dumps({"description": "no name", "handler_code": _HANDLER}),
            encoding="utf-8",
        )
        _write_manifest(tmp_path, "real_tool")
        records = list_skills(tmp_path)
        assert len(records) == 1

    def test_to_dict_excludes_handler_code(self, tmp_path):
        _write_manifest(tmp_path, "my_tool")
        record = list_skills(tmp_path)[0]
        d = record.to_dict()
        assert "handler_code" not in d
        assert d["tool_name"] == "my_tool"


# ---------------------------------------------------------------------------
# load_approved_skills
# ---------------------------------------------------------------------------


class TestLoadApprovedSkills:
    def test_empty_dir_loads_nothing(self, tmp_path):
        registry = ToolRegistry()
        count = load_approved_skills(tmp_path, registry)
        assert count == 0

    def test_pending_tools_not_loaded(self, tmp_path):
        _write_manifest(tmp_path, "pending", approved=False)
        registry = ToolRegistry()
        count = load_approved_skills(tmp_path, registry)
        assert count == 0
        assert registry.get("pending") is None

    def test_approved_tool_registered(self, tmp_path):
        _write_manifest(tmp_path, "echo_tool", approved=True)
        registry = ToolRegistry()
        count = load_approved_skills(tmp_path, registry)
        assert count == 1
        assert registry.get("echo_tool") is not None

    def test_handler_callable_correctly(self, tmp_path):
        _write_manifest(tmp_path, "echo_tool", approved=True)
        registry = ToolRegistry()
        load_approved_skills(tmp_path, registry)
        spec = registry.get("echo_tool")
        assert spec is not None
        result = spec.handler(value="hello")
        assert result == "echo:hello"

    def test_tier_manager_initialized(self, tmp_path):
        _write_manifest(tmp_path, "tiered", approved=True, tier_level=1)
        registry = ToolRegistry()
        tier = TierManager()
        load_approved_skills(tmp_path, registry, tier)
        summary = tier.get_tier_summary()
        assert "tiered" in summary["tools"]
        assert summary["tools"]["tiered"]["level"] == 1

    def test_already_registered_skipped(self, tmp_path):
        _write_manifest(tmp_path, "dup", approved=True)
        registry = ToolRegistry()
        count1 = load_approved_skills(tmp_path, registry)
        count2 = load_approved_skills(tmp_path, registry)
        assert count1 == 1
        assert count2 == 0

    def test_bad_handler_code_skipped_gracefully(self, tmp_path):
        _write_manifest(tmp_path, "broken", approved=True, handler_code="SYNTAX !!!)")
        registry = ToolRegistry()
        count = load_approved_skills(tmp_path, registry)
        assert count == 0

    def test_no_callable_in_namespace_skipped(self, tmp_path):
        _write_manifest(tmp_path, "no_fn", approved=True, handler_code="x = 42")
        registry = ToolRegistry()
        count = load_approved_skills(tmp_path, registry)
        assert count == 0

    def test_multiple_approved_all_loaded(self, tmp_path):
        for i in range(3):
            _write_manifest(tmp_path, f"tool_{i}", approved=True)
        registry = ToolRegistry()
        count = load_approved_skills(tmp_path, registry)
        assert count == 3


# ---------------------------------------------------------------------------
# approve_skill
# ---------------------------------------------------------------------------


class TestApproveSkill:
    def test_returns_none_for_missing(self, tmp_path):
        result = approve_skill(tmp_path, "nonexistent")
        assert result is None

    def test_flips_approved_flag(self, tmp_path):
        _write_manifest(tmp_path, "pending", approved=False)
        record = approve_skill(tmp_path, "pending")
        assert record is not None
        assert record.approved is True

    def test_json_written_with_approved_true(self, tmp_path):
        _write_manifest(tmp_path, "pending", approved=False)
        approve_skill(tmp_path, "pending")
        data = json.loads((tmp_path / "pending.json").read_text(encoding="utf-8"))
        assert data["approved"] is True

    def test_idempotent_for_already_approved(self, tmp_path):
        _write_manifest(tmp_path, "live", approved=True)
        record = approve_skill(tmp_path, "live")
        assert record is not None
        assert record.approved is True
