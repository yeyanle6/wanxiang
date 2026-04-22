"""Tests for project.py — id generation, slug sanitization, status lifecycle."""
from __future__ import annotations

import pytest

from wanxiang.core.project import (
    ALL_STATUSES,
    STATUS_BLOCKED,
    STATUS_ELICIT,
    STATUS_IMPLEMENTING,
    STATUS_PLANNING,
    create_project,
    load_project,
    load_project_by_slug,
    slugify,
    update_status,
)
from wanxiang.core.storage import Storage


@pytest.fixture
def storage(tmp_path):
    s = Storage(tmp_path / "p.db")
    yield s
    s.close()


# ---- Slugify ------------------------------------------------------------


class TestSlugify:
    def test_lowercase(self):
        assert slugify("MyProject") == "myproject"

    def test_spaces_become_hyphens(self):
        assert slugify("heart rate monitor") == "heart-rate-monitor"

    def test_underscores_become_hyphens(self):
        assert slugify("snake_case_name") == "snake-case-name"

    def test_strips_special_chars(self):
        assert slugify("r$$PPG!!!") == "r-ppg"

    def test_collapses_consecutive_hyphens(self):
        assert slugify("foo---bar") == "foo-bar"

    def test_strips_leading_trailing_hyphens(self):
        assert slugify("---foo---") == "foo"

    def test_empty_input_falls_back(self):
        assert slugify("") == "project"
        assert slugify("   ") == "project"
        assert slugify("!!!") == "project"

    def test_truncates_long_names(self):
        s = slugify("a" * 100)
        assert len(s) <= 48

    def test_cjk_falls_back_to_project(self):
        # No ascii-friendly chars → fallback.
        assert slugify("万象") == "project"


# ---- create_project -----------------------------------------------------


class TestCreateProject:
    def test_basic_creation(self, storage):
        rec = create_project(
            storage, name="rPPG Study", user_goal="build rppg pipeline",
            workspace_dir="/tmp/rppg-study",
        )
        assert rec.name == "rPPG Study"
        assert rec.slug == "rppg-study"
        assert rec.user_goal == "build rppg pipeline"
        assert rec.status == STATUS_ELICIT

    def test_returns_persisted_record(self, storage):
        rec = create_project(
            storage, name="X", user_goal="g", workspace_dir="/tmp/x"
        )
        assert load_project(storage, rec.project_id) is not None

    def test_duplicate_name_auto_suffixes_slug(self, storage):
        create_project(storage, name="md2html", user_goal="g", workspace_dir="/tmp/a")
        r2 = create_project(storage, name="md2html", user_goal="g", workspace_dir="/tmp/b")
        assert r2.slug == "md2html-2"
        r3 = create_project(storage, name="md2html", user_goal="g", workspace_dir="/tmp/c")
        assert r3.slug == "md2html-3"

    def test_empty_name_rejected(self, storage):
        with pytest.raises(ValueError, match="name"):
            create_project(storage, name="   ", user_goal="g", workspace_dir="/tmp/x")

    def test_empty_goal_rejected(self, storage):
        with pytest.raises(ValueError, match="user_goal"):
            create_project(storage, name="x", user_goal="", workspace_dir="/tmp/x")

    def test_invalid_initial_status_rejected(self, storage):
        with pytest.raises(ValueError, match="status"):
            create_project(
                storage, name="x", user_goal="g",
                workspace_dir="/tmp/x", status="bogus",
            )

    def test_custom_initial_status(self, storage):
        rec = create_project(
            storage, name="x", user_goal="g",
            workspace_dir="/tmp/x", status=STATUS_PLANNING,
        )
        assert rec.status == STATUS_PLANNING

    def test_project_id_unique(self, storage):
        ids = {
            create_project(storage, name=f"n{i}", user_goal="g", workspace_dir=f"/tmp/n{i}").project_id
            for i in range(5)
        }
        assert len(ids) == 5


# ---- load_project -------------------------------------------------------


class TestLoadProject:
    def test_returns_none_for_missing(self, storage):
        assert load_project(storage, "nope") is None

    def test_loads_by_slug(self, storage):
        rec = create_project(
            storage, name="Heart Rate", user_goal="g",
            workspace_dir="/tmp/hr",
        )
        got = load_project_by_slug(storage, rec.slug)
        assert got is not None
        assert got.project_id == rec.project_id


# ---- update_status ------------------------------------------------------


class TestUpdateStatus:
    def test_basic_transition(self, storage):
        rec = create_project(
            storage, name="x", user_goal="g", workspace_dir="/tmp/x"
        )
        updated = update_status(storage, rec.project_id, STATUS_IMPLEMENTING)
        assert updated.status == STATUS_IMPLEMENTING

    def test_persisted(self, storage):
        rec = create_project(
            storage, name="x", user_goal="g", workspace_dir="/tmp/x"
        )
        update_status(storage, rec.project_id, STATUS_BLOCKED, blocked_on="need API key")
        reloaded = load_project(storage, rec.project_id)
        assert reloaded.status == STATUS_BLOCKED
        assert reloaded.blocked_on == "need API key"

    def test_unknown_status_rejected(self, storage):
        rec = create_project(
            storage, name="x", user_goal="g", workspace_dir="/tmp/x"
        )
        with pytest.raises(ValueError, match="status"):
            update_status(storage, rec.project_id, "not-a-real-status")

    def test_missing_project_rejected(self, storage):
        with pytest.raises(ValueError, match="not found"):
            update_status(storage, "nonexistent", STATUS_IMPLEMENTING)

    def test_clearing_blocked_on(self, storage):
        rec = create_project(
            storage, name="x", user_goal="g", workspace_dir="/tmp/x"
        )
        update_status(storage, rec.project_id, STATUS_BLOCKED, blocked_on="need X")
        update_status(storage, rec.project_id, STATUS_IMPLEMENTING, blocked_on="")
        reloaded = load_project(storage, rec.project_id)
        assert reloaded.status == STATUS_IMPLEMENTING
        assert reloaded.blocked_on is None


class TestStatusConstants:
    def test_all_statuses_is_frozenset(self):
        assert isinstance(ALL_STATUSES, frozenset)

    def test_contains_expected_labels(self):
        assert "elicit" in ALL_STATUSES
        assert "planning" in ALL_STATUSES
        assert "implementing" in ALL_STATUSES
        assert "blocked" in ALL_STATUSES
        assert "done" in ALL_STATUSES
        assert "archived" in ALL_STATUSES
