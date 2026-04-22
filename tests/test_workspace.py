"""Tests for workspace.py — bootstrap + path guards + metadata."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wanxiang.core.project import create_project
from wanxiang.core.storage import ProjectRecord, Storage
from wanxiang.core.workspace import (
    METADATA_FILENAME,
    VENV_DIRNAME,
    bootstrap_workspace,
    read_metadata,
    resolve_workspace_path,
    workspace_root,
)


@pytest.fixture
def storage(tmp_path):
    s = Storage(tmp_path / "ws.db")
    yield s
    s.close()


def _make_project(storage: Storage, *, slug: str = "pilot", name: str = "Pilot") -> ProjectRecord:
    # Use the project.create_project helper so slug is normalized.
    return create_project(
        storage,
        name=name,
        user_goal="test project",
        workspace_dir=str(workspace_root() / slug),
    )


# ---- Path guards --------------------------------------------------------


class TestPathGuards:
    def test_normal_slug_resolves(self, tmp_path):
        p = resolve_workspace_path("clean-slug", projects_root=tmp_path)
        assert p.parent == tmp_path.resolve()
        assert p.name == "clean-slug"

    def test_dot_dot_escape_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="escape"):
            resolve_workspace_path("../escape", projects_root=tmp_path)

    def test_absolute_path_as_slug_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="escape"):
            resolve_workspace_path("/etc/passwd", projects_root=tmp_path)

    def test_nested_path_rejected(self, tmp_path):
        # Nested paths that end INSIDE root are actually legal (this won't escape)
        # but pathological deep slugs with .. should still be rejected.
        with pytest.raises(ValueError, match="escape"):
            resolve_workspace_path("foo/../../../bar", projects_root=tmp_path)


class TestWorkspaceRoot:
    def test_custom_root_returned(self, tmp_path):
        assert workspace_root(tmp_path) == tmp_path.resolve()

    def test_default_root_under_repo(self):
        r = workspace_root()
        assert r.name == "projects"


# ---- Bootstrap metadata (no venv) --------------------------------------


class TestBootstrapMetadata:
    def test_creates_workspace_dir(self, tmp_path, storage):
        project = _make_project(storage)
        workspace = bootstrap_workspace(
            project, projects_root=tmp_path, create_venv=False
        )
        assert workspace.exists()
        assert workspace.is_dir()

    def test_writes_metadata_json(self, tmp_path, storage):
        project = _make_project(storage)
        workspace = bootstrap_workspace(
            project, projects_root=tmp_path, create_venv=False
        )
        meta_path = workspace / METADATA_FILENAME
        assert meta_path.exists()
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["project_id"] == project.project_id
        assert data["slug"] == project.slug
        assert data["user_goal"] == project.user_goal

    def test_idempotent(self, tmp_path, storage):
        # Calling twice should not raise.
        project = _make_project(storage)
        bootstrap_workspace(project, projects_root=tmp_path, create_venv=False)
        bootstrap_workspace(project, projects_root=tmp_path, create_venv=False)

    def test_metadata_refreshes_on_rebootstrap(self, tmp_path, storage):
        project = _make_project(storage)
        bootstrap_workspace(project, projects_root=tmp_path, create_venv=False)

        # Simulate a status change
        from wanxiang.core.project import update_status, STATUS_IMPLEMENTING
        updated = update_status(storage, project.project_id, STATUS_IMPLEMENTING)

        bootstrap_workspace(updated, projects_root=tmp_path, create_venv=False)
        meta = read_metadata(workspace_root(tmp_path) / updated.slug)
        assert meta["status"] == STATUS_IMPLEMENTING


class TestReadMetadata:
    def test_returns_none_when_missing(self, tmp_path):
        assert read_metadata(tmp_path) is None

    def test_returns_none_on_malformed_json(self, tmp_path):
        (tmp_path / METADATA_FILENAME).write_text("{not json", encoding="utf-8")
        assert read_metadata(tmp_path) is None

    def test_roundtrip(self, tmp_path, storage):
        project = _make_project(storage)
        ws = bootstrap_workspace(project, projects_root=tmp_path, create_venv=False)
        meta = read_metadata(ws)
        assert meta is not None
        assert meta["project_id"] == project.project_id


# ---- venv creation (real integration, slower) --------------------------


class TestVenvCreation:
    def test_creates_real_venv(self, tmp_path, storage):
        project = _make_project(storage)
        ws = bootstrap_workspace(project, projects_root=tmp_path, create_venv=True)
        venv = ws / VENV_DIRNAME
        assert venv.exists()
        assert venv.is_dir()
        # venv always has pyvenv.cfg at the top
        assert (venv / "pyvenv.cfg").exists()

    def test_venv_not_recreated_when_present(self, tmp_path, storage):
        project = _make_project(storage)
        ws = bootstrap_workspace(project, projects_root=tmp_path, create_venv=True)
        venv = ws / VENV_DIRNAME
        # Plant a marker file inside the venv; second bootstrap should not delete it.
        marker = venv / "MARKER"
        marker.write_text("preserved", encoding="utf-8")

        bootstrap_workspace(project, projects_root=tmp_path, create_venv=True)
        assert marker.exists()
        assert marker.read_text(encoding="utf-8") == "preserved"
