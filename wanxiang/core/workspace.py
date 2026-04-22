"""workspace — filesystem bootstrapper for project workspaces.

A project workspace is a directory under projects/<slug>/ that contains:
  .venv/                 isolated Python environment for the project
  project_metadata.json  project_id, name, goal, created_at — human- and
                         LLM-readable snapshot so agents can inspect a
                         workspace without SQLite access
  understanding.md       (written by Phase 2 requirements analyst)
  decisions.md           (ditto)
  milestones.md          (ditto)
  <project source files>

This module only owns the initial bootstrap: creating the directory,
creating the venv, writing metadata. The rest is the project's job.

Security: all paths are resolved + confined to projects_root. Any attempt
to escape (via `../` in slug or absolute path) raises ValueError.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .storage import ProjectRecord

logger = logging.getLogger("wanxiang.workspace")


DEFAULT_PROJECTS_ROOT = Path(__file__).resolve().parents[2] / "projects"
METADATA_FILENAME = "project_metadata.json"
VENV_DIRNAME = ".venv"


def workspace_root(custom: Path | None = None) -> Path:
    """Resolve the projects root. Callers can override for tests."""
    root = Path(custom) if custom else DEFAULT_PROJECTS_ROOT
    return root.resolve()


def resolve_workspace_path(slug: str, *, projects_root: Path | None = None) -> Path:
    """Compute the canonical path for a slug and guard against escape.

    Returns the resolved absolute path. Raises ValueError if the resolved
    path leaks outside projects_root.
    """
    root = workspace_root(projects_root)
    candidate = (root / slug).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Workspace path escape detected: slug={slug!r} resolved outside {root}"
        ) from exc
    return candidate


def bootstrap_workspace(
    project: ProjectRecord,
    *,
    projects_root: Path | None = None,
    create_venv: bool = True,
    system_python: str | None = None,
) -> Path:
    """Create the workspace directory, a venv, and metadata file.

    Idempotent: existing directory is fine, existing venv not recreated,
    metadata file overwritten (always reflects current project state).

    `system_python` lets tests use a specific interpreter; defaults to
    sys.executable so the venv uses whatever Python is running wanxiang.

    Returns the absolute workspace path.
    """
    root = workspace_root(projects_root)
    root.mkdir(parents=True, exist_ok=True)

    workspace = resolve_workspace_path(project.slug, projects_root=root)
    workspace.mkdir(parents=True, exist_ok=True)

    if create_venv:
        venv_dir = workspace / VENV_DIRNAME
        if not venv_dir.exists():
            python = system_python or sys.executable
            logger.info("Creating venv at %s", venv_dir)
            try:
                # --without-pip is faster and pip can be bootstrapped later
                # when a project actually needs a dep. Most projects will.
                subprocess.run(
                    [python, "-m", "venv", str(venv_dir)],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"venv creation failed: {exc.stderr.decode('utf-8', errors='replace')}"
                ) from exc

    metadata_path = workspace / METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(
            {
                "project_id": project.project_id,
                "name": project.name,
                "slug": project.slug,
                "user_goal": project.user_goal,
                "status": project.status,
                "workspace_dir": str(workspace),
                "created_at": project.created_at,
                "updated_at": project.updated_at,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Bootstrapped workspace for project %s at %s", project.project_id, workspace
    )
    return workspace


def read_metadata(workspace: Path) -> dict | None:
    """Load the metadata file if present. Returns None on missing / malformed."""
    path = Path(workspace) / METADATA_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Malformed metadata at %s", path)
        return None
