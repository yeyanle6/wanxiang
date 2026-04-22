"""project — facade for project entity lifecycle.

Thin layer above storage.ProjectRecord / create_project / update_project_status.
This module owns id generation, slug sanitization, and the legal status
transitions. Workspace filesystem bootstrap lives in workspace.py so
this file stays I/O-free beyond SQLite.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .storage import ProjectRecord, Storage

logger = logging.getLogger("wanxiang.project")


# Legal statuses. Transitions are not hard-enforced (LLM-native systems
# sometimes jump states for good reasons) but callers should use these
# constants for readability and to keep schema-level values stable.
STATUS_ELICIT = "elicit"           # gathering requirements from user
STATUS_PLANNING = "planning"       # drafting milestones
STATUS_IMPLEMENTING = "implementing"  # active work
STATUS_BLOCKED = "blocked"         # waiting on user input
STATUS_DONE = "done"               # project shipped
STATUS_ARCHIVED = "archived"       # closed, retained for memory

ALL_STATUSES = frozenset({
    STATUS_ELICIT,
    STATUS_PLANNING,
    STATUS_IMPLEMENTING,
    STATUS_BLOCKED,
    STATUS_DONE,
    STATUS_ARCHIVED,
})

_SLUG_CLEAN = re.compile(r"[^a-z0-9-]+")
_SLUG_HYPHENS = re.compile(r"-+")
_MAX_SLUG_LEN = 48


def slugify(name: str) -> str:
    """Filesystem-safe slug: lowercase, alnum + hyphens only, bounded length.

    Empty or pathological names produce "project" as a fallback — workspace
    bootstrap will add the project_id suffix to guarantee uniqueness.
    """
    base = name.strip().lower().replace(" ", "-").replace("_", "-")
    cleaned = _SLUG_CLEAN.sub("-", base)
    cleaned = _SLUG_HYPHENS.sub("-", cleaned).strip("-")
    if not cleaned:
        cleaned = "project"
    return cleaned[:_MAX_SLUG_LEN]


def _next_unique_slug(storage: Storage, base_slug: str) -> str:
    """Suffix with -2, -3, ... when the slug collides."""
    if storage.get_project_by_slug(base_slug) is None:
        return base_slug
    counter = 2
    while True:
        candidate = f"{base_slug}-{counter}"
        if storage.get_project_by_slug(candidate) is None:
            return candidate
        counter += 1


def create_project(
    storage: Storage,
    *,
    name: str,
    user_goal: str,
    workspace_dir: str,
    status: str = STATUS_ELICIT,
) -> ProjectRecord:
    """Generate id + slug, persist, return record. No filesystem side effects.

    Workspace directory path is taken as-is — callers (usually workspace.py)
    are responsible for actually creating it on disk.
    """
    cleaned_name = (name or "").strip()
    cleaned_goal = (user_goal or "").strip()
    if not cleaned_name:
        raise ValueError("Project name cannot be empty.")
    if not cleaned_goal:
        raise ValueError("Project user_goal cannot be empty.")
    if status not in ALL_STATUSES:
        raise ValueError(f"Unknown status '{status}'. Valid: {sorted(ALL_STATUSES)}")

    project_id = str(uuid4())
    slug = _next_unique_slug(storage, slugify(cleaned_name))

    record = storage.create_project(
        project_id=project_id,
        name=cleaned_name,
        slug=slug,
        user_goal=cleaned_goal,
        workspace_dir=workspace_dir,
        status=status,
    )
    logger.info(
        "Created project id=%s slug=%s status=%s", project_id, slug, status
    )
    return record


def load_project(storage: Storage, project_id: str) -> ProjectRecord | None:
    return storage.get_project(project_id)


def load_project_by_slug(storage: Storage, slug: str) -> ProjectRecord | None:
    return storage.get_project_by_slug(slug)


def update_status(
    storage: Storage,
    project_id: str,
    new_status: str,
    *,
    blocked_on: str | None = None,
) -> ProjectRecord:
    """Transition status. Raises if unknown status or project missing."""
    if new_status not in ALL_STATUSES:
        raise ValueError(f"Unknown status '{new_status}'. Valid: {sorted(ALL_STATUSES)}")
    existing = storage.get_project(project_id)
    if existing is None:
        raise ValueError(f"Project not found: {project_id}")
    storage.update_project_status(project_id, status=new_status, blocked_on=blocked_on)
    updated = storage.get_project(project_id)
    assert updated is not None  # just wrote it
    logger.info(
        "Project %s status %s → %s", project_id, existing.status, new_status
    )
    return updated
