"""seed_loader — bootstrap the curriculum queue from configs/seed_tasks.yaml.

Pure I/O module. Autoschool's first run consumes these 30 hand-seeded
tasks before curriculum.py starts generating its own. Idempotent — safe
to call every server startup; rerun only enqueues new tasks.

YAML shape (from configs/seed_tasks.yaml):

  seed_tasks:
    - id: l0_01
      level: 0
      task: "..."
      expected_outcome_keywords: [...]
      tags: [...]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from .storage import Storage

logger = logging.getLogger("wanxiang.seed_loader")


def load_seed_tasks(yaml_path: Path) -> list[dict[str, Any]]:
    """Read and validate the YAML. Bad entries are logged and dropped.

    Returns a list of dicts with keys: level, task, source_id,
    expected_outcome_keywords. Tags are preserved but not structured.
    """
    if not yaml_path.exists():
        logger.warning("Seed tasks file not found: %s", yaml_path)
        return []
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        logger.exception("Failed to parse seed tasks YAML: %s", yaml_path)
        return []

    if not isinstance(raw, dict):
        logger.warning("Seed YAML root must be a dict, got %s", type(raw).__name__)
        return []

    entries = raw.get("seed_tasks")
    if not isinstance(entries, list):
        logger.warning("'seed_tasks' key must be a list in %s", yaml_path)
        return []

    valid: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            logger.warning("Seed entry %d is not a dict; skipping", idx)
            continue
        task_id = str(entry.get("id", "")).strip()
        task_text = str(entry.get("task", "")).strip()
        level = entry.get("level")
        if not task_id or not task_text or not isinstance(level, int):
            logger.warning("Seed entry %d missing id/task/level; skipping", idx)
            continue
        keywords = entry.get("expected_outcome_keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        valid.append(
            {
                "level": int(level),
                "task": task_text,
                "source_id": task_id,
                "expected_outcome_keywords": [str(k) for k in keywords if str(k).strip()],
            }
        )
    return valid


def enqueue_seed_tasks(storage: Storage, yaml_path: Path) -> int:
    """Load YAML and enqueue new entries into the curriculum queue.

    Returns the count of *newly* enqueued tasks (already-present tasks
    are skipped via enqueue_task_if_new, so reruns are cheap no-ops).
    """
    tasks = load_seed_tasks(yaml_path)
    loaded = 0
    for task in tasks:
        inserted = storage.enqueue_task_if_new(
            level=task["level"],
            task=task["task"],
            source="seed",
            expected_outcome_keywords=task["expected_outcome_keywords"] or None,
        )
        if inserted is not None:
            loaded += 1
    if loaded:
        logger.info("Enqueued %d new seed task(s) from %s", loaded, yaml_path)
    return loaded
