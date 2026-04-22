"""curriculum — task generator for the Autonomous Growth loop.

Pure data module. Produces task dicts compatible with seed_loader's
output shape so autoschool can consume seed and generated tasks
uniformly.

Levels:
  L0  perception — per-tool probes (describe, params). Template-only,
      no LLM, deterministic. Runs during bootstrap to map the tool
      landscape.
  L1  atomic ops — single-step operations (string manipulation, parsing,
      arithmetic). LLM meta-prompt generates these. Stub in Day 2 of
      Week 2; filled in once we have L0 outcome data to steer the prompt.

A generator never writes to storage directly — the `commit_tasks`
helper does that so the generation step stays testable and offline.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .storage import Storage
    from .tools import ToolRegistry

logger = logging.getLogger("wanxiang.curriculum")


# ---------------------------------------------------------------------------
# L0 — perception (no LLM)
# ---------------------------------------------------------------------------


def generate_l0_tasks(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Generate 2 probe tasks per non-synthesized tool in the registry.

    Task shape matches seed_loader output: {level, task, source_id,
    expected_outcome_keywords}. source_id is deterministic (tool name +
    probe type) so repeated calls deduplicate via enqueue_task_if_new.
    """
    tasks: list[dict[str, Any]] = []
    for name in registry.list_tools():
        spec = registry.get(name)
        if spec is None:
            continue
        # Synthesized tools are targets of L1+ learning, not L0 discovery.
        if spec.group == "synthesized":
            continue
        tasks.append(
            {
                "level": 0,
                "task": f"Describe the purpose of the tool '{name}' in one sentence.",
                "source_id": f"l0_describe_{name}",
                "expected_outcome_keywords": [name],
            }
        )
        tasks.append(
            {
                "level": 0,
                "task": f"Explain what parameters the tool '{name}' accepts and what each one means.",
                "source_id": f"l0_params_{name}",
                "expected_outcome_keywords": [name, "parameter"],
            }
        )
    return tasks


# ---------------------------------------------------------------------------
# L1 — atomic operations (LLM meta-prompt)
# ---------------------------------------------------------------------------


def generate_l1_tasks(count: int = 5) -> list[dict[str, Any]]:
    """Placeholder for Day 2. Returns empty list until the LLM meta-prompt
    generator lands.

    Intended behavior once implemented: call the director/planning LLM
    with a prompt like "Given the following existing L1 skills: [...],
    propose N new atomic single-step tasks that are slightly harder but
    still solvable by a single agent. Each task must have a
    deterministic success check."
    """
    return []


# ---------------------------------------------------------------------------
# Commit helper — writes a batch to the curriculum queue
# ---------------------------------------------------------------------------


def commit_tasks(
    storage: Storage,
    tasks: list[dict[str, Any]],
    *,
    source: str = "generated",
) -> int:
    """Enqueue a batch. Returns count of newly-enqueued tasks.

    Uses enqueue_task_if_new so re-calling with the same tasks is a
    no-op. The `source` label lets autoschool tell seed vs generated
    vs manual tasks apart downstream.
    """
    loaded = 0
    for task in tasks:
        inserted = storage.enqueue_task_if_new(
            level=int(task["level"]),
            task=str(task["task"]),
            source=source,
            expected_outcome_keywords=task.get("expected_outcome_keywords") or None,
        )
        if inserted is not None:
            loaded += 1
    if loaded:
        logger.info("Committed %d new %s task(s)", loaded, source)
    return loaded
