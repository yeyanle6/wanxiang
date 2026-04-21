"""skill_loader — load and list persisted synthesized skills.

Scans skills/ for JSON manifests written by SkillForge.
Only manifests with ``approved: true`` are registered at startup.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tools import ToolRegistry, ToolSpec

logger = logging.getLogger("wanxiang.skill_loader")


@dataclass
class SkillRecord:
    tool_name: str
    description: str
    input_schema: dict[str, Any]
    handler_code: str
    approved: bool
    tier_level: int = 0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "description": self.description,
            "input_schema": self.input_schema,
            "approved": self.approved,
            "tier_level": self.tier_level,
            "created_at": self.created_at,
        }


def _load_record(path: Path) -> SkillRecord | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Cannot read skill manifest: %s", path)
        return None
    tool_name = str(data.get("tool_name", "")).strip()
    handler_code = str(data.get("handler_code", "")).strip()
    if not tool_name or not handler_code:
        logger.warning("Skill manifest missing tool_name or handler_code: %s", path)
        return None
    input_schema = data.get("input_schema") or {}
    return SkillRecord(
        tool_name=tool_name,
        description=str(data.get("description", "")).strip(),
        input_schema=input_schema if isinstance(input_schema, dict) else {},
        handler_code=handler_code,
        approved=bool(data.get("approved", False)),
        tier_level=int(data.get("tier_level", 0)),
        created_at=str(data.get("created_at", "")),
    )


def list_skills(skills_dir: Path) -> list[SkillRecord]:
    """Return all skill records (pending + approved), sorted by tool_name."""
    if not skills_dir.exists():
        return []
    records: list[SkillRecord] = []
    for path in sorted(skills_dir.glob("*.json")):
        record = _load_record(path)
        if record is not None:
            records.append(record)
    return records


def load_approved_skills(
    skills_dir: Path,
    registry: ToolRegistry,
    tier_manager: Any = None,
) -> int:
    """Exec and register all approved skills. Returns count of newly loaded tools."""
    loaded = 0
    for record in list_skills(skills_dir):
        if not record.approved:
            continue
        if registry.get(record.tool_name) is not None:
            logger.info("Approved skill '%s' already registered; skipping", record.tool_name)
            continue
        try:
            namespace: dict[str, Any] = {}
            exec(record.handler_code, namespace)  # noqa: S102
            handler_fn = namespace.get("handler") or namespace.get(record.tool_name)
            if not callable(handler_fn):
                logger.warning(
                    "Skill '%s' handler_code has no callable 'handler'", record.tool_name
                )
                continue
        except Exception:
            logger.exception("Failed to exec handler_code for skill '%s'", record.tool_name)
            continue
        tool_spec = ToolSpec(
            name=record.tool_name,
            description=record.description,
            input_schema=record.input_schema,
            handler=handler_fn,
            group="synthesized",
        )
        try:
            registry.register(tool_spec)
        except ValueError:
            logger.exception("Failed to register approved skill '%s'", record.tool_name)
            continue
        if tier_manager is not None:
            tier_manager.initialize_tool(record.tool_name, record.tier_level)
        logger.info("Loaded approved skill '%s' (tier=%d)", record.tool_name, record.tier_level)
        loaded += 1
    return loaded


def approve_skill(skills_dir: Path, tool_name: str) -> SkillRecord | None:
    """Flip approved=true in the manifest JSON. Returns the updated record or None if not found."""
    path = skills_dir / f"{tool_name}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Cannot read skill manifest for approval: %s", path)
        return None
    data["approved"] = True
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return _load_record(path)
