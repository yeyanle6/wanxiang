"""Endpoint-level tests for /api/trace/mining.

We call the handler coroutine directly with a faked RunManager rather
than booting TestClient — the real lifespan spawns MCP subprocesses and
contacts external services, which isn't appropriate for unit tests.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

import sys

import wanxiang.server.app  # noqa: F401 — populate sys.modules entry
from wanxiang.server.app import _parse_query_datetime, get_trace_mining

# `wanxiang.server.__init__` rebinds the `app` name to the FastAPI instance,
# so `wanxiang.server.app` as a dotted attribute resolves to that instance.
# Reach the actual module object via sys.modules.
app_module = sys.modules["wanxiang.server.app"]


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "trace_mining"


class _FakeRegistry:
    def __init__(self, audit: list[dict], groups: dict[str, str]) -> None:
        self._audit = audit
        self._groups = groups

    def get_audit_log(self) -> list[dict]:
        return list(self._audit)

    def get_tool_groups(self) -> dict[str, str]:
        return dict(self._groups)


class _FakeFactory:
    def __init__(
        self,
        *,
        tool_registry: _FakeRegistry,
        synthesis_log: list[dict],
    ) -> None:
        self.tool_registry = tool_registry
        self.synthesis_log = synthesis_log


class _FakeRunManager:
    def __init__(
        self,
        *,
        runs: list[dict],
        audit: list[dict],
        groups: dict[str, str],
        synthesis_log: list[dict],
    ) -> None:
        self._runs = runs
        self.factory = _FakeFactory(
            tool_registry=_FakeRegistry(audit, groups),
            synthesis_log=synthesis_log,
        )

    async def read_raw_history(self) -> list[dict]:
        return list(self._runs)


def _load_fixture(name: str) -> Any:
    path = FIXTURE_DIR / name
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


@pytest.fixture
def fake_manager() -> _FakeRunManager:
    return _FakeRunManager(
        runs=_load_fixture("runs.jsonl"),
        audit=_load_fixture("audit_log.json"),
        groups={
            "read_text_file": "filesystem",
            "chinese_numeral_to_int": "synthesized",
            "echo": "",
            "current_time": "",
        },
        synthesis_log=_load_fixture("synthesis_log.json"),
    )


@pytest.fixture(autouse=True)
def _install_fake(monkeypatch: pytest.MonkeyPatch, fake_manager) -> None:
    monkeypatch.setattr(app_module, "_run_manager", fake_manager)


def test_endpoint_returns_full_report(fake_manager):
    response = asyncio.run(get_trace_mining(after=None, before=None))
    payload = response.model_dump()
    assert payload["total_runs"] == 5
    assert payload["final_status_distribution"] == {"success": 3, "error": 2}
    assert set(payload["workflow_mix"].keys()) == {"review_loop", "pipeline", "parallel"}
    # Tool grouping threaded through from the registry.
    assert payload["tool_usage"]["read_text_file"]["group"] == "mcp"
    assert payload["tool_usage"]["chinese_numeral_to_int"]["group"] == "synthesized"
    assert payload["reviewer_convergence"]["total_review_runs"] == 2


def test_endpoint_respects_after_query():
    after_iso = "2026-04-13T00:00:00+00:00"
    response = asyncio.run(get_trace_mining(after=after_iso, before=None))
    payload = response.model_dump()
    # Only the two runs dated 04-13 onward remain.
    assert payload["total_runs"] == 2
    assert payload["window"]["after"] == after_iso


def test_endpoint_accepts_Z_suffix():
    # ISO-8601 'Z' (zulu) suffix is normalized to +00:00.
    response = asyncio.run(get_trace_mining(after="2026-04-12T00:00:00Z", before=None))
    payload = response.model_dump()
    assert payload["total_runs"] == 3


def test_endpoint_rejects_bad_timestamp():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(get_trace_mining(after="not-a-date", before=None))
    assert exc_info.value.status_code == 400
    assert "after" in exc_info.value.detail


def test_parse_query_datetime_handles_none_and_blank():
    assert _parse_query_datetime("after", None) is None
    assert _parse_query_datetime("after", "   ") is None


def test_parse_query_datetime_parses_iso():
    parsed = _parse_query_datetime("after", "2026-04-10T00:00:00+00:00")
    assert parsed == datetime(2026, 4, 10, tzinfo=timezone.utc)
