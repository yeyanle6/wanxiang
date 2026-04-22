"""Phase 1 pilot — project + conversation end-to-end.

Two tiers:
  1. Route-level unit tests with mocked RunManager.wait_for_run/start_run
     and an injected storage. Runs in the normal pytest pass. Verifies
     routing, NEEDS_CLARIFICATION marker handling, workspace bootstrap.
  2. Full LLM integration (md → html pilot) gated behind
     WANXIANG_RUN_INTEGRATION=1 so it doesn't run in the default suite
     (real Claude CLI call).
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_with_storage(tmp_path, monkeypatch):
    """TestClient with storage enabled, autoschool off, temporary DB + workspace root."""
    monkeypatch.setenv("WANXIANG_STORAGE_DUAL_WRITE", "1")
    monkeypatch.setenv("WANXIANG_AUTOSCHOOL_ENABLED", "0")
    monkeypatch.setenv("WANXIANG_LLM_MODE", "cli")  # prevents real API key requirement

    # Route file paths into tmp_path so the test is hermetic.
    import importlib
    app_module = importlib.import_module("wanxiang.server.app")
    monkeypatch.setattr(app_module, "STORAGE_DB_PATH", tmp_path / "wanxiang.db")
    monkeypatch.setattr(app_module, "RUNS_JSONL_PATH", tmp_path / "runs.jsonl")

    # Patch workspace root so bootstrap lands in tmp_path, not the repo.
    from wanxiang.core import workspace as workspace_module
    monkeypatch.setattr(
        workspace_module, "DEFAULT_PROJECTS_ROOT", tmp_path / "projects"
    )

    with TestClient(app_module.app) as client:
        yield client, tmp_path


# ---- Route-level unit tests (no LLM) -----------------------------------


class TestProjectRoutes:
    def test_create_project_and_workspace(self, app_with_storage):
        client, tmp_path = app_with_storage
        resp = client.post(
            "/api/projects",
            json={"name": "md2html", "user_goal": "Convert Markdown to HTML"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "md2html"
        assert data["slug"] == "md2html"
        assert data["status"] == "elicit"
        # Workspace dir exists on disk.
        workspace = Path(data["workspace_dir"])
        assert workspace.exists()
        assert (workspace / ".venv").exists()
        assert (workspace / "project_metadata.json").exists()

    def test_empty_name_rejected(self, app_with_storage):
        client, _ = app_with_storage
        resp = client.post("/api/projects", json={"name": "", "user_goal": "g"})
        # pydantic min_length=1 → 422 (validation error)
        assert resp.status_code == 422

    def test_get_project(self, app_with_storage):
        client, _ = app_with_storage
        created = client.post(
            "/api/projects", json={"name": "a", "user_goal": "g"}
        ).json()
        resp = client.get(f"/api/projects/{created['project_id']}")
        assert resp.status_code == 200
        assert resp.json()["project_id"] == created["project_id"]

    def test_get_missing_project(self, app_with_storage):
        client, _ = app_with_storage
        resp = client.get("/api/projects/bogus")
        assert resp.status_code == 404

    def test_list_projects(self, app_with_storage):
        client, _ = app_with_storage
        for i in range(3):
            client.post(
                "/api/projects", json={"name": f"p{i}", "user_goal": "g"}
            )
        resp = client.get("/api/projects")
        assert resp.status_code == 200
        assert resp.json()["total"] == 3


class TestConversationRoutes:
    def _make_project(self, client) -> str:
        return client.post(
            "/api/projects", json={"name": "dialog-test", "user_goal": "talk"}
        ).json()["project_id"]

    def test_start_conversation_requires_project(self, app_with_storage):
        client, _ = app_with_storage
        resp = client.post(
            "/api/conversations", json={"project_id": "bogus", "message": "hi"}
        )
        assert resp.status_code == 404

    def test_start_conversation(self, app_with_storage):
        client, _ = app_with_storage
        pid = self._make_project(client)
        resp = client.post(
            "/api/conversations", json={"project_id": pid, "message": "hello"}
        )
        assert resp.status_code == 200
        conv = resp.json()
        assert conv["status"] == "open"

    def test_get_conversation_with_turns(self, app_with_storage):
        client, _ = app_with_storage
        pid = self._make_project(client)
        cid = client.post(
            "/api/conversations", json={"project_id": pid, "message": "first"}
        ).json()["conversation_id"]

        detail = client.get(f"/api/conversations/{cid}").json()
        assert len(detail["turns"]) == 1
        assert detail["turns"][0]["speaker"] == "user"
        assert detail["turns"][0]["content"] == "first"


class TestMessageFlow:
    """POST /conversations/{id}/messages with mocked LLM response."""

    def _make_conv(self, client) -> str:
        pid = client.post(
            "/api/projects", json={"name": "msgtest", "user_goal": "test"}
        ).json()["project_id"]
        return client.post(
            "/api/conversations",
            json={"project_id": pid, "message": "initial goal"},
        ).json()["conversation_id"]

    def test_system_normal_reply_stays_open(self, app_with_storage):
        client, _ = app_with_storage
        cid = self._make_conv(client)

        import importlib
        app_module = importlib.import_module("wanxiang.server.app")

        async def fake_start_run(task, **kwargs):
            # Simulate a run that produces a normal reply via storage events.
            rm = app_module._run_manager
            run_id = "test-run-1"
            # Minimal in-memory state so wait_for_run returns immediately.
            from wanxiang.server.runner import _RunState
            state = _RunState(user_task=task)

            async def noop():
                return None
            import asyncio
            state.task = asyncio.create_task(noop())
            rm._runs[run_id] = state
            # Pre-write the run to storage with an agent_completed event.
            from wanxiang.core.storage import RunRecord
            rm.storage.upsert_run(
                RunRecord(
                    run_id=run_id,
                    task=task,
                    started_at="2026-04-22T00:00:00+00:00",
                    completed_at="2026-04-22T00:00:01+00:00",
                    final_status="success",
                    outcome="success",
                    events=[
                        {
                            "type": "agent_completed",
                            "timestamp": "2026-04-22T00:00:01+00:00",
                            "data": {"agent": "responder", "content": "Hello user"},
                        }
                    ],
                )
            )
            return run_id

        with patch.object(app_module._run_manager, "start_run", side_effect=fake_start_run):
            resp = client.post(
                f"/api/conversations/{cid}/messages", json={"message": "say hi"}
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "open"
        assert data["clarification"] is False
        assert data["system_response"] == "Hello user"
        assert data["next_speaker"] == "user"

    def test_system_clarification_transitions_to_awaiting(self, app_with_storage):
        client, _ = app_with_storage
        cid = self._make_conv(client)

        import importlib
        app_module = importlib.import_module("wanxiang.server.app")

        async def fake_start_run(task, **kwargs):
            rm = app_module._run_manager
            run_id = "test-run-2"
            from wanxiang.server.runner import _RunState
            state = _RunState(user_task=task)
            import asyncio
            async def noop(): return None
            state.task = asyncio.create_task(noop())
            rm._runs[run_id] = state
            from wanxiang.core.storage import RunRecord
            rm.storage.upsert_run(
                RunRecord(
                    run_id=run_id, task=task,
                    started_at="2026-04-22T00:00:00+00:00",
                    completed_at="2026-04-22T00:00:01+00:00",
                    final_status="success", outcome="success",
                    events=[
                        {
                            "type": "agent_completed",
                            "timestamp": "2026-04-22T00:00:01+00:00",
                            "data": {
                                "agent": "responder",
                                "content": "NEEDS_CLARIFICATION: which library?",
                            },
                        }
                    ],
                )
            )
            return run_id

        with patch.object(app_module._run_manager, "start_run", side_effect=fake_start_run):
            resp = client.post(
                f"/api/conversations/{cid}/messages", json={"message": "md to html"}
            )

        data = resp.json()
        assert data["status"] == "awaiting_user"
        assert data["clarification"] is True
        assert "which library" in data["system_response"].lower()
        assert data["next_speaker"] == "user"

    def test_missing_conversation_returns_404(self, app_with_storage):
        client, _ = app_with_storage
        resp = client.post(
            "/api/conversations/missing-id/messages", json={"message": "x"}
        )
        assert resp.status_code == 404


# ---- Real LLM integration (gated) --------------------------------------


@pytest.mark.skipif(
    os.getenv("WANXIANG_RUN_INTEGRATION", "").lower() not in {"1", "true", "yes"},
    reason="Requires WANXIANG_RUN_INTEGRATION=1 (hits real LLM via Claude CLI)",
)
class TestMd2HtmlPilot:
    """Full md → html pilot per plan. Needs authenticated Claude CLI."""

    def test_pilot_end_to_end(self, app_with_storage):
        client, tmp_path = app_with_storage

        # 1. Create project.
        p = client.post(
            "/api/projects",
            json={
                "name": "md2html",
                "user_goal": "Convert a Markdown file to HTML using Python",
            },
        )
        assert p.status_code == 200
        project = p.json()

        # 2. Start conversation.
        conv_resp = client.post(
            "/api/conversations",
            json={
                "project_id": project["project_id"],
                "message": "I want a simple CLI that converts Markdown to HTML.",
            },
        )
        assert conv_resp.status_code == 200
        cid = conv_resp.json()["conversation_id"]

        # 3. Post a follow-up message; system should respond.
        msg = client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "Use the mistune library and read from stdin."},
        )
        assert msg.status_code == 200
        data = msg.json()
        # Status is either open (done) or awaiting_user (wanted more info).
        assert data["status"] in {"open", "awaiting_user"}
        assert data["system_response"].strip() != ""
