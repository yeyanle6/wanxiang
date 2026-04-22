"""Tests for ConversationManager — lifecycle, turn-taking, marker, rendering."""
from __future__ import annotations

import pytest

from wanxiang.core.conversation import (
    CLARIFICATION_MARKER,
    STATUS_AWAITING_USER,
    STATUS_CLOSED,
    STATUS_OPEN,
    ConversationManager,
)
from wanxiang.core.project import create_project
from wanxiang.core.storage import Storage


@pytest.fixture
def storage(tmp_path):
    s = Storage(tmp_path / "conv.db")
    yield s
    s.close()


@pytest.fixture
def project(storage):
    return create_project(
        storage, name="pilot", user_goal="test", workspace_dir="/tmp/pilot"
    )


@pytest.fixture
def conversations(storage):
    return ConversationManager(storage)


# ---- start --------------------------------------------------------------


class TestStart:
    def test_basic(self, conversations, project):
        conv = conversations.start(project.project_id, "hello")
        assert conv.project_id == project.project_id
        assert conv.status == STATUS_OPEN
        turns = conversations.turns(conv.conversation_id)
        assert len(turns) == 1
        assert turns[0].speaker == "user"
        assert turns[0].content == "hello"

    def test_empty_initial_message_rejected(self, conversations, project):
        with pytest.raises(ValueError, match="Initial"):
            conversations.start(project.project_id, "   ")

    def test_custom_id(self, conversations, project):
        conv = conversations.start(
            project.project_id, "hi", conversation_id="my-conv-id"
        )
        assert conv.conversation_id == "my-conv-id"


# ---- append_user_turn ---------------------------------------------------


class TestAppendUserTurn:
    def test_open_conversation_accepts(self, conversations, project):
        conv = conversations.start(project.project_id, "first")
        conversations.append_user_turn(conv.conversation_id, "second")
        turns = conversations.turns(conv.conversation_id)
        assert [t.content for t in turns] == ["first", "second"]

    def test_awaiting_user_transitions_to_open(self, conversations, project, storage):
        conv = conversations.start(project.project_id, "first")
        storage.update_conversation_status(conv.conversation_id, STATUS_AWAITING_USER)
        conversations.append_user_turn(conv.conversation_id, "my answer")
        assert storage.get_conversation(conv.conversation_id).status == STATUS_OPEN

    def test_closed_rejects(self, conversations, project, storage):
        conv = conversations.start(project.project_id, "hi")
        conversations.close(conv.conversation_id)
        with pytest.raises(ValueError, match="closed"):
            conversations.append_user_turn(conv.conversation_id, "x")

    def test_missing_rejects(self, conversations):
        with pytest.raises(ValueError, match="not found"):
            conversations.append_user_turn("nope", "x")

    def test_empty_rejects(self, conversations, project):
        conv = conversations.start(project.project_id, "first")
        with pytest.raises(ValueError, match="empty"):
            conversations.append_user_turn(conv.conversation_id, "   ")


# ---- append_system_turn -------------------------------------------------


class TestAppendSystemTurn:
    def test_plain_response_stays_open(self, conversations, project):
        conv = conversations.start(project.project_id, "hi")
        conversations.append_system_turn(conv.conversation_id, "here you go")
        assert conversations.get(conv.conversation_id).status == STATUS_OPEN

    def test_clarification_marker_transitions_to_awaiting(self, conversations, project):
        conv = conversations.start(project.project_id, "I want rPPG")
        conversations.append_system_turn(
            conv.conversation_id,
            f"{CLARIFICATION_MARKER} which algorithm?",
        )
        assert conversations.get(conv.conversation_id).status == STATUS_AWAITING_USER

    def test_marker_with_leading_whitespace_still_detected(self, conversations, project):
        conv = conversations.start(project.project_id, "x")
        conversations.append_system_turn(
            conv.conversation_id,
            f"   {CLARIFICATION_MARKER} which lib?",
        )
        assert conversations.get(conv.conversation_id).status == STATUS_AWAITING_USER

    def test_force_status_wins(self, conversations, project):
        conv = conversations.start(project.project_id, "x")
        conversations.append_system_turn(
            conv.conversation_id,
            f"{CLARIFICATION_MARKER} q?",
            force_status=STATUS_CLOSED,
        )
        assert conversations.get(conv.conversation_id).status == STATUS_CLOSED

    def test_invalid_force_status_rejected(self, conversations, project):
        conv = conversations.start(project.project_id, "x")
        with pytest.raises(ValueError, match="Invalid"):
            conversations.append_system_turn(
                conv.conversation_id, "ok", force_status="bogus"
            )

    def test_run_id_attached(self, conversations, project):
        conv = conversations.start(project.project_id, "x")
        turn = conversations.append_system_turn(
            conv.conversation_id, "done", run_id="run-abc"
        )
        assert turn.run_id == "run-abc"


# ---- Marker helpers -----------------------------------------------------


class TestMarker:
    def test_is_clarification_detects(self):
        assert ConversationManager.is_clarification(f"{CLARIFICATION_MARKER} q?")

    def test_is_clarification_rejects_mid_string(self):
        assert not ConversationManager.is_clarification(f"prefix {CLARIFICATION_MARKER} q?")

    def test_is_clarification_empty(self):
        assert not ConversationManager.is_clarification("")

    def test_strip_marker(self):
        assert ConversationManager.strip_marker(f"{CLARIFICATION_MARKER}  what?") == "what?"

    def test_strip_marker_noop_without_marker(self):
        assert ConversationManager.strip_marker("hello") == "hello"


# ---- next_speaker -------------------------------------------------------


class TestNextSpeaker:
    def test_after_start_its_systems_turn(self, conversations, project):
        conv = conversations.start(project.project_id, "hi")
        assert conversations.next_speaker(conv.conversation_id) == "system"

    def test_after_system_reply_its_users_turn(self, conversations, project):
        conv = conversations.start(project.project_id, "hi")
        conversations.append_system_turn(conv.conversation_id, "done")
        assert conversations.next_speaker(conv.conversation_id) == "user"

    def test_awaiting_user_is_users_turn(self, conversations, project, storage):
        conv = conversations.start(project.project_id, "x")
        storage.update_conversation_status(conv.conversation_id, STATUS_AWAITING_USER)
        assert conversations.next_speaker(conv.conversation_id) == "user"

    def test_closed_returns_none(self, conversations, project):
        conv = conversations.start(project.project_id, "x")
        conversations.close(conv.conversation_id)
        assert conversations.next_speaker(conv.conversation_id) is None

    def test_missing_returns_none(self, conversations):
        assert conversations.next_speaker("nope") is None


# ---- render_context -----------------------------------------------------


class TestRenderContext:
    def test_formats_turns(self, conversations, project):
        conv = conversations.start(project.project_id, "hello")
        conversations.append_system_turn(
            conv.conversation_id, f"{CLARIFICATION_MARKER} what lib?"
        )
        conversations.append_user_turn(conv.conversation_id, "mistune")
        rendered = conversations.render_context(conv.conversation_id)
        assert "user: hello" in rendered
        assert "system:" in rendered
        assert "what lib?" in rendered
        assert "mistune" in rendered

    def test_respects_limit(self, conversations, project):
        conv = conversations.start(project.project_id, "first")
        for i in range(5):
            conversations.append_system_turn(conv.conversation_id, f"reply {i}")
            conversations.append_user_turn(conv.conversation_id, f"followup {i}")
        rendered = conversations.render_context(conv.conversation_id, limit_turns=3)
        # Only 3 lines.
        assert rendered.count("\n") == 2

    def test_empty_conversation_renders_empty(self, conversations, project, storage):
        # Create a conversation via storage directly so there are no turns.
        storage.create_conversation(conversation_id="bare", project_id=project.project_id)
        assert conversations.render_context("bare") == ""

    def test_multiline_content_indented(self, conversations, project):
        conv = conversations.start(project.project_id, "line1\nline2")
        rendered = conversations.render_context(conv.conversation_id)
        # Secondary line should be indented so "user:" prefix is not ambiguous.
        assert "user: line1" in rendered
        assert "  line2" in rendered


# ---- close --------------------------------------------------------------


class TestClose:
    def test_close_sets_status(self, conversations, project):
        conv = conversations.start(project.project_id, "hi")
        conversations.close(conv.conversation_id)
        assert conversations.get(conv.conversation_id).status == STATUS_CLOSED
