"""conversation — multi-turn dialogue manager.

Facade over storage's conversation tables. Owns:
  - conversation lifecycle (open / awaiting_user / closed)
  - turn appending with correct seq numbering
  - prompt-ready context rendering for injection into agent system prompts
  - NEEDS_CLARIFICATION marker detection for the dialogue bypass path

Agents speak by emitting a normal response. When that response starts
with the literal marker `NEEDS_CLARIFICATION:`, the ConversationManager
transitions the conversation to awaiting_user instead of closing it,
so the next user message resumes without starting a fresh conversation.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .storage import ConversationRecord, ConversationTurn, Storage

logger = logging.getLogger("wanxiang.conversation")


STATUS_OPEN = "open"
STATUS_AWAITING_USER = "awaiting_user"
STATUS_CLOSED = "closed"

ALL_STATUSES = frozenset({STATUS_OPEN, STATUS_AWAITING_USER, STATUS_CLOSED})

SPEAKER_USER = "user"
SPEAKER_SYSTEM = "system"

CLARIFICATION_MARKER = "NEEDS_CLARIFICATION:"


class ConversationManager:
    """Thin stateful wrapper over storage.

    Instances are cheap — construct one per request/handler. The storage
    object is shared; this class doesn't cache state.
    """

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    # ---- Lifecycle ----------------------------------------------------------

    def start(
        self,
        project_id: str,
        initial_user_message: str,
        *,
        conversation_id: str | None = None,
    ) -> ConversationRecord:
        """Create a conversation + append the first user turn."""
        cleaned = (initial_user_message or "").strip()
        if not cleaned:
            raise ValueError("Initial user message cannot be empty.")

        cid = conversation_id or str(uuid4())
        record = self.storage.create_conversation(
            conversation_id=cid, project_id=project_id, status=STATUS_OPEN
        )
        self.storage.append_conversation_turn(
            conversation_id=cid, speaker=SPEAKER_USER, content=cleaned
        )
        logger.info("Started conversation %s for project %s", cid, project_id)
        return record

    def close(self, conversation_id: str) -> None:
        self.storage.update_conversation_status(conversation_id, STATUS_CLOSED)

    # ---- Turn I/O -----------------------------------------------------------

    def append_user_turn(
        self, conversation_id: str, content: str
    ) -> ConversationTurn:
        """Record a user message. If the conversation was awaiting_user, reopen it."""
        cleaned = (content or "").strip()
        if not cleaned:
            raise ValueError("User message cannot be empty.")
        conv = self.storage.get_conversation(conversation_id)
        if conv is None:
            raise ValueError(f"Conversation not found: {conversation_id}")
        if conv.status == STATUS_CLOSED:
            raise ValueError(f"Cannot append to closed conversation: {conversation_id}")

        turn = self.storage.append_conversation_turn(
            conversation_id=conversation_id,
            speaker=SPEAKER_USER,
            content=cleaned,
        )
        if conv.status == STATUS_AWAITING_USER:
            self.storage.update_conversation_status(conversation_id, STATUS_OPEN)
        return turn

    def append_system_turn(
        self,
        conversation_id: str,
        content: str,
        *,
        run_id: str | None = None,
        force_status: str | None = None,
    ) -> ConversationTurn:
        """Record a system response and transition status appropriately.

        - If content starts with NEEDS_CLARIFICATION: → awaiting_user
        - Otherwise → open (ready for next user input, caller may close)
        - `force_status` overrides the above (tests / explicit control)
        """
        cleaned = content or ""
        conv = self.storage.get_conversation(conversation_id)
        if conv is None:
            raise ValueError(f"Conversation not found: {conversation_id}")

        turn = self.storage.append_conversation_turn(
            conversation_id=conversation_id,
            speaker=SPEAKER_SYSTEM,
            content=cleaned,
            run_id=run_id,
        )

        if force_status is not None:
            if force_status not in ALL_STATUSES:
                raise ValueError(f"Invalid status: {force_status}")
            new_status = force_status
        elif self.is_clarification(cleaned):
            new_status = STATUS_AWAITING_USER
        else:
            new_status = STATUS_OPEN

        if new_status != conv.status:
            self.storage.update_conversation_status(conversation_id, new_status)
        return turn

    # ---- Queries ------------------------------------------------------------

    def get(self, conversation_id: str) -> ConversationRecord | None:
        return self.storage.get_conversation(conversation_id)

    def turns(
        self, conversation_id: str, *, limit: int = 500
    ) -> list[ConversationTurn]:
        return self.storage.list_conversation_turns(conversation_id, limit=limit)

    def next_speaker(self, conversation_id: str) -> str | None:
        """'user' | 'system' | None (if closed / missing).

        - open: last turn is user → system next; last turn is system → user next
        - awaiting_user: user next
        - closed: None
        """
        conv = self.storage.get_conversation(conversation_id)
        if conv is None or conv.status == STATUS_CLOSED:
            return None
        if conv.status == STATUS_AWAITING_USER:
            return SPEAKER_USER
        turns = self.storage.list_conversation_turns(conversation_id, limit=1_000)
        if not turns:
            return SPEAKER_USER
        return SPEAKER_SYSTEM if turns[-1].speaker == SPEAKER_USER else SPEAKER_USER

    # ---- Prompt rendering ---------------------------------------------------

    def render_context(
        self, conversation_id: str, *, limit_turns: int = 20
    ) -> str:
        """Format prior turns for injection into an agent system prompt.

        Returns a plain-text block like:
          user: hello
          system: NEEDS_CLARIFICATION: what library?
          user: mistune

        Empty string for conversations with no turns.
        """
        turns = self.storage.list_conversation_turns(
            conversation_id, limit=max(1, int(limit_turns))
        )
        if not turns:
            return ""
        lines = []
        for turn in turns:
            content = turn.content.strip().replace("\n", "\n  ")
            lines.append(f"{turn.speaker}: {content}")
        return "\n".join(lines)

    # ---- Markers ------------------------------------------------------------

    @staticmethod
    def is_clarification(content: str) -> bool:
        return bool(content) and content.lstrip().startswith(CLARIFICATION_MARKER)

    @staticmethod
    def strip_marker(content: str) -> str:
        if not ConversationManager.is_clarification(content):
            return content
        stripped = content.lstrip()
        return stripped[len(CLARIFICATION_MARKER):].lstrip()
