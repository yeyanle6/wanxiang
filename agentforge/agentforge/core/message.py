from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class MessageStatus(str, Enum):
    SUCCESS = "success"
    NEEDS_REVISION = "needs_revision"
    ERROR = "error"


@dataclass(slots=True)
class Message:
    intent: str
    content: str
    sender: str
    status: MessageStatus = MessageStatus.SUCCESS
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    context: list[str] = field(default_factory=list)
    turn: int = 1
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def create_reply(
        self,
        *,
        intent: str,
        content: str,
        sender: str,
        status: MessageStatus,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Create a reply message linked to the current message."""
        next_context = [*self.context, self.content]
        return Message(
            intent=intent,
            content=content,
            sender=sender,
            status=status,
            metadata={} if metadata is None else metadata,
            parent_id=self.id,
            context=next_context,
            turn=self.turn + 1,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize message for logging and storage."""
        return {
            "id": self.id,
            "intent": self.intent,
            "content": self.content,
            "sender": self.sender,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "parent_id": self.parent_id,
            "context": list(self.context),
            "turn": self.turn,
            "metadata": dict(self.metadata),
        }

    def to_prompt(self) -> str:
        """Render a message into an LLM-friendly handoff format."""
        context_text = (
            "\n".join(f"{idx}. {item}" for idx, item in enumerate(self.context, start=1))
            if self.context
            else "(empty)"
        )
        return (
            f"Intent:\n{self.intent}\n\n"
            f"Content:\n{self.content}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Envelope:\n"
            f"- sender: {self.sender}\n"
            f"- status: {self.status.value}\n"
            f"- turn: {self.turn}\n"
            f"- message_id: {self.id}\n"
            f"- parent_id: {self.parent_id}\n"
            f"- timestamp: {self.timestamp.isoformat()}"
        )

