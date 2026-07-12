from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MessagePurpose(StrEnum):
    """Why one subsystem is handing natural language to another."""

    UNDERSTANDING = "understanding"
    EXPRESSION = "expression"
    DECISION = "decision"
    REFLECTION = "reflection"
    MEMORY = "memory"


@dataclass(frozen=True)
class NaturalMessage:
    """The only message exchanged by refactored cognitive subsystems.

    Routing and durable references remain typed metadata.  The actual handoff
    is always ``text`` written as ordinary language, never a JSON command or a
    subsystem-specific object graph.
    """

    sender: str
    recipient: str
    purpose: MessagePurpose
    text: str
    event_id: str | None = None
    conversation_id: str | None = None
    turn_id: str | None = None
    task_id: str | None = None
    references: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized = " ".join(self.text.split())
        if not normalized:
            raise ValueError("NaturalMessage.text must not be empty")
        object.__setattr__(self, "text", normalized)

    def audit_record(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "purpose": self.purpose.value,
            "text": self.text,
            "event_id": self.event_id,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "task_id": self.task_id,
            "references": list(self.references),
        }
