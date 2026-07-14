from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from agent.protocols import AgentEvent, utc_now


ControlTarget = Literal["decision", "dmn", "note"]
_TARGETS = {"decision", "dmn", "note"}


@dataclass(frozen=True)
class ControlDirective:
    directive_id: str
    agent_id: str
    target: ControlTarget
    text: str
    created_at: str
    source: str = "local_operator"

    def __post_init__(self) -> None:
        normalized = " ".join(self.text.split())
        if not normalized:
            raise ValueError("Control directive text must not be empty")
        if self.target not in _TARGETS:
            raise ValueError(f"Unsupported control target: {self.target}")
        object.__setattr__(self, "text", normalized)

    def to_event(self) -> AgentEvent:
        event_type = {
            "decision": "operator.directive",
            "dmn": "reflection.requested",
            "note": "operator.note",
        }[self.target]
        return AgentEvent(
            event_id=f"evt_control_{self.directive_id}",
            agent_id=self.agent_id,
            type=event_type,
            source="operator.control",
            payload={
                "content": self.text,
                "control_target": self.target,
                "directive_id": self.directive_id,
                "operator_requested": True,
            },
            priority=5 if self.target == "decision" else 70,
        )


@dataclass(frozen=True)
class ClaimedDirective:
    directive: ControlDirective
    path: Path


class ControlInbox:
    """Small durable filesystem inbox for cross-process internal steering."""

    def __init__(self, root: str | Path, agent_id: str) -> None:
        safe_agent_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", agent_id).strip("._")
        if not safe_agent_id:
            raise ValueError("agent_id has no safe filesystem representation")
        self.agent_id = agent_id
        self.root = Path(root) / "control" / safe_agent_id
        self.inbox = self.root / "inbox"
        self.processing = self.root / "processing"
        self.processed = self.root / "processed"
        self.rejected = self.root / "rejected"
        for directory in (self.inbox, self.processing, self.processed, self.rejected):
            directory.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        *,
        target: ControlTarget,
        text: str,
        source: str = "local_operator",
    ) -> ControlDirective:
        directive = ControlDirective(
            directive_id=uuid4().hex,
            agent_id=self.agent_id,
            target=target,
            text=text,
            created_at=utc_now(),
            source=source,
        )
        path = self.inbox / f"{directive.created_at.replace(':', '-')}_{directive.directive_id}.json"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(asdict(directive), file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
        return directive

    def recover(self) -> int:
        recovered = 0
        for path in sorted(self.processing.glob("*.json")):
            destination = self.inbox / path.name
            if destination.exists():
                destination = self.inbox / f"recovered_{uuid4().hex}_{path.name}"
            os.replace(path, destination)
            recovered += 1
        return recovered

    def claim(self, *, limit: int = 20) -> list[ClaimedDirective]:
        claimed: list[ClaimedDirective] = []
        for path in sorted(self.inbox.glob("*.json"))[: max(1, limit)]:
            processing_path = self.processing / path.name
            try:
                os.replace(path, processing_path)
                data = json.loads(processing_path.read_text(encoding="utf-8"))
                directive = ControlDirective(
                    directive_id=str(data["directive_id"]),
                    agent_id=str(data["agent_id"]),
                    target=str(data["target"]),  # type: ignore[arg-type]
                    text=str(data["text"]),
                    created_at=str(data["created_at"]),
                    source=str(data.get("source") or "local_operator"),
                )
                if directive.agent_id != self.agent_id:
                    raise ValueError("Directive agent_id does not match this inbox")
            except Exception:
                if processing_path.exists():
                    os.replace(processing_path, self.rejected / processing_path.name)
                continue
            claimed.append(ClaimedDirective(directive=directive, path=processing_path))
        return claimed

    def acknowledge(self, item: ClaimedDirective) -> Path:
        destination = self.processed / item.path.name
        os.replace(item.path, destination)
        return destination

    def is_processed(self, directive_id: str) -> bool:
        return any(self.processed.glob(f"*_{directive_id}.json"))
