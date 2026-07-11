from __future__ import annotations

from hashlib import sha256
import json
from typing import TYPE_CHECKING, Any, Optional

from ...protocols import (
    AgentEvent,
    JsonDict,
    ensure_json_dict,
    ensure_json_dict_list,
)

if TYPE_CHECKING:
    from ..kernel.event_bus import EventBus


class PerceptionSystem:
    """Normalizes external/internal stimuli into runtime AgentEvents."""

    USER_MESSAGE_NAMES = {
        "user.message",
        "user_message",
        "message",
        "chat",
        "chat_message",
        "submit_user_message",
    }

    def __init__(self, *, event_bus: Optional["EventBus"] = None) -> None:
        self.event_bus = event_bus

    def bind_event_bus(self, event_bus: "EventBus") -> None:
        self.event_bus = event_bus

    async def publish(self, event: AgentEvent) -> AgentEvent:
        if self.event_bus is None:
            raise RuntimeError("PerceptionSystem is not bound to an EventBus.")
        await self.event_bus.publish(event)
        return event

    async def perceive_local_user_message(self, *, agent_id: str, content: str) -> AgentEvent:
        return await self.publish(
            self.local_user_message(
                agent_id=agent_id,
                content=content,
            )
        )

    def local_user_message(self, *, agent_id: str, content: str) -> AgentEvent:
        return AgentEvent.make(
            agent_id=agent_id,
            type="user.message",
            source="user",
            payload={"content": content},
            priority=10,
        )

    def star_tool_specification(
        self,
        *,
        agent_id: str,
        sender: str,
        tools: list[dict[str, Any]],
    ) -> AgentEvent:
        return self._protocol_event(
            agent_id=agent_id,
            event_type="protocol.tool_specification",
            payload={"sender": sender, "tools": ensure_json_dict_list(tools)},
        )

    def star_outcome(
        self,
        *,
        agent_id: str,
        sender: str,
        content: JsonDict,
    ) -> AgentEvent:
        content = ensure_json_dict(content)
        return self._protocol_event(
            agent_id=agent_id,
            event_type="protocol.outcome",
            payload={"sender": sender, "content": content},
        )

    def star_action(
        self,
        *,
        agent_id: str,
        sender: str,
        content: JsonDict,
    ) -> AgentEvent:
        content = ensure_json_dict(content)
        action_name = str(content.get("name") or "")
        params = ensure_json_dict(content.get("params"))
        if self.is_user_message_name(action_name):
            return self._user_message_from_star(
                agent_id=agent_id,
                sender=sender,
                content=params,
                protocol_payload={"kind": "action", "content": content},
            )
        return self._protocol_event(
            agent_id=agent_id,
            event_type="protocol.action",
            payload={"sender": sender, "content": content},
        )

    def star_event(
        self,
        *,
        agent_id: str,
        sender: str,
        content: JsonDict,
        broadcast: bool = False,
    ) -> AgentEvent:
        content = ensure_json_dict(content)
        event_name = str(content.get("name") or content.get("type") or "")
        if self.is_user_message_name(event_name):
            data = content.get("data")
            return self._user_message_from_star(
                agent_id=agent_id,
                sender=sender,
                content=dict(data) if isinstance(data, dict) else content,
                protocol_payload={
                    "kind": "event",
                    "content": content,
                    "broadcast": broadcast,
                },
            )
        return self._protocol_event(
            agent_id=agent_id,
            event_type="protocol.event",
            payload={"sender": sender, "content": content, "broadcast": broadcast},
        )

    def star_stream(
        self,
        *,
        agent_id: str,
        sender: str,
        content: JsonDict,
        broadcast: bool = False,
    ) -> AgentEvent:
        content = ensure_json_dict(content)
        return self._protocol_event(
            agent_id=agent_id,
            event_type="protocol.stream",
            payload={"sender": sender, "content": content, "broadcast": broadcast},
        )

    def star_action_completed(
        self,
        *,
        agent_id: str,
        task_id: str,
        action_run_id: str,
        action_name: str,
        external_action_id: str,
        sender: str,
        result: JsonDict,
        causation_id: Optional[str] = None,
    ) -> AgentEvent:
        return AgentEvent.make(
            agent_id=agent_id,
            type="action.completed",
            source="star_protocol",
            task_id=task_id,
            action_run_id=action_run_id,
            causation_id=causation_id,
            payload={
                "action_name": action_name,
                "external_action_id": external_action_id,
                "sender": sender,
                "result": ensure_json_dict(result),
            },
        )

    def star_action_failed(
        self,
        *,
        agent_id: str,
        task_id: str,
        action_run_id: str,
        action_name: str,
        external_action_id: str,
        sender: str,
        error: Any,
        causation_id: Optional[str] = None,
    ) -> AgentEvent:
        return AgentEvent.make(
            agent_id=agent_id,
            type="action.failed",
            source="star_protocol",
            task_id=task_id,
            action_run_id=action_run_id,
            causation_id=causation_id,
            payload={
                "action_name": action_name,
                "external_action_id": external_action_id,
                "sender": sender,
                "error": {"message": str(error)},
            },
        )

    def star_action_progress(
        self,
        *,
        agent_id: str,
        task_id: str,
        action_run_id: str,
        action_name: str,
        external_action_id: str,
        sender: str,
        content: JsonDict,
        causation_id: Optional[str] = None,
    ) -> AgentEvent:
        content = ensure_json_dict(content)
        return AgentEvent.make(
            agent_id=agent_id,
            type="action.progress",
            source="star_protocol",
            task_id=task_id,
            action_run_id=action_run_id,
            causation_id=causation_id,
            payload={
                "action_name": action_name,
                "external_action_id": external_action_id,
                "sender": sender,
                "progress": {
                    "sequence": content.get("sequence"),
                    "chunk": content.get("chunk"),
                    "is_end": content.get("is_end", False),
                },
            },
        )

    def is_user_message_name(self, name: str) -> bool:
        normalized = name.strip().lower().replace("-", "_")
        return normalized in self.USER_MESSAGE_NAMES

    def _user_message_from_star(
        self,
        *,
        agent_id: str,
        sender: str,
        content: JsonDict,
        protocol_payload: JsonDict,
    ) -> AgentEvent:
        content = ensure_json_dict(content)
        protocol_payload = ensure_json_dict(protocol_payload)
        protocol_content = protocol_payload.get("content")
        transport_message_id = (
            protocol_content.get("id")
            if isinstance(protocol_content, dict)
            else None
        )
        message_id = (
            content.get("message_id")
            or content.get("id")
            or transport_message_id
        )
        event = AgentEvent.make(
            agent_id=agent_id,
            type="user.message",
            source="star_protocol",
            payload={
                "content": self._extract_message_text(content),
                "sender": sender,
                "conversation_id": (
                    content.get("conversation_id")
                    or content.get("chat_id")
                    or content.get("session_id")
                ),
                "message_id": message_id,
                "speaker_context": (
                    content.get("speaker_context")
                    or content.get("human_profile")
                    or content.get("profile")
                    or {}
                ),
                "scene_context": content.get("scene_context") or {},
                "protocol": protocol_payload,
            },
            idempotency_key=(
                f"star_user_message:{sender}:{message_id}"
                if message_id
                else None
            ),
            priority=10,
        )
        if message_id:
            event.event_id = self._stable_user_message_event_id(
                agent_id=agent_id,
                sender=sender,
                message_id=str(message_id),
            )
        return event

    def _stable_user_message_event_id(
        self,
        *,
        agent_id: str,
        sender: str,
        message_id: str,
    ) -> str:
        canonical = json.dumps(
            [agent_id, sender, message_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = sha256(canonical.encode("utf-8")).hexdigest()[:20]
        return f"evt_star_user_{digest}"

    def _extract_message_text(self, content: JsonDict) -> str:
        for key in ("content", "message", "text", "msg", "prompt"):
            value = content.get(key)
            if value is not None:
                return str(value)
        return str(content)

    def _protocol_event(
        self,
        *,
        agent_id: str,
        event_type: str,
        payload: JsonDict,
    ) -> AgentEvent:
        return AgentEvent.make(
            agent_id=agent_id,
            type=event_type,
            source="star_protocol",
            payload=ensure_json_dict(payload),
        )
