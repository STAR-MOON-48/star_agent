from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from star_protocol.client import AgentClient
from star_protocol.models import Envelope, MessagePayload, gen_id

from ...protocols import ActionSpec, AgentEvent, JsonDict
from ..perception_systems import PerceptionSystem
from .protocol import ProtocolInterface


@dataclass
class _PendingAction:
    agent_id: str
    task_id: str
    action_run_id: str
    action_name: str
    causation_id: Optional[str] = None


class _StarAgentClient(AgentClient):
    def __init__(
        self,
        session: "StarSession",
        *,
        client_id: str,
        auto_reconnect: bool,
        monitorable: bool,
        logger: Optional[logging.Logger],
    ) -> None:
        super().__init__(
            client_id=client_id,
            auto_reconnect=auto_reconnect,
            monitorable=monitorable,
            logger=logger,
        )
        self._session = session

    async def on_tool_specification(self, sender: str, tools: list[dict[str, Any]]) -> None:
        await self._session.handle_tool_specification(sender, tools)

    async def on_outcome(self, sender: str, content: dict[str, Any]) -> None:
        await self._session.handle_outcome(sender, content)

    async def on_action(self, sender: str, content: dict[str, Any]) -> None:
        await self._session.handle_action(sender, content)

    async def on_event(self, sender: str, content: dict[str, Any]) -> None:
        await self._session.handle_event(sender, content)

    async def on_stream(self, sender: str, content: dict[str, Any]) -> None:
        await self._session.handle_stream(sender, content)

    async def on_broadcast_event(self, sender: str, content: dict[str, Any]) -> None:
        await self._session.handle_event(sender, content, broadcast=True)

    async def on_broadcast_stream(self, sender: str, content: dict[str, Any]) -> None:
        await self._session.handle_stream(sender, content, broadcast=True)


class StarSession(ProtocolInterface):
    """Star Protocol-backed communication interface."""

    def __init__(
        self,
        *,
        hub_url: str,
        client_id: Optional[str] = None,
        env_id: Optional[str] = None,
        join_timeout: float = 30.0,
        retry_interval: float = 2.0,
        auto_reconnect: bool = True,
        auto_rejoin: bool = True,
        monitorable: bool = False,
        perception_system: Optional[PerceptionSystem] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.hub_url = hub_url
        self.client_id = client_id
        self.env_id = env_id
        self.join_timeout = join_timeout
        self.retry_interval = retry_interval
        self.auto_reconnect = auto_reconnect
        self.auto_rejoin = auto_rejoin
        self.monitorable = monitorable
        self.perception_system = perception_system or PerceptionSystem()
        self.logger = logger

        self._agent_id: Optional[str] = None
        self._client: Optional[_StarAgentClient] = None
        self._events: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._action_specs: dict[str, ActionSpec] = {}
        self._pending: dict[str, _PendingAction] = {}

    async def start(self, *, agent_id: str) -> None:
        if self._client is not None:
            return

        self._agent_id = agent_id
        self._client = _StarAgentClient(
            self,
            client_id=self.client_id or agent_id,
            auto_reconnect=self.auto_reconnect,
            monitorable=self.monitorable,
            logger=self.logger,
        )
        await self._client.connect(self.hub_url)
        await self._client.start()

        if self.auto_rejoin:
            self._client.enable_auto_rejoin(
                timeout=max(self.join_timeout, 0.0),
                retry_interval=self.retry_interval,
            )

        if self.env_id:
            await self._client.join_with_retry(
                self.env_id,
                timeout=self.join_timeout,
                retry_interval=self.retry_interval,
            )

    async def stop(self) -> None:
        if self._client is None:
            return
        await self._client.stop()
        self._client = None

    def list_action_specs(self) -> Iterable[ActionSpec]:
        return list(self._action_specs.values())

    async def rediscover(self) -> None:
        if self._client is None:
            raise RuntimeError("StarSession is not started.")
        await self._client.rediscover()

    async def send_action(
        self,
        *,
        agent_id: str,
        task_id: str,
        action_run_id: str,
        action_name: str,
        args: JsonDict,
        target: Optional[str] = None,
        causation_id: Optional[str] = None,
    ) -> str:
        if self._client is None:
            raise RuntimeError("StarSession is not started.")

        recipient = target or self.env_id or self._client.current_env
        if not recipient:
            raise RuntimeError("No Star Protocol environment target is available.")

        external_action_id = gen_id("action", action_name)
        self._pending[external_action_id] = _PendingAction(
            agent_id=agent_id,
            task_id=task_id,
            action_run_id=action_run_id,
            action_name=action_name,
            causation_id=causation_id,
        )
        try:
            await self._client.send(
                Envelope(
                    type="message",
                    sender=self._client.client_id,
                    recipient=recipient,
                    payload=MessagePayload(
                        type="action",
                        content={
                            "id": external_action_id,
                            "name": action_name,
                            "params": args,
                        },
                    ),
                )
            )
        except Exception:
            self._pending.pop(external_action_id, None)
            raise
        return external_action_id

    async def send_event(
        self,
        *,
        agent_id: str,
        recipient: str,
        event_name: str,
        data: JsonDict,
        causation_id: Optional[str] = None,
    ) -> str:
        if self._client is None:
            raise RuntimeError("StarSession is not started.")

        event_id = gen_id("event", event_name)
        await self._client.send(
            Envelope(
                type="message",
                sender=self._client.client_id,
                recipient=recipient,
                payload=MessagePayload(
                    type="event",
                    content={
                        "id": event_id,
                        "name": event_name,
                        "data": {
                            **data,
                            "agent_id": agent_id,
                            "causation_id": causation_id,
                        },
                    },
                ),
            )
        )
        return event_id

    async def next_event(self, timeout: Optional[float] = None) -> AgentEvent:
        return await self._next_from_queue(self._events, timeout)

    async def handle_tool_specification(
        self, sender: str, tools: list[dict[str, Any]]
    ) -> None:
        for tool in tools:
            spec = self._tool_to_action_spec(sender, tool)
            if spec is not None:
                self._action_specs[spec.name] = spec

        await self._queue_protocol_event(
            self.perception_system.star_tool_specification(
                agent_id=self._require_agent_id(),
                sender=sender,
                tools=tools,
            )
        )

    async def handle_outcome(self, sender: str, content: dict[str, Any]) -> None:
        ref_id = str(content.get("ref_id") or content.get("id") or "")
        pending = self._pending.pop(ref_id, None) if ref_id else None
        if pending is None:
            await self._queue_protocol_event(
                self.perception_system.star_outcome(
                    agent_id=self._require_agent_id(),
                    sender=sender,
                    content=content,
                )
            )
            return

        success = bool(content.get("success", True))
        if success:
            await self._events.put(
                self.perception_system.star_action_completed(
                    agent_id=pending.agent_id,
                    task_id=pending.task_id,
                    action_run_id=pending.action_run_id,
                    action_name=pending.action_name,
                    external_action_id=ref_id,
                    sender=sender,
                    result=content.get("data", content),
                    causation_id=pending.causation_id,
                )
            )
            return

        await self._events.put(
            self.perception_system.star_action_failed(
                agent_id=pending.agent_id,
                task_id=pending.task_id,
                action_run_id=pending.action_run_id,
                action_name=pending.action_name,
                external_action_id=ref_id,
                sender=sender,
                error=content.get("error", content),
                causation_id=pending.causation_id,
            )
        )

    async def handle_action(self, sender: str, content: dict[str, Any]) -> None:
        event = self.perception_system.star_action(
            agent_id=self._require_agent_id(),
            sender=sender,
            content=content,
        )
        await self._queue_protocol_event(event)
        if event.type == "user.message":
            await self._acknowledge_user_message_action(
                sender=sender,
                content=content,
                event=event,
            )

    async def handle_event(
        self,
        sender: str,
        content: dict[str, Any],
        *,
        broadcast: bool = False,
    ) -> None:
        await self._queue_protocol_event(
            self.perception_system.star_event(
                agent_id=self._require_agent_id(),
                sender=sender,
                content=content,
                broadcast=broadcast,
            )
        )

    async def handle_stream(
        self,
        sender: str,
        content: dict[str, Any],
        *,
        broadcast: bool = False,
    ) -> None:
        ref_id = str(content.get("ref_id") or content.get("id") or "")
        pending = self._pending.get(ref_id) if ref_id else None
        if pending is None:
            await self._queue_protocol_event(
                self.perception_system.star_stream(
                    agent_id=self._require_agent_id(),
                    sender=sender,
                    content=content,
                    broadcast=broadcast,
                )
            )
            return

        await self._events.put(
            self.perception_system.star_action_progress(
                agent_id=pending.agent_id,
                task_id=pending.task_id,
                action_run_id=pending.action_run_id,
                action_name=pending.action_name,
                external_action_id=ref_id,
                sender=sender,
                content=content,
                causation_id=pending.causation_id,
            )
        )

    def _tool_to_action_spec(
        self, sender: str, tool: dict[str, Any]
    ) -> Optional[ActionSpec]:
        name = tool.get("name")
        if not name:
            return None
        return ActionSpec(
            name=str(name),
            description=str(tool.get("description", "")),
            input_schema=dict(tool.get("parameters") or tool.get("input_schema") or {}),
            mode="async",
            timeout_ms=int(tool.get("timeout_ms", 60_000)),
            cancelable=False,
            requires_approval=bool(tool.get("requires_approval", False)),
            side_effect_level=str(tool.get("side_effect_level", "external_effect")),
            source="star_protocol",
            target=sender,
            metadata={
                "tags": list(tool.get("tags") or []),
                "raw_tool": dict(tool),
            },
        )

    def _require_agent_id(self) -> str:
        if self._agent_id is None:
            raise RuntimeError("StarSession is not bound to an agent_id.")
        return self._agent_id

    async def _queue_protocol_event(self, event: AgentEvent) -> None:
        await self._events.put(event)

    async def _acknowledge_user_message_action(
        self,
        *,
        sender: str,
        content: dict[str, Any],
        event: AgentEvent,
    ) -> None:
        action_id = str(content.get("id") or "")
        if not action_id or self._client is None:
            return
        try:
            await self._client.send_outcome(
                sender,
                {
                    "ref_id": action_id,
                    "success": True,
                    "data": {
                        "accepted": True,
                        "event_id": event.event_id,
                        "message_id": event.payload.get("message_id"),
                    },
                },
            )
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning(
                    "Failed to acknowledge user_message action %s: %s",
                    action_id,
                    exc,
                )
