from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from ...protocols import ActionSpec, AgentEvent, JsonDict


class ProtocolInterface(ABC):
    """Boundary for external communication protocols."""

    @abstractmethod
    async def start(self, *, agent_id: str) -> None:
        """Connect and begin receiving protocol messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop receiving protocol messages and close the connection."""

    @abstractmethod
    def list_action_specs(self) -> Iterable[ActionSpec]:
        """Return externally discovered action specs."""

    @abstractmethod
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
        """Send an external action and return the protocol-level action id."""

    @abstractmethod
    async def send_event(
        self,
        *,
        agent_id: str,
        recipient: str,
        event_name: str,
        data: JsonDict,
        causation_id: Optional[str] = None,
    ) -> str:
        """Send an external protocol event and return the protocol-level event id."""

    @abstractmethod
    async def next_event(self, timeout: Optional[float] = None) -> AgentEvent:
        """Return the next protocol event converted to an internal AgentEvent."""

    async def rediscover(self) -> None:
        """Ask the external protocol to refresh tool/action specs."""

    async def _next_from_queue(
        self, queue: asyncio.Queue[AgentEvent], timeout: Optional[float]
    ) -> AgentEvent:
        if timeout is None:
            return await queue.get()
        return await asyncio.wait_for(queue.get(), timeout=timeout)
