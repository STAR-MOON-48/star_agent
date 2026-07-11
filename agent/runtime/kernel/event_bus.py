from __future__ import annotations

import asyncio
from typing import Optional

from ...protocols import AgentEvent
from ..console import trace_line


class EventBus:
    """In-memory event bus for the MVP.

    Production systems can replace this with Redis Streams, Kafka, NATS, SQS,
    Postgres LISTEN/NOTIFY, Temporal signals, etc. The runtime only depends on
    publish() and next_event().
    """

    def __init__(self, *, trace: bool = True) -> None:
        self._queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self.trace = trace

    async def publish(self, event: AgentEvent) -> None:
        if self.trace:
            extra = f"id=[cyan]{event.event_id}[/cyan] source=[cyan]{event.source}[/cyan]"
            if event.task_id:
                extra += f" task=[magenta]{event.task_id}[/magenta]"
            if event.action_run_id:
                extra += f" run=[magenta]{event.action_run_id}[/magenta]"
            trace_line(
                "event.bus",
                f"publish type=[cyan]{event.type}[/cyan] {extra}",
            )
        await self._queue.put(event)

    async def next_event(self, timeout: Optional[float] = None) -> AgentEvent:
        if timeout is None:
            return await self._queue.get()
        return await asyncio.wait_for(self._queue.get(), timeout=timeout)

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()
