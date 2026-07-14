from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
import unittest

from agent.protocols import ActionSpec, AgentEvent
from agent.runtime.interfaces import ModelInterface, ModelResult, ProtocolInterface
from baseline_agent import ToolLoopAgent


class _ScriptedModel(ModelInterface):
    def __init__(self, results: list[ModelResult]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages: Any, **kwargs: Any) -> ModelResult:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.results.pop(0)


class _FakeProtocol(ProtocolInterface):
    def __init__(self, specs: list[ActionSpec] | None = None) -> None:
        self.specs = specs or []
        self.events: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self.sent_actions: list[dict[str, object]] = []
        self.sent_events: list[dict[str, object]] = []
        self.outbound = asyncio.Event()
        self.started = False

    async def start(self, *, agent_id: str) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    def list_action_specs(self) -> list[ActionSpec]:
        return self.specs

    async def send_action(self, **kwargs: object) -> str:
        self.sent_actions.append(dict(kwargs))
        await self.events.put(
            AgentEvent.make(
                agent_id=str(kwargs["agent_id"]),
                type="action.completed",
                source="star_protocol",
                task_id=str(kwargs["task_id"]),
                action_run_id=str(kwargs["action_run_id"]),
                payload={"result": {"temperature": 21}},
            )
        )
        return "star-action-1"

    async def send_event(self, **kwargs: object) -> str:
        self.sent_events.append(dict(kwargs))
        self.outbound.set()
        return "star-event-1"

    async def next_event(self, timeout: float | None = None) -> AgentEvent:
        if timeout is None:
            return await self.events.get()
        return await asyncio.wait_for(self.events.get(), timeout)


class BaselineToolLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_executes_star_tool_and_returns_final_model_text(self) -> None:
        model = _ScriptedModel(
            [
                ModelResult(
                    text="",
                    raw=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                name="observe_room",
                                arguments={"focus": "temperature"},
                            )
                        ]
                    ),
                ),
                ModelResult(text="Room temperature is 21°C."),
            ]
        )
        protocol = _FakeProtocol(
            [
                ActionSpec(
                    name="observe_room",
                    description="Observe the room.",
                    input_schema={"type": "object"},
                    mode="async",
                    source="star_protocol",
                    target="room-env",
                )
            ]
        )
        agent = ToolLoopAgent(agent_id="baseline", model=model, protocol=protocol)

        await agent.start()
        try:
            answer = await agent.run("What is the room temperature?")
        finally:
            await agent.stop()

        self.assertEqual(answer, "Room temperature is 21°C.")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(protocol.sent_actions), 1)
        action = protocol.sent_actions[0]
        self.assertEqual(action["action_name"], "observe_room")
        self.assertEqual(action["args"], {"focus": "temperature"})
        first_kwargs = model.calls[0]["kwargs"]
        self.assertEqual(first_kwargs["tools"][0]["function"]["name"], "observe_room")

    async def test_star_user_message_gets_assistant_message_reply(self) -> None:
        model = _ScriptedModel([ModelResult(text="Hello from baseline.")])
        protocol = _FakeProtocol()
        agent = ToolLoopAgent(agent_id="baseline", model=model, protocol=protocol)
        await agent.start()
        serve_task = asyncio.create_task(agent.serve())
        try:
            await protocol.events.put(
                AgentEvent.make(
                    agent_id="baseline",
                    type="user.message",
                    source="star_protocol",
                    payload={
                        "content": "Hello",
                        "sender": "human-1",
                        "conversation_id": "chat-1",
                    },
                )
            )
            await asyncio.wait_for(protocol.outbound.wait(), timeout=1.0)
        finally:
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass
            await agent.stop()

        self.assertEqual(len(protocol.sent_events), 1)
        reply = protocol.sent_events[0]
        self.assertEqual(reply["recipient"], "human-1")
        self.assertEqual(reply["event_name"], "assistant.message")
        self.assertEqual(reply["data"]["content"], "Hello from baseline.")
        self.assertEqual(reply["data"]["conversation_id"], "chat-1")


if __name__ == "__main__":
    unittest.main()
