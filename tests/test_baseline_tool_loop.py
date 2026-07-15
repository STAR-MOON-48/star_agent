from __future__ import annotations

import asyncio
from io import StringIO
from types import SimpleNamespace
from typing import Any
import unittest

from rich.console import Console

from agent.protocols import ActionSpec, AgentEvent
from agent.runtime.interfaces import ModelInterface, ModelResult, ProtocolInterface
from baseline_agent import ToolLoopAgent
from baseline_agent.rich_output import RichTraceRenderer


class _ScriptedModel(ModelInterface):
    def __init__(self, results: list[ModelResult]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages: Any, **kwargs: Any) -> ModelResult:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.results.pop(0)


class _FakeProtocol(ProtocolInterface):
    def __init__(
        self,
        specs: list[ActionSpec] | None = None,
        *,
        action_results: list[dict[str, object]] | None = None,
    ) -> None:
        self.specs = specs or []
        self.action_results = list(action_results or [])
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
        result = (
            self.action_results.pop(0)
            if self.action_results
            else {"temperature": 21}
        )
        await self.events.put(
            AgentEvent.make(
                agent_id=str(kwargs["agent_id"]),
                type="action.completed",
                source="star_protocol",
                task_id=str(kwargs["task_id"]),
                action_run_id=str(kwargs["action_run_id"]),
                payload={"result": result},
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
    def tool_call_result(self, call_id: str) -> ModelResult:
        return ModelResult(
            text="",
            raw=SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        id=call_id,
                        name="observe_room",
                        arguments={"focus": call_id},
                    )
                ]
            ),
        )

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

    async def test_old_rounds_are_summarized_while_newest_round_stays_verbatim(
        self,
    ) -> None:
        old_marker = "OLD_OBSERVATION_" + "a" * 180
        recent_marker = "RECENT_OBSERVATION_" + "b" * 180
        model = _ScriptedModel(
            [
                self.tool_call_result("call-old"),
                self.tool_call_result("call-recent"),
                ModelResult(
                    text=(
                        "## Objective and current status\nContinue observing.\n"
                        "## Verified facts\nThe old observation was retained."
                    )
                ),
                ModelResult(text="Compaction preserved continuity."),
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
            ],
            action_results=[
                {"observation": old_marker},
                {"observation": recent_marker},
            ],
        )
        trace_events: list[tuple[str, dict[str, Any]]] = []
        agent = ToolLoopAgent(
            agent_id="baseline",
            model=model,
            protocol=protocol,
            system_prompt="Use tools and finish the objective.",
            max_tokens=100,
            context_window_tokens=1500,
            context_safety_margin_tokens=50,
            compaction_trigger_ratio=0.7,
            compaction_target_ratio=0.3,
            keep_recent_rounds=1,
            summary_max_tokens=80,
            chars_per_token=1.0,
            trace=lambda event, data: trace_events.append((event, data)),
        )

        await agent.start()
        try:
            answer = await agent.run("Observe until enough evidence is available.")
        finally:
            await agent.stop()

        self.assertEqual(answer, "Compaction preserved continuity.")
        self.assertEqual(len(model.calls), 4)
        summary_call = model.calls[2]
        self.assertNotIn("tools", summary_call["kwargs"])
        final_context = model.calls[3]["messages"]
        final_contents = "\n".join(
            str(getattr(message, "content", ""))
            for message in final_context.messages
        )
        self.assertIn("# Compacted execution history", final_contents)
        self.assertIn("The old observation was retained.", final_contents)
        self.assertNotIn(old_marker, final_contents)
        self.assertIn(recent_marker, final_contents)
        completed = [
            data
            for event, data in trace_events
            if event == "context.compaction.completed"
        ]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["archived_rounds"], 1)
        self.assertEqual(completed[0]["retained_verbatim_rounds"], 1)
        self.assertLess(
            completed[0]["after_estimated_input_tokens"],
            completed[0]["before_estimated_input_tokens"],
        )

    def test_rich_renderer_has_clear_context_tool_and_answer_sections(self) -> None:
        output = StringIO()
        renderer = RichTraceRenderer(
            Console(file=output, force_terminal=False, width=100)
        )
        renderer(
            "context.usage",
            {
                "step": 3,
                "estimated_input_tokens": 700,
                "available_input_tokens": 1000,
                "remaining_input_tokens": 300,
                "trigger_tokens": 800,
                "verbatim_rounds": 2,
                "compaction_count": 1,
            },
        )
        renderer(
            "context.compaction.completed",
            {
                "before_estimated_input_tokens": 900,
                "after_estimated_input_tokens": 350,
                "compression_ratio": 0.6111,
                "summary_estimated_tokens": 80,
                "summary_source": "model",
                "archived_rounds": 5,
                "retained_verbatim_rounds": 2,
            },
        )
        renderer(
            "tool.completed",
            {"name": "observe_room", "success": True, "result": {"ok": True}},
        )
        renderer(
            "request.completed",
            {"steps": 3, "answer": "Finished."},
        )

        rendered = output.getvalue()
        self.assertIn("Context Budget", rendered)
        self.assertIn("Context Compacted", rendered)
        self.assertIn("Tool ✓ observe_room", rendered)
        self.assertIn("Final Answer", rendered)


if __name__ == "__main__":
    unittest.main()
