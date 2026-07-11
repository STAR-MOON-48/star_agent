from __future__ import annotations

import unittest

from agent.protocols import ActionSpec, AgentEvent, AgentState
from agent.runtime.action_systems.actions import (
    ActionExecutor,
    ActionRegistry,
    action_idempotency_key,
)
from agent.runtime.action_systems.task_system import (
    MULTI_STEP_OBJECTIVE_PURPOSE,
    TaskSystem,
)
from agent.runtime.kernel.event_bus import EventBus


class RecordingProtocol:
    def __init__(self) -> None:
        self.sent_actions: list[dict[str, object]] = []

    async def send_action(self, **kwargs: object) -> str:
        self.sent_actions.append(dict(kwargs))
        return f"external-{len(self.sent_actions)}"


class ActionIdempotencyTests(unittest.IsolatedAsyncioTestCase):
    def test_key_is_independent_of_argument_order(self) -> None:
        first = action_idempotency_key(
            task_id="task-1",
            action_name="write_external_state",
            args={"alpha": 1, "nested": {"x": 2, "y": 3}},
        )
        second = action_idempotency_key(
            task_id="task-1",
            action_name="write_external_state",
            args={"nested": {"y": 3, "x": 2}, "alpha": 1},
        )

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("sha256:"))

    async def test_persisted_success_is_replayed_without_resending(self) -> None:
        registry = ActionRegistry()
        registry.register(
            ActionSpec(
                name="write_external_state",
                description="Write one value to an external system.",
                input_schema={"type": "object"},
                mode="async",
                side_effect_level="write",
                source="star_protocol",
                target="environment",
            )
        )
        task_system = TaskSystem()
        state = AgentState.new("idempotency-test")
        task = task_system.create_task(
            state,
            title="Persist an external value",
            goal="Write the value exactly once",
            purpose=MULTI_STEP_OBJECTIVE_PURPOSE,
        )
        first_protocol = RecordingProtocol()
        executor = ActionExecutor(
            EventBus(trace=False),
            registry,
            protocol_interface=first_protocol,  # type: ignore[arg-type]
            task_system=task_system,
            trace=False,
        )
        args = {"value": 42, "metadata": {"source": "test"}}

        started_events = await executor.start_action(
            state=state,
            task_id=task.task_id,
            action_name="write_external_state",
            args=args,
            causation_id="event-1",
        )
        self.assertEqual(len(first_protocol.sent_actions), 1)
        self.assertEqual(len(started_events), 1)
        task_system.apply_event(state, started_events[0])
        run_id = started_events[0].action_run_id
        self.assertIsNotNone(run_id)
        task_system.apply_event(
            state,
            AgentEvent.make(
                agent_id=state.agent_id,
                type="action.completed",
                source="test",
                task_id=task.task_id,
                action_run_id=run_id,
                payload={"result": {"written": True}},
            ),
        )
        state.action_runs[run_id].idempotency_key = "legacy-process-hash"

        restored = AgentState.from_dict(state.to_dict())
        second_protocol = RecordingProtocol()
        restarted_executor = ActionExecutor(
            EventBus(trace=False),
            registry,
            protocol_interface=second_protocol,  # type: ignore[arg-type]
            task_system=task_system,
            trace=False,
        )
        replay_events = await restarted_executor.start_action(
            state=restored,
            task_id=task.task_id,
            action_name="write_external_state",
            args={"metadata": {"source": "test"}, "value": 42},
            causation_id="event-after-restart",
        )

        self.assertEqual(second_protocol.sent_actions, [])
        self.assertEqual(len(replay_events), 1)
        self.assertEqual(replay_events[0].type, "action.completed")
        self.assertEqual(replay_events[0].action_run_id, run_id)
        self.assertTrue(replay_events[0].payload["deduplicated"])
        self.assertEqual(replay_events[0].payload["result"], {"written": True})
        self.assertTrue(restored.action_runs[run_id].idempotency_key.startswith("sha256:"))

    async def test_read_action_is_executed_again_instead_of_replayed(self) -> None:
        registry = ActionRegistry()
        task_system = TaskSystem()
        state = AgentState.new("read-action-test")
        task = task_system.create_task(
            state,
            title="Inspect runtime state",
            goal="Read the latest state",
            purpose=MULTI_STEP_OBJECTIVE_PURPOSE,
        )
        executor = ActionExecutor(
            EventBus(trace=False),
            registry,
            task_system=task_system,
            trace=False,
        )

        first_events = await executor.start_action(
            state=state,
            task_id=task.task_id,
            action_name="query_task_status",
            args={},
        )
        for event in first_events:
            task_system.apply_event(state, event)

        second_events = await executor.start_action(
            state=state,
            task_id=task.task_id,
            action_name="query_task_status",
            args={},
        )

        self.assertEqual(len(first_events), 2)
        self.assertEqual(len(second_events), 2)
        self.assertNotEqual(
            first_events[0].action_run_id,
            second_events[0].action_run_id,
        )


if __name__ == "__main__":
    unittest.main()
