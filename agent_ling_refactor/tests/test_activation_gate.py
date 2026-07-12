from __future__ import annotations

import unittest

from agent.protocols import AgentEvent, AgentState
from agent.runtime.action_systems.task_system import MULTI_STEP_OBJECTIVE_PURPOSE

from agent_ling_refactor.activation import ModelActivationGate
from agent_ling_refactor.scheduling import RefactoredTaskSystem
from agent_ling_refactor.settings import load_refactor_settings


class ActivationGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = load_refactor_settings()
        self.gate = ModelActivationGate(self.settings.activation)
        self.state = AgentState.new("activation-test")
        self.tasks = RefactoredTaskSystem()
        self.task = self.tasks.create_task(
            self.state,
            title="test",
            goal="test activation",
            purpose=MULTI_STEP_OBJECTIVE_PURPOSE,
        )

    def event(self, event_type: str, payload: dict[str, object]) -> AgentEvent:
        return AgentEvent.make(
            agent_id=self.state.agent_id,
            type=event_type,
            source="test",
            task_id=self.task.task_id,
            payload=payload,
        )

    def test_internal_task_updates_never_activate_model(self) -> None:
        decision = self.gate.evaluate(
            self.state,
            self.event(
                "action.internal.completed",
                {"action_name": "runtime_update_task", "result": {"status": "waiting"}},
            ),
        )
        self.assertFalse(decision.activate)
        self.assertIn("existing decision", decision.reason)

    def test_task_not_runnable_failure_is_state_only(self) -> None:
        decision = self.gate.evaluate(
            self.state,
            self.event(
                "action.failed",
                {
                    "action_name": "observe_scene",
                    "error": {
                        "type": "task_not_runnable",
                        "blockers": [{"kind": "unresolved_waits"}],
                    },
                },
            ),
        )
        self.assertFalse(decision.activate)
        self.assertIn("deterministic task constraints", decision.reason)

    def test_waiting_task_suppresses_action_results(self) -> None:
        self.tasks.add_wait(
            self.state,
            self.task.task_id,
            {"awaiting": "human_response"},
        )
        decision = self.gate.evaluate(
            self.state,
            self.event(
                "action.completed",
                {"action_name": "observe_scene", "result": {"seen": True}},
            ),
        )
        self.assertFalse(decision.activate)
        self.assertIn("explicitly waiting", decision.reason)

    def test_equivalent_evidence_activates_only_once(self) -> None:
        first = self.gate.evaluate(
            self.state,
            self.event(
                "action.completed",
                {"action_name": "observe_scene", "result": {"value": 1}},
            ),
        )
        second = self.gate.evaluate(
            self.state,
            self.event(
                "action.completed",
                {"action_name": "observe_scene", "result": {"value": 1}},
            ),
        )
        self.assertTrue(first.activate)
        self.assertFalse(second.activate)
        self.assertIn("already evaluated", second.reason)

    def test_self_broadcast_does_not_duplicate_action_outcome(self) -> None:
        decision = self.gate.evaluate(
            self.state,
            AgentEvent.make(
                agent_id=self.state.agent_id,
                type="protocol.event",
                source="star_protocol",
                payload={
                    "broadcast": True,
                    "content": {
                        "name": "social_action_performed",
                        "data": {"actor": self.state.agent_id, "performed": True},
                    },
                },
            ),
        )
        self.assertFalse(decision.activate)
        self.assertIn("duplicates an action outcome", decision.reason)

    def test_decision_chain_has_a_hard_hop_budget(self) -> None:
        for index in range(self.settings.activation.max_decision_hops):
            decision = self.gate.evaluate(
                self.state,
                self.event(
                    "action.completed",
                    {"action_name": "step", "result": {"index": index}},
                ),
            )
            self.assertTrue(decision.activate)
        blocked = self.gate.evaluate(
            self.state,
            self.event(
                "action.completed",
                {"action_name": "step", "result": {"index": 999}},
            ),
        )
        self.assertFalse(blocked.activate)
        self.assertIn("hop budget", blocked.reason)
        self.assertEqual(self.task.status, "blocked")
        self.assertIn("model_activation_paused", self.task.progress)

        resumed = self.gate.begin_user_turn(
            self.state,
            AgentEvent.make(
                agent_id=self.state.agent_id,
                type="user.message",
                source="test",
                payload={"content": "continue"},
            ),
        )
        self.assertTrue(resumed.activate)
        self.assertEqual(self.task.status, "runnable")

    def test_rate_limit_creates_backoff(self) -> None:
        retryable = self.gate.record_error(
            self.state,
            RuntimeError("429 Too many requests"),
        )
        self.assertTrue(retryable)
        self.assertTrue(self.gate.in_backoff(self.state))
        admission = self.gate.begin_user_turn(
            self.state,
            AgentEvent.make(
                agent_id=self.state.agent_id,
                type="user.message",
                source="test",
                payload={"content": "hello"},
            ),
        )
        self.assertFalse(admission.activate)


class RefactoredTaskSystemTests(unittest.TestCase):
    def test_empty_wait_is_ignored_and_user_message_resolves_human_wait(self) -> None:
        state = AgentState.new("wait-test")
        system = RefactoredTaskSystem()
        task = system.create_task(
            state,
            title="wait",
            goal="wait for human",
            purpose=MULTI_STEP_OBJECTIVE_PURPOSE,
        )
        system.add_wait(state, task.task_id, {})
        self.assertEqual(task.waiting_on, [])
        system.add_wait(
            state,
            task.task_id,
            {"awaiting": "human_response", "scene": "test"},
        )
        self.assertEqual(task.status, "waiting")

        system.apply_event(
            state,
            AgentEvent.make(
                agent_id=state.agent_id,
                type="user.message",
                source="test",
                payload={"content": "我回复了"},
            ),
        )

        self.assertEqual(task.waiting_on, [])
        self.assertEqual(task.status, "runnable")
        self.assertIn("last_wait_resolution", task.progress)


if __name__ == "__main__":
    unittest.main()
