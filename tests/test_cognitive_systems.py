from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from agent.protocols import AgentEvent, AgentState
from agent.runtime.action_systems.actions import ActionRegistry
from agent.runtime.cognition_system import DecisionSystem, EmotionSystem
from agent.runtime.persistence_system import MemoryStore
from agent.runtime.state_systems import ContextBuilder, MemorySystem
from agent_ling.config import load_agent_config


class MemorySystemTests(unittest.TestCase):
    def test_event_capture_is_durable_searchable_redacted_and_idempotent(self) -> None:
        config = load_agent_config()
        with TemporaryDirectory() as directory:
            store = MemoryStore(directory)
            system = MemorySystem(config.memory, store=store)
            state = AgentState.new("memory-test")
            event = AgentEvent.make(
                agent_id=state.agent_id,
                type="user.message",
                source="test",
                payload={
                    "content": "请记住修复中继器需要铜线",
                    "api_key": "must-not-be-persisted",
                },
            )

            first = system.observe_event(state, event)
            second = system.observe_event(state, event)

            self.assertIsNotNone(first)
            self.assertEqual(first.memory_id, second.memory_id)
            restored = store.get(state.agent_id, first.memory_id)
            self.assertIsNotNone(restored)
            self.assertIn("修复中继器", restored.content)
            self.assertNotIn("must-not-be-persisted", restored.content)
            self.assertIn("[REDACTED]", restored.content)
            matches = system.search(
                agent_id=state.agent_id,
                query="修复中继器",
            )
            self.assertEqual([item["memory_id"] for item in matches], [first.memory_id])
            pending = state.workspace.variables["memory_system"][
                "pending_reflection_ids"
            ]
            self.assertEqual(pending, [first.memory_id])


class EmotionSystemTests(unittest.TestCase):
    def test_success_and_failure_update_persistent_affect(self) -> None:
        config = load_agent_config()
        system = EmotionSystem(config.emotion)
        state = AgentState.new("emotion-test")

        positive = system.observe_event(
            state,
            AgentEvent.make(
                agent_id=state.agent_id,
                type="action.completed",
                source="test",
                payload={"result": {"completed": True}},
            ),
        )
        positive_valence = float(positive["valence"])
        self.assertGreater(positive_valence, 0)

        negative = system.observe_event(
            state,
            AgentEvent.make(
                agent_id=state.agent_id,
                type="action.failed",
                source="test",
                payload={"error": {"message": "repair failed"}},
            ),
        )
        self.assertLess(float(negative["valence"]), positive_valence)
        self.assertGreater(float(negative["arousal"]), 0.15)
        self.assertIn("primary", negative)
        self.assertEqual(len(negative["history"]), 2)


class DecisionSystemTests(unittest.TestCase):
    def test_decision_context_contains_emotion_memory_and_memory_tools(self) -> None:
        config = load_agent_config()
        with TemporaryDirectory() as directory:
            memory = MemorySystem(config.memory, store=MemoryStore(directory))
            emotion = EmotionSystem(config.emotion)
            builder = ContextBuilder(
                config.generator.prompt_for("decision").context_policy
            )
            decision = DecisionSystem(
                config.decision,
                context_builder=builder,
                memory_system=memory,
                emotion_system=emotion,
            )
            state = AgentState.new("decision-test")
            event = AgentEvent.make(
                agent_id=state.agent_id,
                type="agent.thought",
                source="test",
                payload={"content": "继续修复中继器"},
            )
            memory.observe_event(state, event)
            emotion.observe_event(state, event)

            context = decision.build_context(
                state=state,
                event=event,
                action_specs=ActionRegistry().list_specs(),
            )

            self.assertIn("emotion", context["cognition"])
            self.assertEqual(context["long_term_memory"]["count"], 1)
            tool_names = {
                tool["function"]["name"] for tool in context["_model_tools"]
            }
            self.assertIn("search_memory", tool_names)
            self.assertIn("read_memory", tool_names)


if __name__ == "__main__":
    unittest.main()
