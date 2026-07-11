from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from agent.protocols import AgentEvent, AgentState
from agent.runtime.cognition_system.conversation_system import ConversationSystem
from agent.runtime.interfaces.star_session import StarSession
from agent.runtime.perception_systems import PerceptionSystem
from agent.runtime.persistence_system import ConversationStore
from agent_ling.config import load_agent_config


class _UnusedGeneratorRuntime:
    pass


class _RecordingStarClient:
    def __init__(self) -> None:
        self.outcomes: list[tuple[str, dict[str, object]]] = []

    async def send_outcome(
        self,
        recipient: str,
        content: dict[str, object],
    ) -> None:
        self.outcomes.append((recipient, content))


class UserMessageNormalizationTests(unittest.IsolatedAsyncioTestCase):
    def test_transport_action_id_produces_stable_event_identity(self) -> None:
        perception = PerceptionSystem()
        content = {
            "id": "action-user-message-1",
            "name": "user_message",
            "params": {
                "content": "连续说，不用等我回复",
                "conversation_id": "conversation-1",
            },
        }

        first = perception.star_action(
            agent_id="agent-1",
            sender="human-1",
            content=content,
        )
        second = perception.star_action(
            agent_id="agent-1",
            sender="human-1",
            content=content,
        )

        self.assertEqual(first.event_id, second.event_id)
        self.assertEqual(first.payload["message_id"], "action-user-message-1")
        self.assertEqual(
            first.idempotency_key,
            "star_user_message:human-1:action-user-message-1",
        )

    async def test_user_message_action_is_acknowledged_immediately(self) -> None:
        session = StarSession(hub_url="ws://unused")
        client = _RecordingStarClient()
        session._agent_id = "agent-1"
        session._client = client  # type: ignore[assignment]

        await session.handle_action(
            "human-1",
            {
                "id": "action-user-message-2",
                "name": "user_message",
                "params": {"content": "你好"},
            },
        )

        event = await session.next_event(timeout=0.1)
        self.assertEqual(event.type, "user.message")
        self.assertEqual(len(client.outcomes), 1)
        recipient, outcome = client.outcomes[0]
        self.assertEqual(recipient, "human-1")
        self.assertEqual(outcome["ref_id"], "action-user-message-2")
        self.assertTrue(outcome["success"])
        self.assertEqual(outcome["data"]["event_id"], event.event_id)  # type: ignore[index]


class ConversationFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        config = load_agent_config().conversation
        self.system = ConversationSystem(
            generator_runtime=_UnusedGeneratorRuntime(),  # type: ignore[arg-type]
            store=ConversationStore(self.temporary.name),
            config=config,
        )
        self.state = AgentState.new("agent-1")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def user_event(self, content: str, suffix: str) -> AgentEvent:
        return AgentEvent(
            event_id=f"event-{suffix}",
            agent_id=self.state.agent_id,
            type="user.message",
            source="star_protocol",
            payload={
                "content": content,
                "sender": "human-1",
                "conversation_id": "conversation-1",
            },
        )

    def test_pending_old_turn_is_skipped_when_newer_input_already_exists(self) -> None:
        first_request = self.system.receive_utterance(
            self.state,
            self.user_event("你可以一直说话", "1"),
        )
        second_request = self.system.receive_utterance(
            self.state,
            self.user_event("不用等我回复", "2"),
        )

        suppressed = self.system.supersede_stale_understanding(
            state=self.state,
            event=first_request,
        )

        self.assertIsNotNone(suppressed)
        self.assertEqual(
            suppressed["superseded_by_turn_id"],
            second_request.payload["turn_id"],
        )
        first_turn = self.system.store.get_turn(
            self.state.agent_id,
            str(first_request.payload["turn_id"]),
        )
        self.assertEqual(first_turn.status, "superseded")

    def test_explicit_permission_schedules_bounded_proactive_continuations(self) -> None:
        understanding_request = self.system.receive_utterance(
            self.state,
            self.user_event("不用等我回复，你可以继续说", "1"),
        )
        turn_payload = dict(understanding_request.payload)
        ready = AgentEvent.make(
            agent_id=self.state.agent_id,
            type="conversation.utterance.ready",
            source="test",
            correlation_id="conversation-1",
            payload={
                **turn_payload,
                "content": "好，我会自然地继续。",
                "speech_intent": {
                    "kind": "direct_conversation_response",
                    **turn_payload,
                },
            },
        )
        sent = self.system.mark_sent(self.state, ready)

        follow_up, delay = self.system.proactive_follow_up_after_sent(
            state=self.state,
            event=sent,
        )

        self.assertIsNotNone(follow_up)
        self.assertGreater(delay, 0)
        intent = follow_up.payload["speech_intent"]  # type: ignore[union-attr]
        self.assertEqual(intent["kind"], "proactive_continuation")
        self.assertEqual(
            intent["proactive_remaining_after_this"],
            self.system.config.proactive_burst_messages - 1,
        )
        self.assertIsNone(
            self.system.suppress_if_newer_turn_is_waiting(
                state=self.state,
                event=follow_up,  # type: ignore[arg-type]
            )
        )

    def test_new_stop_instruction_cancels_a_pending_proactive_message(self) -> None:
        request = self.system.receive_utterance(
            self.state,
            self.user_event("不用等我回复", "1"),
        )
        sent = AgentEvent.make(
            agent_id=self.state.agent_id,
            type="conversation.utterance.sent",
            source="test",
            correlation_id="conversation-1",
            payload={**request.payload, "content": "好。"},
        )
        follow_up, _ = self.system.proactive_follow_up_after_sent(
            state=self.state,
            event=sent,
        )
        self.assertIsNotNone(follow_up)

        self.system.receive_utterance(
            self.state,
            self.user_event("先别说了，等我回复再说", "2"),
        )
        suppressed = self.system.suppress_if_newer_turn_is_waiting(
            state=self.state,
            event=follow_up,  # type: ignore[arg-type]
        )

        self.assertIsNotNone(suppressed)
        self.assertEqual(suppressed["reason"], "proactive_mode_changed")


if __name__ == "__main__":
    unittest.main()
