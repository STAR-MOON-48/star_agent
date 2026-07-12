from __future__ import annotations

from typing import Any

from agent.protocols import AgentEvent, AgentState, ConversationTurn, JsonDict, utc_now
from agent.runtime.persistence_system import ConversationStore

from .messages import MessagePurpose, NaturalMessage
from .settings import ConversationSettings


_ENABLE_MARKERS = (
    "不用等我",
    "不必等我",
    "继续说",
    "一直说",
    "keep talking",
    "continue without",
)
_DISABLE_MARKERS = (
    "先别说",
    "不要继续",
    "等我回复",
    "停一下",
    "stop talking",
    "wait for me",
)


class ConversationLedger:
    """Durable turn bookkeeping without a second cognitive model chain."""

    def __init__(
        self,
        *,
        store: ConversationStore,
        settings: ConversationSettings,
    ) -> None:
        self.store = store
        self.settings = settings

    def receive(self, state: AgentState, event: AgentEvent) -> ConversationTurn:
        sender = str(event.payload.get("sender") or "local_user")
        conversation_id = str(
            event.payload.get("conversation_id") or f"{state.agent_id}--{sender}"
        )
        latest = self.store.latest_turn(
            agent_id=state.agent_id,
            conversation_id=conversation_id,
        )
        if latest is not None and latest.response_event_id is None and latest.status not in {
            "failed",
            "superseded",
            "completed_without_response",
        }:
            latest.status = "superseded"
            latest.suppressed_speech_intents.append(
                {
                    "reason": "newer_turn_received",
                    "superseded_by_event_id": event.event_id,
                    "created_at": utc_now(),
                }
            )
            self.store.save_turn(latest)
        turn = self.store.create_turn(
            agent_id=state.agent_id,
            conversation_id=conversation_id,
            speaker_id=sender,
            recipient_id=state.agent_id,
            channel=event.source,
            source_event_id=event.event_id,
            utterance=str(event.payload.get("content") or ""),
            speaker_context=_mapping(event.payload.get("speaker_context")),
            scene_context=_mapping(event.payload.get("scene_context")),
        )
        turn.status = "processing"
        turn.understanding = {
            "semantic_summary": turn.utterance,
            "source": f"speaker:{sender}",
            "note": "保留说话者来源；未将自述自动视为外部事实。",
        }
        self.store.save_turn(turn)
        self._update_proactive_mode(state, turn)
        return turn

    def understanding_message(
        self,
        turn: ConversationTurn,
        event: AgentEvent,
    ) -> NaturalMessage:
        return NaturalMessage(
            sender="conversation",
            recipient="understanding",
            purpose=MessagePurpose.UNDERSTANDING,
            text=(
                f"请理解 {turn.speaker_id} 刚刚说的这句话：{turn.utterance}。"
                "保留说话者来源；没有证据的地方不要补写。"
            ),
            event_id=event.event_id,
            conversation_id=turn.conversation_id,
            turn_id=turn.turn_id,
        )

    def expression_message(
        self,
        *,
        turn: ConversationTurn,
        event: AgentEvent,
        understanding: str,
    ) -> NaturalMessage:
        return NaturalMessage(
            sender="understanding",
            recipient="expression",
            purpose=MessagePurpose.EXPRESSION,
            text=(
                f"对方原话是：{turn.utterance}。"
                f"对这句话的理解是：{understanding}。"
                "请据此形成此刻自然说出口的回复。"
            ),
            event_id=event.event_id,
            conversation_id=turn.conversation_id,
            turn_id=turn.turn_id,
        )

    def decision_message(
        self,
        *,
        turn: ConversationTurn,
        event: AgentEvent,
        understanding: str,
        request: str,
    ) -> NaturalMessage:
        request_text = request or "这句话涉及行动、承诺或重要判断，需要进一步决定。"
        return NaturalMessage(
            sender="understanding",
            recipient="decision",
            purpose=MessagePurpose.DECISION,
            text=(
                f"对方原话是：{turn.utterance}。"
                f"理解区域认为：{understanding}。"
                f"需要你处理的事项是：{request_text}"
            ),
            event_id=event.event_id,
            conversation_id=turn.conversation_id,
            turn_id=turn.turn_id,
        )

    def record_understanding(
        self,
        turn: ConversationTurn,
        *,
        text: str,
        decision_requested: bool,
    ) -> None:
        turn.understanding = {
            "semantic_summary": text,
            "source": f"speaker:{turn.speaker_id}",
            "natural_language": True,
            "decision_requested": decision_requested,
        }
        turn.status = "understood"
        self.store.save_turn(turn)

    def recent_context(self, turn: ConversationTurn) -> list[JsonDict]:
        return self.store.context_turns(
            agent_id=turn.agent_id,
            conversation_id=turn.conversation_id,
            before_turn_id=turn.turn_id,
            limit=self.settings.recent_turns,
            verbatim_limit=min(6, self.settings.recent_turns),
            compact_limit=self.settings.recent_turns,
        )

    def record_model_intent(
        self,
        turn: ConversationTurn,
        *,
        text: str,
        used_tools: bool,
    ) -> None:
        turn.speech_intent = {
            "kind": "direct_response" if not used_tools else "action_status",
            "content": text,
            "natural_language": True,
        }
        turn.status = "awaiting_action" if used_tools else "response_ready"
        self.store.save_turn(turn)

    def record_sent(
        self,
        turn: ConversationTurn,
        *,
        content: str,
        response_event_id: str,
        final: bool,
    ) -> None:
        turn.response_text = content
        turn.response_event_id = response_event_id
        turn.outbound_utterances.append(
            {
                "event_id": response_event_id,
                "content": content,
                "natural_language": True,
                "created_at": utc_now(),
            }
        )
        turn.status = "completed" if final else "awaiting_action"
        self.store.save_turn(turn)

    def mark_complete(self, turn: ConversationTurn) -> None:
        turn.status = "completed"
        self.store.save_turn(turn)

    def context_from_task(self, state: AgentState, event: AgentEvent) -> ConversationTurn | None:
        task = state.tasks.get(event.task_id or "")
        if task is None:
            return None
        turn_id = task.continuation.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            return None
        try:
            return self.store.get_turn(state.agent_id, turn_id)
        except KeyError:
            return None

    def proactive_event_after_reply(
        self,
        *,
        state: AgentState,
        turn: ConversationTurn,
        causation_id: str,
    ) -> AgentEvent | None:
        if not self.settings.proactive_enabled:
            return None
        mode = self._mode(state, turn.conversation_id)
        remaining = int(mode.get("remaining") or 0)
        if not mode.get("enabled") or remaining <= 0:
            return None
        mode["remaining"] = remaining - 1
        mode["updated_at"] = utc_now()
        return AgentEvent.make(
            agent_id=state.agent_id,
            type="conversation.proactive",
            source="conversation",
            correlation_id=turn.conversation_id,
            causation_id=causation_id,
            payload={
                "content": (
                    "对方明确允许不等待回复。请自然、简短地延续当前话题，"
                    "补充一个新的相关想法，不重复上一条内容。"
                ),
                "conversation_id": turn.conversation_id,
                "turn_id": turn.turn_id,
                "recipient_id": turn.speaker_id,
                "generation": mode.get("generation"),
            },
        )

    def proactive_is_current(self, state: AgentState, event: AgentEvent) -> bool:
        conversation_id = str(event.payload.get("conversation_id") or "")
        mode = self._mode(state, conversation_id)
        return bool(
            mode.get("enabled")
            and event.payload.get("generation") == mode.get("generation")
        )

    def _update_proactive_mode(self, state: AgentState, turn: ConversationTurn) -> None:
        normalized = " ".join(turn.utterance.casefold().split())
        disable = any(marker in normalized for marker in _DISABLE_MARKERS)
        enable = not disable and any(marker in normalized for marker in _ENABLE_MARKERS)
        modes = state.workspace.variables.setdefault("conversation_autonomy", {})
        if not isinstance(modes, dict):
            modes = {}
            state.workspace.variables["conversation_autonomy"] = modes
        mode = dict(modes.get(turn.conversation_id) or {})
        generation = int(mode.get("generation") or 0)
        if enable or disable or mode.get("enabled"):
            generation += 1
        if enable:
            mode.update(
                enabled=True,
                generation=generation,
                remaining=self.settings.proactive_burst_messages,
                speaker_id=turn.speaker_id,
                reason="explicit_user_permission",
                updated_at=utc_now(),
            )
        elif disable:
            mode.update(
                enabled=False,
                generation=generation,
                remaining=0,
                reason="explicit_user_stop",
                updated_at=utc_now(),
            )
        elif mode.get("enabled"):
            mode.update(
                generation=generation,
                remaining=self.settings.proactive_burst_messages,
                speaker_id=turn.speaker_id,
                reason="new_turn_while_enabled",
                updated_at=utc_now(),
            )
        if mode:
            modes[turn.conversation_id] = mode

    def _mode(self, state: AgentState, conversation_id: str) -> JsonDict:
        modes = state.workspace.variables.get("conversation_autonomy")
        if not isinstance(modes, dict):
            return {}
        mode = modes.get(conversation_id)
        return mode if isinstance(mode, dict) else {}


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, dict) else {}
