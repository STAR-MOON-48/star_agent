from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ....config import ConversationConfig
from ....protocols import (
    AgentEvent,
    AgentState,
    ConversationTurn,
    ConversationUnderstanding,
    JsonDict,
    utc_now,
)
from ...kernel.generator_runtime import GeneratorRuntime
from ...persistence_system import ConversationStore
from .broca import BrocaResult, BrocaSystem
from .wernicke import WernickeResult, WernickeSystem
from ...state_systems import MemorySystem


@dataclass(frozen=True)
class ConversationUnderstandingResult:
    event: AgentEvent
    result: WernickeResult


@dataclass(frozen=True)
class ConversationSpeechResult:
    event: AgentEvent
    result: BrocaResult


class ConversationSystem:
    """Owns conversation turn lifecycle; semantic work belongs to Wernicke/Broca."""

    _PROACTIVE_ENABLE_MARKERS = (
        "不用等我回复",
        "不要等我回复",
        "不必等我回复",
        "无需等我回复",
        "别等我回复",
        "你可以一直说",
        "可以一直说",
        "继续说下去",
        "你自己说",
        "主动说",
        "不用一问一答",
        "不要一问一答",
        "别一问一答",
        "don't wait for my reply",
        "do not wait for my reply",
        "keep talking",
        "keep speaking",
    )
    _PROACTIVE_DISABLE_MARKERS = (
        "别说了",
        "不要再说",
        "不用继续",
        "先别说",
        "暂停说话",
        "安静一下",
        "等我回复再说",
        "不要主动说",
        "别主动说",
        "stop talking",
        "stop speaking",
        "wait for my reply",
        "pause talking",
    )

    def __init__(
        self,
        *,
        generator_runtime: GeneratorRuntime,
        store: ConversationStore,
        config: ConversationConfig,
        memory_system: MemorySystem | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.wernicke = WernickeSystem(
            generator_runtime=generator_runtime,
            store=store,
            config=config,
            memory_system=memory_system,
        )
        self.broca = BrocaSystem(
            generator_runtime=generator_runtime,
            store=store,
            config=config,
        )

    def receive_utterance(self, state: AgentState, event: AgentEvent) -> AgentEvent:
        speaker_id = str(event.payload.get("sender") or "human")
        conversation_id = str(
            event.payload.get("conversation_id")
            or f"{event.source}:{speaker_id}"
        )
        turn = self.store.create_turn(
            agent_id=state.agent_id,
            conversation_id=conversation_id,
            speaker_id=speaker_id,
            recipient_id=state.agent_id,
            channel=event.source,
            source_event_id=event.event_id,
            utterance=str(event.payload.get("content") or ""),
            speaker_context=self._dict_value(event.payload.get("speaker_context")),
            scene_context=self._dict_value(event.payload.get("scene_context")),
        )
        self._update_proactive_mode(state, turn)
        return AgentEvent.make(
            agent_id=state.agent_id,
            type="conversation.understanding.requested",
            source="conversation_system",
            correlation_id=conversation_id,
            causation_id=event.event_id,
            payload=self._turn_payload(turn),
            priority=event.priority,
        )

    async def understand(
        self,
        state: AgentState,
        event: AgentEvent,
    ) -> ConversationUnderstandingResult:
        turn = self._turn_for_event(state.agent_id, event)
        turn.status = "understanding"
        self.store.save_turn(turn)
        result = await self.wernicke.understand(state=state, turn=turn)
        turn.understanding = result.understanding.to_dict()
        turn.status = "understood"
        self.store.save_turn(turn)
        ready = AgentEvent.make(
            agent_id=state.agent_id,
            type="conversation.understanding.ready",
            source="wernicke_system",
            correlation_id=turn.conversation_id,
            causation_id=event.event_id,
            payload={
                **self._turn_payload(turn),
                "understanding": result.understanding.to_dict(),
            },
            priority=event.priority,
        )
        return ConversationUnderstandingResult(event=ready, result=result)

    def supersede_stale_understanding(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
    ) -> Optional[JsonDict]:
        turn = self._turn_for_event(state.agent_id, event)
        latest = self.store.latest_turn(
            agent_id=state.agent_id,
            conversation_id=turn.conversation_id,
        )
        if latest is None or latest.turn_id == turn.turn_id:
            return None
        record = {
            "event_id": event.event_id,
            "event_type": event.type,
            "reason": "newer_turn_already_received",
            "superseded_by_turn_id": latest.turn_id,
            "created_at": event.created_at,
        }
        turn.suppressed_speech_intents.append(record)
        turn.status = "superseded"
        self.store.save_turn(turn)
        return record

    def route_understanding(self, state: AgentState, event: AgentEvent) -> list[AgentEvent]:
        turn = self._turn_for_event(state.agent_id, event)
        understanding = self._understanding_for_event(turn, event)
        if understanding.decision_needed:
            turn.status = "awaiting_decision"
            self.store.save_turn(turn)
            return [
                AgentEvent.make(
                    agent_id=state.agent_id,
                    type="conversation.decision.requested",
                    source="conversation_system",
                    correlation_id=turn.conversation_id,
                    causation_id=event.event_id,
                    payload={
                        **self._turn_payload(turn),
                        "content": turn.utterance,
                        "understanding": understanding.to_dict(),
                        "decision_request": understanding.decision_request,
                    },
                    priority=event.priority,
                )
            ]
        if understanding.response_needed:
            return [
                self.request_speech(
                    state=state,
                    event=event,
                    content="回应对方当前话语，不虚构未经确认的事实或行动结果。",
                    kind="direct_conversation_response",
                )
            ]
        turn.status = "completed_without_response"
        self.store.save_turn(turn)
        return []

    def request_speech(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        content: str,
        kind: str = "decision_response",
    ) -> AgentEvent:
        context = self.context_for_event(state, event)
        turn = self._optional_turn(state.agent_id, context.get("turn_id"))
        speech_intent: JsonDict = {
            "kind": kind,
            "content": content,
            "conversation_id": context.get("conversation_id"),
            "turn_id": context.get("turn_id"),
            "speaker_id": context.get("speaker_id"),
            "recipient_id": context.get("recipient_id"),
        }
        if turn is not None:
            turn.speech_intent = speech_intent
            turn.status = "speech_requested"
            self.store.save_turn(turn)
        return AgentEvent.make(
            agent_id=state.agent_id,
            type="conversation.speech.requested",
            source="conversation_system",
            task_id=event.task_id,
            correlation_id=str(context.get("conversation_id") or event.correlation_id or "") or None,
            causation_id=event.event_id,
            payload={**context, "speech_intent": speech_intent},
            priority=event.priority,
        )

    async def speak(
        self,
        state: AgentState,
        event: AgentEvent,
    ) -> ConversationSpeechResult:
        context = self.context_for_event(state, event)
        turn = self._optional_turn(state.agent_id, context.get("turn_id"))
        speech_intent = event.payload.get("speech_intent")
        if not isinstance(speech_intent, dict):
            speech_intent = {"kind": "unspecified", "content": str(speech_intent or "")}
        if turn is not None:
            turn.status = "speaking"
            self.store.save_turn(turn)
        result = await self.broca.speak(
            state=state,
            turn=turn,
            speech_intent=dict(speech_intent),
        )
        ready = AgentEvent.make(
            agent_id=state.agent_id,
            type="conversation.utterance.ready",
            source="broca_system",
            task_id=event.task_id,
            correlation_id=str(context.get("conversation_id") or event.correlation_id or "") or None,
            causation_id=event.event_id,
            payload={
                **context,
                "content": result.text,
                "speech_intent": speech_intent,
            },
            priority=event.priority,
        )
        return ConversationSpeechResult(event=ready, result=result)

    def suppress_if_newer_turn_is_waiting(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
    ) -> Optional[JsonDict]:
        context = self.context_for_event(state, event)
        turn = self._optional_turn(state.agent_id, context.get("turn_id"))
        conversation_id = context.get("conversation_id")
        if turn is None or not isinstance(conversation_id, str):
            return None
        speech_intent = event.payload.get("speech_intent")
        speech_kind = (
            str(speech_intent.get("kind") or "")
            if isinstance(speech_intent, dict)
            else ""
        )
        if speech_kind == "proactive_continuation":
            mode = self._proactive_mode(state, conversation_id)
            expected_generation = (
                speech_intent.get("proactive_generation")
                if isinstance(speech_intent, dict)
                else None
            )
            if (
                not mode.get("enabled")
                or expected_generation != mode.get("generation")
            ):
                suppressed = {
                    "event_id": event.event_id,
                    "event_type": event.type,
                    "speech_intent": dict(speech_intent),
                    "content": event.payload.get("content"),
                    "reason": "proactive_mode_changed",
                    "created_at": event.created_at,
                }
                turn.suppressed_speech_intents.append(suppressed)
                self.store.save_turn(turn)
                return suppressed
        if turn.response_event_id is not None and speech_kind not in {
            "progress_response",
            "proactive_continuation",
        }:
            suppressed = {
                "event_id": event.event_id,
                "event_type": event.type,
                "speech_intent": (
                    dict(speech_intent)
                    if isinstance(speech_intent, dict)
                    else speech_intent
                ),
                "content": event.payload.get("content"),
                "reason": "turn_already_answered",
                "response_event_id": turn.response_event_id,
                "created_at": event.created_at,
            }
            turn.suppressed_speech_intents.append(suppressed)
            self.store.save_turn(turn)
            return suppressed
        latest = self.store.latest_turn(
            agent_id=state.agent_id,
            conversation_id=conversation_id,
        )
        if latest is None or latest.turn_id == turn.turn_id:
            return None
        if speech_kind != "proactive_continuation" and (
            latest.response_event_id is not None
            or latest.status in {"completed_without_response", "failed"}
        ):
            return None
        suppressed = {
            "event_id": event.event_id,
            "event_type": event.type,
            "speech_intent": dict(speech_intent) if isinstance(speech_intent, dict) else speech_intent,
            "content": event.payload.get("content"),
            "reason": (
                "newer_turn_exists"
                if speech_kind == "proactive_continuation"
                else "newer_turn_waiting_for_response"
            ),
            "superseded_by_turn_id": latest.turn_id,
            "created_at": event.created_at,
        }
        turn.suppressed_speech_intents.append(suppressed)
        turn.status = "superseded"
        self.store.save_turn(turn)
        return suppressed

    def mark_sent(self, state: AgentState, event: AgentEvent) -> AgentEvent:
        context = self.context_for_event(state, event)
        turn = self._optional_turn(state.agent_id, context.get("turn_id"))
        if turn is not None:
            content = str(event.payload.get("content") or "")
            turn.response_text = content
            turn.response_event_id = event.event_id
            turn.outbound_utterances.append(
                {
                    "event_id": event.event_id,
                    "content": content,
                    "speech_intent": event.payload.get("speech_intent"),
                    "created_at": event.created_at,
                }
            )
            turn.status = "completed"
            self.store.save_turn(turn)
        return AgentEvent.make(
            agent_id=state.agent_id,
            type="conversation.utterance.sent",
            source="conversation_system",
            task_id=event.task_id,
            correlation_id=event.correlation_id,
            causation_id=event.event_id,
            payload={
                **context,
                "content": str(event.payload.get("content") or ""),
                "speech_intent": event.payload.get("speech_intent"),
            },
            priority=event.priority,
        )

    def proactive_follow_up_after_sent(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
    ) -> tuple[Optional[AgentEvent], float]:
        if not self.config.proactive_enabled:
            return None, 0.0
        context = self.context_for_event(state, event)
        conversation_id = context.get("conversation_id")
        turn = self._optional_turn(state.agent_id, context.get("turn_id"))
        if turn is None or not isinstance(conversation_id, str):
            return None, 0.0
        latest = self.store.latest_turn(
            agent_id=state.agent_id,
            conversation_id=conversation_id,
        )
        if latest is None or latest.turn_id != turn.turn_id:
            return None, 0.0
        mode = self._proactive_mode(state, conversation_id)
        remaining = int(mode.get("remaining_follow_ups") or 0)
        if not mode.get("enabled") or remaining <= 0:
            return None, 0.0
        mode["remaining_follow_ups"] = remaining - 1
        mode["updated_at"] = utc_now()
        speech_intent: JsonDict = {
            "kind": "proactive_continuation",
            "content": (
                "对方明确允许不等待回复。自然、简短地延续当前话题或分享一个相关念头；"
                "不要用提问把话轮重新交回对方，不要重复上一条内容，也不要表演式长篇独白。"
            ),
            "conversation_id": conversation_id,
            "turn_id": turn.turn_id,
            "speaker_id": turn.speaker_id,
            "recipient_id": turn.speaker_id,
            "proactive_generation": mode.get("generation"),
            "proactive_remaining_after_this": remaining - 1,
        }
        follow_up = AgentEvent.make(
            agent_id=state.agent_id,
            type="conversation.speech.requested",
            source="conversation_system.proactive",
            correlation_id=conversation_id,
            causation_id=event.event_id,
            payload={**context, "speech_intent": speech_intent},
            priority=event.priority,
        )
        return follow_up, self.config.proactive_interval_seconds

    def context_for_event(self, state: AgentState, event: AgentEvent) -> JsonDict:
        context: JsonDict = {}
        for key in ("conversation_id", "turn_id", "speaker_id", "recipient_id", "channel"):
            value = event.payload.get(key)
            if value not in (None, ""):
                context[key] = value
        if event.task_id and event.task_id in state.tasks:
            continuation = state.tasks[event.task_id].continuation
            for key in ("conversation_id", "turn_id", "speaker_id", "recipient_id", "channel"):
                if key not in context and continuation.get(key) not in (None, ""):
                    context[key] = continuation[key]
        if "recipient_id" not in context:
            recipient = event.payload.get("sender")
            if recipient:
                context["recipient_id"] = recipient
        return context

    def attach_context_to_continuation(
        self,
        state: AgentState,
        event: AgentEvent,
        continuation: JsonDict,
    ) -> None:
        for key, value in self.context_for_event(state, event).items():
            continuation.setdefault(key, value)

    def record_decision(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        decision: JsonDict,
    ) -> None:
        context = self.context_for_event(state, event)
        turn = self._optional_turn(state.agent_id, context.get("turn_id"))
        if turn is None:
            return
        turn.decision = dict(decision)
        has_action = any(
            isinstance(command, dict) and command.get("type") == "start_action"
            for command in decision.get("commands", [])
        )
        turn.status = "awaiting_action" if has_action else "decision_ready"
        self.store.save_turn(turn)

    def mark_failed(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        error: JsonDict,
    ) -> None:
        context = self.context_for_event(state, event)
        turn = self._optional_turn(state.agent_id, context.get("turn_id"))
        if turn is None:
            return
        turn.status = "failed"
        if event.type == "conversation.understanding.requested":
            turn.understanding = {"error": dict(error)}
        else:
            turn.speech_intent = {
                **(turn.speech_intent or {}),
                "error": dict(error),
            }
        self.store.save_turn(turn)

    def _turn_for_event(self, agent_id: str, event: AgentEvent) -> ConversationTurn:
        turn_id = event.payload.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            raise KeyError(f"Conversation event {event.type} has no turn_id")
        return self.store.get_turn(agent_id, turn_id)

    def _optional_turn(self, agent_id: str, turn_id: object) -> Optional[ConversationTurn]:
        if not isinstance(turn_id, str) or not turn_id:
            return None
        try:
            return self.store.get_turn(agent_id, turn_id)
        except KeyError:
            return None

    def _understanding_for_event(
        self,
        turn: ConversationTurn,
        event: AgentEvent,
    ) -> ConversationUnderstanding:
        value = event.payload.get("understanding") or turn.understanding
        if not isinstance(value, dict):
            raise ValueError(f"Conversation turn {turn.turn_id} has no understanding")
        return ConversationUnderstanding.from_dict(value)

    def _turn_payload(self, turn: ConversationTurn) -> JsonDict:
        return {
            "conversation_id": turn.conversation_id,
            "turn_id": turn.turn_id,
            "speaker_id": turn.speaker_id,
            "recipient_id": turn.speaker_id,
            "channel": turn.channel,
        }

    def _dict_value(self, value: object) -> JsonDict:
        return dict(value) if isinstance(value, dict) else {}

    def _update_proactive_mode(
        self,
        state: AgentState,
        turn: ConversationTurn,
    ) -> None:
        modes = state.workspace.variables.setdefault("conversation_autonomy", {})
        if not isinstance(modes, dict):
            modes = {}
            state.workspace.variables["conversation_autonomy"] = modes
        current = modes.get(turn.conversation_id)
        mode = dict(current) if isinstance(current, dict) else {}
        normalized = " ".join(turn.utterance.casefold().split())
        disable = any(
            marker in normalized for marker in self._PROACTIVE_DISABLE_MARKERS
        )
        enable = (
            not disable
            and any(marker in normalized for marker in self._PROACTIVE_ENABLE_MARKERS)
        )
        generation = int(mode.get("generation") or 0)
        if enable:
            generation += 1
            mode.update(
                {
                    "enabled": True,
                    "speaker_id": turn.speaker_id,
                    "generation": generation,
                    "remaining_follow_ups": self.config.proactive_burst_messages,
                    "reason": "explicit_user_permission",
                    "updated_at": utc_now(),
                }
            )
        elif disable:
            generation += 1
            mode.update(
                {
                    "enabled": False,
                    "speaker_id": turn.speaker_id,
                    "generation": generation,
                    "remaining_follow_ups": 0,
                    "reason": "explicit_user_stop",
                    "updated_at": utc_now(),
                }
            )
        elif mode.get("enabled"):
            generation += 1
            mode.update(
                {
                    "speaker_id": turn.speaker_id,
                    "generation": generation,
                    "remaining_follow_ups": self.config.proactive_burst_messages,
                    "reason": "active_mode_new_turn",
                    "updated_at": utc_now(),
                }
            )
        if mode:
            modes[turn.conversation_id] = mode

    def _proactive_mode(
        self,
        state: AgentState,
        conversation_id: str,
    ) -> JsonDict:
        modes = state.workspace.variables.get("conversation_autonomy")
        if not isinstance(modes, dict):
            return {}
        value = modes.get(conversation_id)
        return value if isinstance(value, dict) else {}
