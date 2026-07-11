from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ....config import ConversationConfig
from ....protocols import AgentState, ConversationTurn, JsonDict, ensure_json_dict
from ...kernel.generator_runtime import GeneratorRuntime
from ...persistence_system import ConversationStore


@dataclass(frozen=True)
class BrocaResult:
    text: str
    trace: JsonDict

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace", ensure_json_dict(self.trace))


class BrocaSystem:
    """Turns a communicative intent into natural, persona-consistent speech."""

    def __init__(
        self,
        *,
        generator_runtime: GeneratorRuntime,
        store: ConversationStore,
        config: ConversationConfig,
    ) -> None:
        self.generator_runtime = generator_runtime
        self.store = store
        self.config = config

    async def speak(
        self,
        *,
        state: AgentState,
        turn: ConversationTurn | None,
        speech_intent: JsonDict,
    ) -> BrocaResult:
        recent_turns: List[JsonDict] = []
        if turn is not None:
            recent_turns = self.store.context_turns(
                agent_id=turn.agent_id,
                conversation_id=turn.conversation_id,
                before_turn_id=turn.turn_id,
                limit=self.config.recent_turn_limit,
                verbatim_limit=self.config.verbatim_turn_limit,
                compact_limit=self.config.compact_turn_limit,
            )
        runtime_context = self._context(
            state=state,
            turn=turn,
            recent_turns=recent_turns,
            speech_intent=speech_intent,
        )
        result, trace = await self.generator_runtime.generate_text_with_trace(
            runtime_context,
            session_id="broca",
        )
        return BrocaResult(text=(result.text or "").strip(), trace=trace)

    def _context(
        self,
        *,
        state: AgentState,
        turn: ConversationTurn | None,
        recent_turns: List[JsonDict],
        speech_intent: JsonDict,
    ) -> JsonDict:
        current_task = state.tasks.get(state.workspace.current_task_id or "")
        speaker_id = turn.speaker_id if turn else str(speech_intent.get("speaker_id") or "")
        relationships = state.workspace.variables.get("relationships")
        relationship = (
            relationships.get(speaker_id)
            if isinstance(relationships, dict) and speaker_id
            else None
        )
        return {
            "agent": state.profile.to_dict(),
            "conversation": {
                "stage": "speech",
                "conversation_id": turn.conversation_id if turn else speech_intent.get("conversation_id"),
                "turn_id": turn.turn_id if turn else speech_intent.get("turn_id"),
                "speaker_id": speaker_id,
                "speaker_context": turn.speaker_context if turn else speech_intent.get("speaker_context"),
                "scene_context": turn.scene_context if turn else speech_intent.get("scene_context"),
                "incoming_utterance": turn.utterance if turn else None,
                "understanding": turn.understanding if turn else speech_intent.get("understanding"),
                "recent_turns": recent_turns,
                "speech_intent": speech_intent,
            },
            "expression_state": {
                "emotion": state.workspace.variables.get("emotion_state"),
                "relationship": relationship,
                "current_task": self._task_view(current_task),
                "scene": state.workspace.variables.get("scene"),
            },
        }

    def _task_view(self, task: object) -> JsonDict | None:
        if task is None:
            return None
        return {
            "task_id": getattr(task, "task_id", None),
            "title": getattr(task, "title", ""),
            "goal": getattr(task, "goal", ""),
            "status": getattr(task, "status", ""),
            "progress": getattr(task, "progress", {}),
        }
