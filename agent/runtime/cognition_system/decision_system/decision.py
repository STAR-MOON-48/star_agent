from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ....config import DecisionConfig
from ....protocols import (
    ActionSpec,
    AgentEvent,
    AgentState,
    GeneratorDecision,
    JsonDict,
    ensure_json_dict,
    ensure_json_dict_list,
    utc_now,
)
from ...kernel.generator_runtime import GeneratorRuntime
from ...state_systems import ContextBuilder, MemorySystem
from ..emotion_system import EmotionSystem


DECISION_VARIABLE_KEY = "decision_system"


@dataclass(frozen=True)
class DecisionEvaluation:
    context: JsonDict
    public_context: JsonDict
    model_tools: list[JsonDict]
    decision: GeneratorDecision
    model_trace: JsonDict

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", ensure_json_dict(self.context))
        object.__setattr__(self, "public_context", ensure_json_dict(self.public_context))
        object.__setattr__(self, "model_tools", ensure_json_dict_list(self.model_tools))
        object.__setattr__(self, "decision", ensure_json_dict(self.decision))
        object.__setattr__(self, "model_trace", ensure_json_dict(self.model_trace))


class DecisionSystem:
    """Cognition owner for context enrichment, model decision, and audit history."""

    def __init__(
        self,
        config: DecisionConfig,
        *,
        context_builder: ContextBuilder,
        memory_system: MemorySystem,
        emotion_system: EmotionSystem,
    ) -> None:
        self.config = config
        self.context_builder = context_builder
        self.memory_system = memory_system
        self.emotion_system = emotion_system

    def build_context(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        action_specs: Iterable[ActionSpec],
    ) -> JsonDict:
        context = self.context_builder.build(
            state=state,
            event=event,
            action_specs=action_specs,
        )
        context["cognition"] = {
            "emotion": self.emotion_system.context_view(state),
            "decision_history": self._recent_history(state, limit=5),
        }
        context["long_term_memory"] = self.memory_system.context_for(
            state=state,
            event=event,
            limit=self.config.memory_retrieval_limit,
        )
        context["instruction"] = (
            str(context.get("instruction") or "")
            + " Use long_term_memory as fallible recalled evidence: respect confidence and source_refs, "
            "and use search_memory/read_memory when exact details are required. Emotion may shape "
            "priority and expression intent but must not override verified facts or safety constraints."
        )
        return context

    async def evaluate(
        self,
        *,
        context: JsonDict,
        generator_runtime: GeneratorRuntime,
    ) -> DecisionEvaluation:
        result = await generator_runtime.generate_with_trace(context)
        decision = self._normalize(result.decision)
        return DecisionEvaluation(
            context=context,
            public_context=generator_runtime.public_context(context),
            model_tools=generator_runtime.model_tools(context),
            decision=decision,
            model_trace=result.trace,
        )

    def record(
        self,
        state: AgentState,
        *,
        event: AgentEvent,
        decision: GeneratorDecision,
    ) -> None:
        decision_state = self._state(state)
        history = decision_state.setdefault("history", [])
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "event_id": event.event_id,
                "event_type": event.type,
                "task_id": event.task_id,
                "decision_summary": decision.get("decision_summary", ""),
                "command_types": [
                    str(command.get("type") or "")
                    for command in decision.get("commands", [])
                    if isinstance(command, dict)
                ],
                "created_at": utc_now(),
            }
        )
        decision_state["history"] = history[-self.config.history_limit :]
        decision_state["last_event_id"] = event.event_id
        decision_state["last_decision_at"] = utc_now()
        state.workspace.last_decision_summary = str(
            decision.get("decision_summary") or ""
        )

    def record_error(self, state: AgentState, *, event: AgentEvent, exc: Exception) -> None:
        decision_state = self._state(state)
        decision_state["last_error"] = {
            "event_id": event.event_id,
            "type": type(exc).__name__,
            "message": str(exc),
            "created_at": utc_now(),
        }

    def _normalize(self, decision: GeneratorDecision) -> GeneratorDecision:
        normalized = dict(decision)
        commands = normalized.get("commands")
        if not isinstance(commands, list):
            normalized["commands"] = []
        else:
            normalized_commands: list[JsonDict] = []
            for command in commands:
                if not isinstance(command, dict):
                    continue
                normalized_command = dict(command)
                for key in ("args", "condition", "continuation", "patch", "result"):
                    if key in normalized_command:
                        normalized_command[key] = ensure_json_dict(
                            normalized_command[key]
                        )
                normalized_commands.append(normalized_command)
            normalized["commands"] = normalized_commands
        normalized["decision_summary"] = str(
            normalized.get("decision_summary") or ""
        )
        return normalized

    def _state(self, state: AgentState) -> JsonDict:
        decision_state = state.workspace.variables.setdefault(DECISION_VARIABLE_KEY, {})
        if not isinstance(decision_state, dict):
            decision_state = {}
            state.workspace.variables[DECISION_VARIABLE_KEY] = decision_state
        return decision_state

    def _recent_history(self, state: AgentState, *, limit: int) -> list[JsonDict]:
        history = self._state(state).get("history")
        return list(history[-limit:]) if isinstance(history, list) else []
