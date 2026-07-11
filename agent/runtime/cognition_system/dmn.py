from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable, Optional

from ...config import ContextPolicyConfig, DmnConfig
from ...protocols import ActionRun, ActionSpec, AgentEvent, AgentState, AgentTask, GeneratorDecision, JsonDict, utc_now
from ..state_systems.context_policy import (
    CONTEXT_POLICY_VARIABLE_KEY,
    ContextCandidate,
    estimate_tokens,
    select_candidates,
)

if TYPE_CHECKING:
    from ..kernel.generator_runtime import GeneratorRuntime
    from ..state_systems import MemorySystem
    from .emotion_system import EmotionSystem


ACTIVE_TASK_STATES = {"runnable", "running"}
ACTIVE_ACTION_STATES = {"created", "running"}
DMN_VARIABLE_KEY = "dmn"


@dataclass(frozen=True)
class DMNReflection:
    trigger_event: AgentEvent
    context: JsonDict
    public_context: JsonDict
    decision: GeneratorDecision
    model_trace: JsonDict
    emitted_events: list[AgentEvent] = field(default_factory=list)


class DMNSystem:
    """Default-mode cognition: low-frequency reflection while the agent is idle."""

    def __init__(
        self,
        config: DmnConfig,
        *,
        context_policy: Optional[ContextPolicyConfig] = None,
        memory_system: "MemorySystem | None" = None,
        emotion_system: "EmotionSystem | None" = None,
    ) -> None:
        self.config = config
        self.context_policy = context_policy or ContextPolicyConfig.empty()
        self.memory_system = memory_system
        self.emotion_system = emotion_system

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def poll_interval_seconds(self) -> float:
        return max(1.0, min(self.config.interval_seconds, 30.0))

    async def maybe_reflect(
        self,
        *,
        state: AgentState,
        generator_runtime: "GeneratorRuntime",
        action_specs: Iterable[ActionSpec],
    ) -> Optional[DMNReflection]:
        should_reflect, reason = self.should_reflect(state)
        if not should_reflect:
            return None

        trigger_event = AgentEvent.make(
            agent_id=state.agent_id,
            type="dmn.tick",
            source="dmn",
            payload={"reason": reason},
            priority=self.config.thought_priority,
        )
        context = self.build_context(
            state=state,
            trigger_event=trigger_event,
            action_specs=action_specs,
            reason=reason,
        )
        public_context = generator_runtime.public_context(context)
        result = await generator_runtime.generate_with_trace(context)
        emitted_events = self.apply_decision(state, trigger_event, result.decision)
        self._record_reflection(state, result.decision, reason=reason)
        return DMNReflection(
            trigger_event=trigger_event,
            context=context,
            public_context=public_context,
            decision=result.decision,
            model_trace=result.trace,
            emitted_events=emitted_events,
        )

    def should_reflect(self, state: AgentState) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "dmn_disabled"
        if self._has_active_runtime_work(state):
            return False, "runtime_has_active_work"

        now = datetime.now(timezone.utc)
        dmn_state = self._dmn_state(state)
        last_reflection_at = _parse_datetime(dmn_state.get("last_reflection_at"))
        if last_reflection_at is not None:
            since_last = (now - last_reflection_at).total_seconds()
            if since_last < self.config.interval_seconds:
                return False, "dmn_interval_cooldown"

        last_activity_at = self._last_activity_at(state)
        if last_activity_at is not None:
            idle_seconds = (now - last_activity_at).total_seconds()
            if idle_seconds < self.config.idle_after_seconds:
                return False, "runtime_not_idle_long_enough"

        last_state_version = dmn_state.get("last_state_version")
        if last_state_version == state.version and last_reflection_at is not None:
            unchanged_seconds = (now - last_reflection_at).total_seconds()
            if unchanged_seconds < self.config.unchanged_interval_seconds:
                return False, "state_unchanged_quiet_interval"

        return True, "runtime_idle"

    def build_context(
        self,
        *,
        state: AgentState,
        trigger_event: AgentEvent,
        action_specs: Iterable[ActionSpec],
        reason: str,
    ) -> JsonDict:
        action_spec_list = list(action_specs)
        dmn_state = self._dmn_state(state)
        context: JsonDict = {
            "context_kind": "dmn_reflection_pack",
            "agent": state.profile.to_dict(),
            "dmn": {
                "generator_session": "dmn",
                "trigger": {
                    "event_id": trigger_event.event_id,
                    "type": trigger_event.type,
                    "source": trigger_event.source,
                    "payload": trigger_event.payload,
                },
                "reason": reason,
                "last_reflection_at": dmn_state.get("last_reflection_at"),
                "last_summary": dmn_state.get("last_summary"),
                "output_contract": (
                    "Return GeneratorDecision JSON only. Allowed commands: "
                    "think, emit_thought, create_task_suggestion, sleep, no_op. "
                    "Do not call tools and do not write a user-facing reply."
                ),
            },
            "runtime": {
                "mode": (
                    "star_protocol"
                    if any(spec.source == "star_protocol" for spec in action_spec_list)
                    else "local"
                ),
                "idle": not self._has_active_runtime_work(state),
                "stored_counts": {
                    "tasks": len(state.tasks),
                    "action_runs": len(state.action_runs),
                    "transcript_messages": len(state.workspace.transcript),
                    "notes": len(state.workspace.notes),
                    "available_actions": len(action_spec_list),
                },
            },
            "workspace": {
                "workspace_id": state.workspace.workspace_id,
                "current_task_id": state.workspace.current_task_id,
                "last_decision_summary": state.workspace.last_decision_summary,
                "notes": [],
                "transcript": [],
            },
            "cognition": {
                "emotion": (
                    self.emotion_system.context_view(state)
                    if self.emotion_system is not None
                    else state.workspace.variables.get("emotion_state")
                ),
                "long_term_memory": (
                    self.memory_system.search(
                        agent_id=state.agent_id,
                        query=state.workspace.last_decision_summary,
                        limit=4,
                    )
                    if self.memory_system is not None
                    else []
                ),
            },
            "tasks": [],
            "action_runs": [],
            "_generator_session": "dmn",
            "instruction": (
                "DMNSystem runs only when the runtime is idle. Review recent state, notice patterns, "
                "surface useful private thoughts, and emit an agent.thought only when it may help the "
                "DecisionSystem choose future work. Do not execute tools directly."
            ),
        }
        initial_used = estimate_tokens(
            {key: value for key, value in context.items() if not key.startswith("_")},
            self.context_policy,
        )
        candidates: list[ContextCandidate] = []
        context_policy_state = state.workspace.variables.get(CONTEXT_POLICY_VARIABLE_KEY)
        decision_policy_state = (
            context_policy_state.get("decision")
            if isinstance(context_policy_state, dict)
            else None
        )
        summary = (
            decision_policy_state.get("summary")
            if isinstance(decision_policy_state, dict)
            else None
        )
        if isinstance(summary, dict) and summary.get("content"):
            candidates.append(
                ContextCandidate(
                    ref={"type": "context_summary", "id": summary.get("summary_id")},
                    value={
                        "summary_id": summary.get("summary_id"),
                        "content": summary.get("content"),
                        "created_at": summary.get("created_at"),
                    },
                    priority=1000,
                    order=0,
                    reason="decision_context_summary",
                    mandatory=True,
                )
            )
        for index, task in enumerate(state.tasks.values()):
            candidates.append(
                ContextCandidate(
                    ref={"type": "task", "id": task.task_id},
                    value=self._task_view(task),
                    priority=850 if task.status not in {"completed", "failed", "cancelled"} else 350 + index,
                    order=10_000 + index,
                    reason="nonterminal_task" if task.status not in {"completed", "failed", "cancelled"} else "historical_task",
                )
            )
        action_runs = list(state.action_runs.values())
        recent_run_start = max(
            0,
            len(action_runs) - self.context_policy.preferred_recent_action_runs,
        )
        for index, run in enumerate(action_runs):
            candidates.append(
                ContextCandidate(
                    ref={"type": "action_run", "id": run.action_run_id},
                    value=self._action_run_view(run),
                    priority=(
                        900
                        if run.status in ACTIVE_ACTION_STATES
                        else 820 + (index - recent_run_start)
                        if index >= recent_run_start
                        else 300 + index
                    ),
                    order=20_000 + index,
                    reason="active_action_run" if run.status in ACTIVE_ACTION_STATES else "historical_action_run",
                    mandatory=run.status in ACTIVE_ACTION_STATES,
                )
            )
        transcript_start = max(
            0,
            len(state.workspace.transcript)
            - self.context_policy.preferred_recent_transcript_messages,
        )
        for index, message in enumerate(state.workspace.transcript):
            candidates.append(
                ContextCandidate(
                    ref={"type": "transcript", "id": message.get("event_id") or index},
                    value=message,
                    priority=(
                        800 + (index - transcript_start)
                        if index >= transcript_start
                        else 500 + index
                    ),
                    order=30_000 + index,
                    reason="recent_dialogue",
                )
            )
        note_start = max(
            0,
            len(state.workspace.notes) - self.context_policy.preferred_recent_notes,
        )
        for index, note in enumerate(state.workspace.notes):
            candidates.append(
                ContextCandidate(
                    ref={"type": "workspace_note", "id": index},
                    value=note,
                    priority=(
                        790 + (index - note_start)
                        if index >= note_start
                        else 400 + index
                    ),
                    order=40_000 + index,
                    reason="recent_note",
                )
            )
        selection = select_candidates(
            candidates,
            policy=self.context_policy,
            budget_tokens=max(
                1024,
                self.context_policy.target_input_tokens
                - self.context_policy.fixed_prompt_reserve_tokens,
            ),
            initial_used_tokens=initial_used,
        )
        for candidate in selection.selected:
            ref_type = candidate.ref.get("type")
            if ref_type == "context_summary":
                context["workspace"]["context_summary"] = candidate.value
            elif ref_type == "task":
                context["tasks"].append(candidate.value)
            elif ref_type == "action_run":
                context["action_runs"].append(candidate.value)
            elif ref_type == "transcript":
                context["workspace"]["transcript"].append(candidate.value)
            elif ref_type == "workspace_note":
                context["workspace"]["notes"].append(candidate.value)
        context["context_selection"] = {
            "session_id": "dmn",
            "selected_count": len(selection.selected),
            "available_by_reference_count": len(selection.not_selected),
            "estimated_context_tokens": selection.used_tokens,
            "budget_tokens": selection.budget_tokens,
        }
        context["_selection_manifest"] = {
            **selection.manifest,
            "session_id": "dmn",
        }
        return context

    def apply_decision(
        self,
        state: AgentState,
        trigger_event: AgentEvent,
        decision: GeneratorDecision,
    ) -> list[AgentEvent]:
        emitted_events: list[AgentEvent] = []
        for command in decision.get("commands", []):
            ctype = str(command.get("type") or "")
            if ctype in {"think", "note", "remember", "reply"}:
                content = str(command.get("content") or command.get("thought") or "")
                if content:
                    self._record_thought(state, content, kind=ctype)
                continue

            if ctype in {"emit_thought", "create_task_suggestion"}:
                if len(emitted_events) >= self.config.max_events_per_cycle:
                    continue
                content = str(command.get("content") or command.get("thought") or command.get("goal") or "")
                if not content:
                    continue
                payload: JsonDict = {
                    "kind": "task_suggestion" if ctype == "create_task_suggestion" else "thought",
                    "content": content,
                    "intent": command.get("intent"),
                    "urgency": command.get("urgency", "low"),
                    "reason": command.get("reason"),
                    "title": command.get("title"),
                    "goal": command.get("goal"),
                }
                emitted_events.append(
                    AgentEvent.make(
                        agent_id=state.agent_id,
                        type="agent.thought",
                        source="dmn",
                        payload={key: value for key, value in payload.items() if value is not None},
                        causation_id=trigger_event.event_id,
                        priority=self.config.thought_priority,
                    )
                )
                self._record_thought(state, content, kind=payload["kind"])
                continue

            if ctype == "sleep":
                dmn_state = self._dmn_state(state)
                if command.get("seconds") is not None:
                    dmn_state["suggested_sleep_seconds"] = command.get("seconds")
                continue

            if ctype and ctype != "no_op":
                state.workspace.note(f"dmn ignored unsupported command type={ctype}")
        return emitted_events

    def record_error(self, state: AgentState, exc: Exception) -> None:
        dmn_state = self._dmn_state(state)
        dmn_state["last_reflection_at"] = utc_now()
        dmn_state["last_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        state.workspace.note(
            f"dmn reflection error type={type(exc).__name__} message={exc}"
        )

    def _record_reflection(
        self,
        state: AgentState,
        decision: GeneratorDecision,
        *,
        reason: str,
    ) -> None:
        dmn_state = self._dmn_state(state)
        dmn_state["last_reflection_at"] = utc_now()
        dmn_state["last_state_version"] = state.version
        dmn_state["last_reason"] = reason
        dmn_state["last_summary"] = decision.get("decision_summary", "")
        dmn_state["cycle_count"] = int(dmn_state.get("cycle_count", 0)) + 1

    def _record_thought(self, state: AgentState, content: str, *, kind: str) -> None:
        dmn_state = self._dmn_state(state)
        thoughts = dmn_state.setdefault("thoughts", [])
        if not isinstance(thoughts, list):
            thoughts = []
            dmn_state["thoughts"] = thoughts
        thoughts.append(
            {
                "kind": kind,
                "content": content,
                "created_at": utc_now(),
            }
        )
        dmn_state["thoughts"] = thoughts
        state.workspace.note(f"dmn {kind}: {content}")

    def _dmn_state(self, state: AgentState) -> JsonDict:
        dmn_state = state.workspace.variables.setdefault(DMN_VARIABLE_KEY, {})
        if not isinstance(dmn_state, dict):
            dmn_state = {}
            state.workspace.variables[DMN_VARIABLE_KEY] = dmn_state
        return dmn_state

    def _has_active_runtime_work(self, state: AgentState) -> bool:
        return any(
            task.status in ACTIVE_TASK_STATES
            for task in state.tasks.values()
        ) or any(
            run.status in ACTIVE_ACTION_STATES
            for run in state.action_runs.values()
        )

    def _last_activity_at(self, state: AgentState) -> Optional[datetime]:
        candidates = [
            _parse_datetime(state.updated_at),
            _parse_datetime(state.workspace.updated_at),
        ]
        candidates.extend(_parse_datetime(task.updated_at) for task in state.tasks.values())
        for run in state.action_runs.values():
            candidates.extend(
                [
                    _parse_datetime(run.finished_at),
                    _parse_datetime(run.started_at),
                    _parse_datetime(run.created_at),
                ]
            )
        valid = [candidate for candidate in candidates if candidate is not None]
        if not valid:
            return None
        return max(valid)

    def _task_view(self, task: AgentTask) -> JsonDict:
        return {
            "task_id": task.task_id,
            "title": task.title,
            "goal": task.goal,
            "purpose": task.purpose,
            "status": task.status,
            "parent_task_id": task.parent_task_id,
            "child_task_ids": task.child_task_ids,
            "dependencies": task.dependencies,
            "waiting_on": task.waiting_on,
            "scheduling": task.scheduling,
            "progress": task.progress,
            "result": _compact(task.result),
            "error": _compact(task.error),
            "updated_at": task.updated_at,
        }

    def _action_run_view(self, run: ActionRun) -> JsonDict:
        return {
            "action_run_id": run.action_run_id,
            "task_id": run.task_id,
            "action_name": run.action_name,
            "status": run.status,
            "args": _compact(run.args),
            "result": _compact(run.result),
            "error": _compact(run.error),
            "finished_at": run.finished_at,
        }


def _parse_datetime(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _compact(value: object) -> object:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {
            str(key): _compact(item_value)
            for key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return str(value)
