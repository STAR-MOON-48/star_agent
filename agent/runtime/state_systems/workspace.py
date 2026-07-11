from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import re
from typing import Any, Iterable, List, Optional

from ...config import ContextPolicyConfig
from ...protocols import (
    ActionRun,
    ActionSpec,
    AgentEvent,
    AgentState,
    AgentTask,
    JsonDict,
    new_id,
    utc_now,
)
from .context_policy import (
    CONTEXT_POLICY_VARIABLE_KEY,
    ContextCandidate,
    estimate_tokens,
    select_candidates,
    stable_digest,
)


TERMINAL_TASK_STATES = {"completed", "failed", "cancelled"}
TERMINAL_ACTION_STATES = {"succeeded", "failed", "cancelled"}
CONTEXT_RETRIEVAL_ACTION_NAMES = {
    "query_task_status",
    "search_actions",
    "read_task",
    "read_action_run",
    "search_memory",
    "read_memory",
    "search_workspace",
}


class ContextBuilder:
    """Builds the context consumed by Generator.

    In this design the Generator does not directly inspect all runtime internals.
    It receives a selected, explicit context. This makes Workspace the
    working-memory boundary and keeps the Generator capability boundary clear.
    """

    def __init__(self, policy: Optional[ContextPolicyConfig] = None) -> None:
        self.policy = policy or ContextPolicyConfig.empty()

    def build(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        action_specs: Iterable[ActionSpec],
    ) -> JsonDict:
        action_spec_list = list(action_specs)
        focus_task = self._focus_task(state, event)
        focus_runs = self._focus_action_runs(state, event, focus_task)
        candidate_actions = self._select_action_specs(
            action_specs=action_spec_list,
            event=event,
            focus_task=focus_task,
            state=state,
        )
        tool_call_allowed = self._tool_call_allowed(event, focus_task, focus_runs)
        model_tools = (
            [self._action_spec_to_model_tool(spec) for spec in candidate_actions]
            if tool_call_allowed and candidate_actions
            else []
        )
        tool_tokens = estimate_tokens(model_tools, self.policy) if model_tools else 0
        runtime_mode = (
            "star_protocol"
            if any(spec.source == "star_protocol" for spec in action_spec_list)
            else "local"
        )

        base_context: JsonDict = {
            "context_kind": "decision_pack",
            "agent": state.profile.to_dict(),
            "decision": {
                "generator_session": "decision",
                "generator_system": "DecisionSystem",
                "trigger": self._event_view(event),
                "focus_task_id": focus_task.task_id if focus_task else None,
                "next_step_hint": self._next_step_hint(event, focus_task),
                "output_contract": (
                    "Use model-request tools for actions. Return natural language text as a communicative intent for BrocaSystem. "
                    "If you say you will call a tool, make the tool_call in the same response. "
                    "Use internal_runtime tools for local task control. "
                    "Use natural language for speech intent and tool_call for actions. "
                    "Use evidence and selected actions only; do not infer unavailable workspace data."
                ),
            },
            "runtime": {
                "mode": runtime_mode,
                "task_graph": self._task_graph_summary(state),
                "execution_memory": self._execution_memory(focus_runs),
                "action_guidance": {
                    "action_counts": {
                        "candidate": len(candidate_actions),
                        "total": len(action_spec_list),
                        "internal_runtime": sum(
                            1 for spec in action_spec_list if spec.source == "internal_runtime"
                        ),
                        "local_runtime": sum(
                            1
                            for spec in action_spec_list
                            if spec.source not in {"internal_runtime", "star_protocol"}
                        ),
                        "external_environment": sum(1 for spec in action_spec_list if spec.source == "star_protocol"),
                    },
                    "candidate_internal_runtime_action_names": [
                        spec.name for spec in candidate_actions if spec.source == "internal_runtime"
                    ],
                    "candidate_local_action_names": [
                        spec.name
                        for spec in candidate_actions
                        if spec.source not in {"internal_runtime", "star_protocol"}
                    ],
                    "candidate_external_action_names": [
                        spec.name for spec in candidate_actions if spec.source == "star_protocol"
                    ],
                    "notes": [
                        "Full callable action schemas are sent through the model request tools field, not this context body.",
                        "Use only model-request tools as callable action names.",
                        "ActionSystem is the unified execution entry and routes tools by ActionSpec.source.",
                        "Use external_environment_actions for Star Protocol environment state and work.",
                        "Use runtime_* internal_runtime tools only to manage this agent runtime's task state.",
                        "Use local runtime tools for local demo/host capabilities, not for Star Protocol environment state.",
                    ],
                },
            },
            "focus": {
                "task": self._task_decision_view(focus_task) if focus_task else None,
                "action_runs": [],
            },
            "evidence": [],
            "workspace": {
                "workspace_id": state.workspace.workspace_id,
                "current_task_id": state.workspace.current_task_id,
                "last_decision_summary": state.workspace.last_decision_summary,
                "stored_counts": {
                    "tasks": len(state.tasks),
                    "action_runs": len(state.action_runs),
                    "transcript_messages": len(state.workspace.transcript),
                    "notes": len(state.workspace.notes),
                },
                "selected_refs": [],
            },
            "tasks": [],
            "tooling": {
                "candidate_action_names": [spec.name for spec in candidate_actions],
                "candidate_count": len(candidate_actions),
                "total_action_count": len(action_spec_list),
                "schema_transport": "model_request.tools",
                "model_tools_available": tool_call_allowed and bool(candidate_actions),
            },
            "_model_tools": model_tools,
            "_generator_session": "decision",
            "instruction": (
                "Use model-request tools for action calls. Use natural language text for a direct assistant reply. "
                "Do not promise a tool call in text unless the same model response includes that tool call. "
                "For unfinished multi-step Star Protocol work, keep calling the next needed tool instead of only narrating the plan. "
                "Use runtime_create_task/runtime_wait/runtime_update_task/runtime_complete_task/runtime_cancel_task "
                "for local runtime task control through ActionSystem. Do not execute actions directly. "
                "query_task_status only reports internal runtime "
                "state; use star_protocol actions for external environment state. "
                "When action results include both agent_location and location, agent_location is the current agent position; "
                "location may be the inspected or scanned target. "
                "agent.thought is an internal DMN thought, not a user message; do not send a user-facing reply for it. "
                "If it is not actionable, do not call tools and do not send a user-facing reply. "
                "The context is a selected decision pack, not a full workspace dump. "
                "Callable action schemas are provided separately through model request tools."
            ),
        }

        available_tokens = self.policy.available_input_tokens
        target_tokens = self.policy.target_input_tokens
        selection_budget = max(
            1024,
            target_tokens
            - self.policy.fixed_prompt_reserve_tokens
            - tool_tokens,
        )
        initial_used_tokens = estimate_tokens(
            {
                key: value
                for key, value in base_context.items()
                if not key.startswith("_")
            },
            self.policy,
        )
        candidates = self._context_candidates(
            state=state,
            event=event,
            focus_task=focus_task,
            focus_runs=focus_runs,
        )
        selection = select_candidates(
            candidates,
            policy=self.policy,
            budget_tokens=selection_budget,
            initial_used_tokens=initial_used_tokens,
        )

        selected_runs: List[JsonDict] = []
        selected_tasks: List[JsonDict] = []
        evidence: List[JsonDict] = []
        selected_refs: List[JsonDict] = []
        for candidate in selection.selected:
            ref_type = str(candidate.ref.get("type") or "")
            selected_refs.append({**candidate.ref, "reason": candidate.reason})
            if ref_type == "action_run":
                selected_runs.append(candidate.value)
            elif ref_type == "task":
                selected_tasks.append(candidate.value)
            else:
                evidence.append(candidate.value)

        not_selected_counts = Counter(
            str(candidate.ref.get("type") or "unknown")
            for candidate in selection.not_selected
        )
        selected_runs.sort(key=lambda run: str(run.get("created_at") or ""))
        base_context["focus"]["action_runs"] = selected_runs
        base_context["evidence"] = evidence
        base_context["workspace"]["selected_refs"] = selected_refs
        base_context["tasks"] = selected_tasks
        base_context["context_selection"] = {
            "session_id": "decision",
            "policy": {
                "max_context_tokens": self.policy.max_context_tokens,
                "reserve_output_tokens": self.policy.reserve_output_tokens,
                "safety_margin_tokens": self.policy.safety_margin_tokens,
                "available_input_tokens": available_tokens,
                "compaction_trigger_tokens": self.policy.compaction_trigger_tokens,
                "compaction_target_tokens": self.policy.compaction_target_tokens,
                "fixed_prompt_reserve_tokens": self.policy.fixed_prompt_reserve_tokens,
                "tool_budget_tokens": self.policy.tool_budget_tokens,
            },
            "estimated": {
                "tool_tokens": tool_tokens,
                "base_context_tokens": initial_used_tokens,
                "selected_context_tokens": selection.used_tokens,
            },
            "selected_count": len(selection.selected),
            "available_by_reference_count": len(selection.not_selected),
            "available_by_reference_counts": dict(not_selected_counts),
            "summary": self._summary_public_view(state),
        }
        base_context["_selection_manifest"] = {
            **selection.manifest,
            "session_id": "decision",
            "tool_tokens": tool_tokens,
            "selected_tool_names": [spec.name for spec in candidate_actions],
            "available_tool_count": len(action_spec_list),
            "tool_entries": [
                {
                    "ref": {"type": "action_spec", "id": spec.name},
                    "source": spec.source,
                    "selected": spec in candidate_actions,
                    "selection_reason": (
                        "model_request.tools"
                        if spec in candidate_actions
                        else "available_via_search_actions"
                    ),
                }
                for spec in action_spec_list
            ],
        }
        summary_request = self._build_summary_request(
            state=state,
            event=event,
            not_selected=selection.not_selected,
        )
        if summary_request is not None:
            base_context["_summary_request"] = summary_request
            base_context["_selection_manifest"]["summary_request"] = {
                "source_digest": summary_request["summary"]["source_digest"],
                "covered_refs": summary_request["_covered_refs"],
                "estimated_source_tokens": summary_request["summary"][
                    "estimated_source_tokens"
                ],
            }
        return base_context

    def _focus_task(self, state: AgentState, event: AgentEvent) -> Optional[AgentTask]:
        if event.task_id and event.task_id in state.tasks:
            event_task = state.tasks[event.task_id]
            if event_task.status in TERMINAL_TASK_STATES:
                current_task_id = state.workspace.current_task_id
                current = state.tasks.get(current_task_id) if current_task_id else None
                if current is not None and current.status not in TERMINAL_TASK_STATES:
                    return current
                parent = self._nonterminal_parent_task(state, event_task)
                if parent is not None:
                    return parent
            return event_task
        current_task_id = state.workspace.current_task_id
        if current_task_id and current_task_id in state.tasks:
            return state.tasks[current_task_id]
        for task in reversed(list(state.tasks.values())):
            if task.status not in TERMINAL_TASK_STATES:
                return task
        return None

    def _nonterminal_parent_task(
        self,
        state: AgentState,
        task: AgentTask,
    ) -> Optional[AgentTask]:
        if not task.parent_task_id:
            return None
        parent = state.tasks.get(task.parent_task_id)
        if parent is None or parent.status in TERMINAL_TASK_STATES:
            return None
        return parent

    def _focus_action_runs(
        self,
        state: AgentState,
        event: AgentEvent,
        focus_task: Optional[AgentTask],
    ) -> List[ActionRun]:
        selected: List[ActionRun] = []
        if event.action_run_id and event.action_run_id in state.action_runs:
            selected.append(state.action_runs[event.action_run_id])
        if focus_task:
            for run_id in focus_task.active_action_runs:
                run = state.action_runs.get(run_id)
                if run and run not in selected:
                    selected.append(run)
            related = [
                run for run in state.action_runs.values()
                if run.task_id == focus_task.task_id and run not in selected
            ]
            selected.extend(related)
        return selected

    def _context_candidates(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        focus_task: Optional[AgentTask],
        focus_runs: List[ActionRun],
    ) -> List[ContextCandidate]:
        candidates: List[ContextCandidate] = []
        selection_text = self._selection_text(event, focus_task, state)
        focus_task_id = focus_task.task_id if focus_task else None
        related_task_ids = self._related_task_ids(state, focus_task)

        for index, task in enumerate(state.tasks.values()):
            if task.task_id == focus_task_id:
                continue
            relevance = self._text_relevance_score(
                selection_text,
                " ".join(
                    [task.title, task.goal, task.purpose, task.status, str(task.result or "")]
                ),
            )
            if task.task_id in related_task_ids and task.status not in TERMINAL_TASK_STATES:
                priority = 900 + relevance
                reason = "focus_task_graph"
            elif task.task_id in related_task_ids:
                priority = 500 + relevance
                reason = "terminal_focus_task_graph"
            elif task.status not in TERMINAL_TASK_STATES:
                priority = 700 + relevance
                reason = "nonterminal_task"
            else:
                priority = 250 + relevance
                reason = "historical_task"
            candidates.append(
                ContextCandidate(
                    ref={"type": "task", "id": task.task_id},
                    value=self._task_decision_view(task),
                    priority=priority,
                    order=10_000 + index,
                    reason=reason,
                )
            )

        recent_run_ids = {
            run.action_run_id
            for run in sorted(focus_runs, key=lambda item: item.created_at)[
                -self.policy.preferred_recent_action_runs :
            ]
        } if self.policy.preferred_recent_action_runs else set()
        for index, run in enumerate(focus_runs):
            is_trigger_run = run.action_run_id == event.action_run_id
            is_active = run.status in {"created", "running"}
            priority = (
                1000
                if is_trigger_run
                else 950
                if is_active
                else 920 + index
                if run.action_run_id in recent_run_ids
                else 780 + index
            )
            reason = (
                "trigger_action_run"
                if is_trigger_run
                else "active_action_run"
                if is_active
                else "focus_tool_history"
            )
            candidates.append(
                ContextCandidate(
                    ref={"type": "action_run", "id": run.action_run_id},
                    value=self._action_run_decision_view(run, detail=True),
                    priority=priority,
                    order=20_000 + index,
                    reason=reason,
                    mandatory=is_trigger_run or is_active,
                    token_multiplier=2.2 if run.status in TERMINAL_ACTION_STATES else 1.0,
                )
            )

        summary = self._current_summary(state)
        if summary:
            candidates.append(
                ContextCandidate(
                    ref={"type": "context_summary", "id": summary.get("summary_id")},
                    value={
                        "type": "context_summary",
                        "summary_id": summary.get("summary_id"),
                        "content": summary.get("content", ""),
                        "covered_ref_count": len(summary.get("covered_refs") or []),
                        "created_at": summary.get("created_at"),
                    },
                    priority=980,
                    order=30_000,
                    reason="rolling_context_summary",
                    mandatory=True,
                )
            )

        transcript_count = len(state.workspace.transcript)
        recent_transcript_start = max(
            0,
            transcript_count - self.policy.preferred_recent_transcript_messages,
        )
        for index, message in enumerate(state.workspace.transcript):
            if message.get("event_id") == event.event_id:
                continue
            content = str(message.get("content", ""))
            recency = max(0, 300 - (transcript_count - index - 1) * 20)
            relevance = self._text_relevance_score(selection_text, content) * 20
            role_bonus = 80 if message.get("role") == "user" else 40
            candidates.append(
                ContextCandidate(
                    ref={
                        "type": "transcript",
                        "id": message.get("event_id") or message.get("created_at") or index,
                    },
                    value={
                        "type": "transcript",
                        "role": message.get("role"),
                        "content": content,
                        "event_id": message.get("event_id"),
                        "created_at": message.get("created_at"),
                    },
                    priority=(
                        910 + (index - recent_transcript_start)
                        if index >= recent_transcript_start
                        else 400 + recency + relevance + role_bonus
                    ),
                    order=40_000 + index,
                    reason="relevant_recent_dialogue",
                )
            )

        note_count = len(state.workspace.notes)
        recent_note_start = max(0, note_count - self.policy.preferred_recent_notes)
        for index, note in enumerate(state.workspace.notes):
            recency = max(0, 180 - (note_count - index - 1) * 12)
            relevance = self._text_relevance_score(selection_text, note) * 20
            candidates.append(
                ContextCandidate(
                    ref={"type": "workspace_note", "id": index},
                    value={
                        "type": "workspace_note",
                        "note_index": index,
                        "content": note,
                    },
                    priority=(
                        900 + (index - recent_note_start)
                        if index >= recent_note_start
                        else 250 + recency + relevance
                    ),
                    order=50_000 + index,
                    reason="relevant_recent_note",
                )
            )
        return candidates

    def _related_task_ids(
        self,
        state: AgentState,
        focus_task: Optional[AgentTask],
    ) -> set[str]:
        if focus_task is None:
            return set()
        related = set(focus_task.child_task_ids)
        related.update(focus_task.dependencies)
        related.update(
            task.task_id
            for task in state.tasks.values()
            if focus_task.task_id in task.dependencies
        )
        current = focus_task
        seen: set[str] = set()
        while current.parent_task_id and current.parent_task_id in state.tasks:
            if current.task_id in seen:
                break
            seen.add(current.task_id)
            related.add(current.parent_task_id)
            current = state.tasks[current.parent_task_id]
        scheduling = focus_task.scheduling
        for key in (
            "pending_dependency_ids",
            "failed_dependency_ids",
            "nonterminal_child_ids",
            "failed_child_ids",
            "cancelled_child_ids",
        ):
            values = scheduling.get(key)
            if isinstance(values, list):
                related.update(str(value) for value in values)
        return related

    def _text_relevance_score(self, query: str, text: str) -> int:
        if not query or not text:
            return 0
        query_terms = self._relevance_terms(query)
        text_terms = self._relevance_terms(text)
        return len(query_terms & text_terms)

    def _relevance_terms(self, text: str) -> set[str]:
        lowered = text.lower()
        terms = set(re.findall(r"[a-z0-9_]{2,}", lowered))
        cjk_chunks = re.findall(r"[\u4e00-\u9fff]+", lowered)
        for chunk in cjk_chunks:
            if len(chunk) == 1:
                terms.add(chunk)
                continue
            terms.update(chunk[index : index + 2] for index in range(len(chunk) - 1))
        return terms

    def _context_policy_state(self, state: AgentState) -> JsonDict:
        policy_state = state.workspace.variables.setdefault(
            CONTEXT_POLICY_VARIABLE_KEY,
            {},
        )
        if not isinstance(policy_state, dict):
            policy_state = {}
            state.workspace.variables[CONTEXT_POLICY_VARIABLE_KEY] = policy_state
        decision_state = policy_state.setdefault("decision", {})
        if not isinstance(decision_state, dict):
            decision_state = {}
            policy_state["decision"] = decision_state
        return decision_state

    def _current_summary(self, state: AgentState) -> Optional[JsonDict]:
        summary = self._context_policy_state(state).get("summary")
        return summary if isinstance(summary, dict) else None

    def _summary_public_view(self, state: AgentState) -> JsonDict:
        summary = self._current_summary(state)
        if not summary:
            return {"available": False}
        return {
            "available": True,
            "summary_id": summary.get("summary_id"),
            "covered_ref_count": len(summary.get("covered_refs") or []),
            "created_at": summary.get("created_at"),
            "source_digest": summary.get("source_digest"),
        }

    def _build_summary_request(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        not_selected: List[ContextCandidate],
    ) -> Optional[JsonDict]:
        if not self.policy.model_summary_enabled:
            return None
        policy_state = self._context_policy_state(state)
        last_error = policy_state.get("last_error")
        if isinstance(last_error, dict) and self.policy.summary_min_interval_seconds > 0:
            error_at = last_error.get("created_at")
            if isinstance(error_at, str):
                try:
                    parsed_error_at = datetime.fromisoformat(error_at)
                except ValueError:
                    parsed_error_at = None
                if parsed_error_at is not None:
                    if parsed_error_at.tzinfo is None:
                        parsed_error_at = parsed_error_at.replace(tzinfo=timezone.utc)
                    elapsed = (
                        datetime.now(timezone.utc) - parsed_error_at
                    ).total_seconds()
                    if elapsed < self.policy.summary_min_interval_seconds:
                        return None
        summary = self._current_summary(state)
        if summary and self.policy.summary_min_interval_seconds > 0:
            created_at = summary.get("created_at")
            if isinstance(created_at, str):
                try:
                    created = datetime.fromisoformat(created_at)
                except ValueError:
                    created = None
                if created is not None:
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    elapsed = (datetime.now(timezone.utc) - created).total_seconds()
                    if elapsed < self.policy.summary_min_interval_seconds:
                        return None
        covered_ref_keys = {
            stable_digest(ref)
            for ref in (summary.get("covered_refs") or [])
            if isinstance(ref, dict)
        } if summary else set()
        eligible = [
            candidate
            for candidate in not_selected
            if candidate.ref.get("type")
            in {"transcript", "workspace_note", "task", "action_run"}
            and stable_digest(candidate.ref) not in covered_ref_keys
        ]
        total_tokens = sum(candidate.estimated_tokens(self.policy) for candidate in eligible)
        if total_tokens < self.policy.summary_trigger_tokens:
            return None

        previous_summary_tokens = estimate_tokens(summary.get("content", ""), self.policy) if summary else 0
        source_candidates = [
            ContextCandidate(
                ref=candidate.ref,
                value=candidate.value,
                priority=candidate.priority,
                order=-candidate.order,
                reason="context_summary_source",
                token_multiplier=candidate.token_multiplier,
            )
            for candidate in eligible
        ]
        source_selection = select_candidates(
            source_candidates,
            policy=self.policy,
            budget_tokens=self.policy.summary_source_tokens,
            initial_used_tokens=previous_summary_tokens,
        )
        if not source_selection.selected:
            return None
        materials = [
            {"ref": candidate.ref, "content": candidate.value}
            for candidate in source_selection.selected
        ]
        digest_source = {
            "previous_summary_digest": summary.get("source_digest") if summary else None,
            "materials": materials,
        }
        source_digest = stable_digest(digest_source)
        if policy_state.get("last_attempt_digest") == source_digest:
            return None
        covered_refs = list(summary.get("covered_refs") or []) if summary else []
        for candidate in source_selection.selected:
            if candidate.ref not in covered_refs:
                covered_refs.append(candidate.ref)
        return {
            "context_kind": "context_summary_request",
            "agent": state.profile.to_dict(),
            "summary": {
                "source_digest": source_digest,
                "covered_ref_count": len(covered_refs),
                "estimated_source_tokens": source_selection.used_tokens,
                "previous_summary": summary.get("content") if summary else None,
            },
            "trigger": {
                "event_id": event.event_id,
                "type": event.type,
            },
            "materials": materials,
            "instruction": (
                "Merge the previous summary and source materials into a concise Markdown working summary. "
                "Preserve objectives, unresolved constraints, decisions, tool evidence, failures, and exact ids "
                "needed for later retrieval. Do not invent facts and do not output JSON."
            ),
            "_generator_session": "context_builder",
            "_covered_refs": covered_refs,
        }

    def store_summary(
        self,
        state: AgentState,
        request: JsonDict,
        content: str,
    ) -> JsonDict:
        summary_request = request.get("summary")
        if not isinstance(summary_request, dict):
            raise ValueError("Invalid context summary request.")
        record: JsonDict = {
            "summary_id": new_id("ctxsum"),
            "content": content,
            "covered_refs": list(request.get("_covered_refs") or []),
            "source_digest": summary_request.get("source_digest"),
            "created_at": utc_now(),
        }
        policy_state = self._context_policy_state(state)
        policy_state["summary"] = record
        policy_state["last_attempt_digest"] = summary_request.get("source_digest")
        policy_state.pop("last_error", None)
        state.workspace.note(
            f"stored context summary {record['summary_id']} covering "
            f"{len(record['covered_refs'])} references"
        )
        return record

    def record_summary_error(
        self,
        state: AgentState,
        request: JsonDict,
        exc: Exception,
    ) -> None:
        summary_request = request.get("summary")
        source_digest = (
            summary_request.get("source_digest")
            if isinstance(summary_request, dict)
            else None
        )
        policy_state = self._context_policy_state(state)
        policy_state["last_attempt_digest"] = source_digest
        policy_state["last_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "created_at": utc_now(),
        }
        state.workspace.note(
            f"context summary failed digest={source_digest}: {type(exc).__name__}: {exc}"
        )

    def _event_view(self, event: AgentEvent) -> JsonDict:
        return {
            "event_id": event.event_id,
            "type": event.type,
            "source": event.source,
            "task_id": event.task_id,
            "action_run_id": event.action_run_id,
            "payload": self._event_payload_view(event),
        }

    def _event_payload_view(self, event: AgentEvent) -> JsonDict:
        payload = event.payload or {}
        if event.type == "user.message":
            return {"content": str(payload.get("content", ""))}
        if event.type in {"action.completed", "action.failed", "action.cancelled"}:
            return {
                "action_name": payload.get("action_name"),
                "result": self._action_result_view(payload.get("result")),
                "error": self._compact_value(payload.get("error")),
                "reason": payload.get("reason"),
            }
        if event.type == "action.progress":
            return {
                "action_name": payload.get("action_name"),
                "progress": self._compact_value(payload.get("progress", payload)),
            }
        if event.type == "protocol.tool_specification":
            tools = payload.get("tools")
            tool_count = len(tools) if isinstance(tools, list) else None
            return {"tool_count": tool_count, "summary": self._compact_value(payload)}
        return self._compact_value(payload)

    def _task_decision_view(self, task: AgentTask) -> JsonDict:
        return {
            "task_id": task.task_id,
            "title": task.title,
            "goal": task.goal,
            "purpose": task.purpose,
            "status": task.status,
            "parent_task_id": task.parent_task_id,
            "child_task_ids": task.child_task_ids,
            "dependencies": task.dependencies,
            "active_action_runs": task.active_action_runs,
            "waiting_on": task.waiting_on,
            "scheduling": self._compact_value(task.scheduling),
            "progress": task.progress,
            "result": self._compact_value(task.result),
            "error": self._compact_value(task.error),
            "continuation": self._compact_value(task.continuation),
        }

    def _action_run_decision_view(self, run: ActionRun, *, detail: bool = False) -> JsonDict:
        view: JsonDict = {
            "action_run_id": run.action_run_id,
            "task_id": run.task_id,
            "action_name": run.action_name,
            "mode": run.mode,
            "source": run.source,
            "status": run.status,
            "progress": self._compact_value(run.progress),
            "created_at": run.created_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
        }
        if detail:
            view["args"] = self._compact_value(run.args)
        if detail or run.status in {"succeeded", "failed", "cancelled"}:
            view["result"] = self._action_result_view(run.result)
            view["error"] = self._compact_value(run.error)
        return view

    def _next_step_hint(
        self,
        event: AgentEvent,
        focus_task: Optional[AgentTask],
    ) -> str:
        if event.type == "user.message":
            if focus_task is None:
                return (
                    "No focus task exists. Reply directly if no action is needed; "
                    "otherwise call a selected tool or use runtime_create_task for a durable local task."
                )
            return "Decide whether to reply directly, call a selected tool, or manage the local task with runtime_* tools."
        if event.type == "conversation.decision.requested":
            return (
                "Use Wernicke's attributed understanding to decide facts, actions, commitments, "
                "or a communicative intent. Natural-language output is passed to BrocaSystem, not sent directly."
            )
        if event.type == "runtime.continue":
            if focus_task:
                classification = str(focus_task.scheduling.get("classification") or "")
                if classification == "child_requires_resolution":
                    return "Resolve or replan the blocked child tasks before requesting parent completion."
                if classification == "review_child_outcomes":
                    return "Review failed/cancelled child outcomes, recover or explicitly acknowledge them, then decide completion."
            return "Continue the runnable focus task with the next action or request guarded completion."
        if event.type == "agent.thought":
            return "Evaluate this internal thought; act only if it is useful, otherwise return no commands."
        if event.type == "action.completed":
            return "Use the action result evidence to reply and complete/update the focus task."
        if event.type == "action.failed":
            return "Use the action error evidence to retry, fail, or ask for help."
        if event.type == "protocol.tool_specification":
            return "Use newly available external action candidates if the current task needs environment work."
        if focus_task and focus_task.status == "waiting":
            return "Check whether the wait condition is satisfied; otherwise keep waiting."
        return "Produce the next runtime command that advances or closes the focus task."

    def _execution_memory(self, focus_runs: List[ActionRun]) -> JsonDict:
        ordered = sorted(focus_runs, key=lambda run: run.created_at)
        recent_limit = max(
            8,
            min(16, self.policy.preferred_recent_action_runs * 3),
        )
        recent_attempts: List[JsonDict] = []
        repeated: dict[str, JsonDict] = {}
        for run in ordered:
            signature = stable_digest(
                {
                    "action_name": run.action_name,
                    "args": run.args,
                    "status": run.status,
                    "error": run.error,
                }
            )
            group = repeated.setdefault(
                signature,
                {
                    "action_name": run.action_name,
                    "args": self._compact_value(run.args),
                    "status": run.status,
                    "error": self._compact_value(run.error),
                    "count": 0,
                    "last_action_run_id": run.action_run_id,
                },
            )
            group["count"] = int(group["count"]) + 1
            group["last_action_run_id"] = run.action_run_id

        for run in ordered[-recent_limit:]:
            attempt: JsonDict = {
                "action_run_id": run.action_run_id,
                "action_name": run.action_name,
                "status": run.status,
                "args": self._compact_value(run.args),
            }
            if run.status == "succeeded":
                attempt["result"] = self._action_result_memory_view(run.result)
            elif run.error:
                attempt["error"] = self._compact_value(run.error)
            recent_attempts.append(attempt)

        return {
            "recent_attempts": recent_attempts,
            "repeated_attempts": [
                group
                for group in repeated.values()
                if int(group.get("count", 0)) > 1
            ],
        }

    def _action_result_memory_view(self, result: Any) -> Any:
        if not isinstance(result, dict):
            return self._compact_value(result)
        view: JsonDict = {}
        noisy_sequence_keys = {
            "activity_log",
            "events",
            "history",
            "logs",
            "messages",
            "shared_observations",
            "signals",
        }
        for key, value in result.items():
            if key in noisy_sequence_keys:
                continue
            if value is None or isinstance(value, (str, int, float, bool)):
                view[str(key)] = value
                continue
            if isinstance(value, list) and len(value) <= 20 and all(
                item is None or isinstance(item, (str, int, float, bool))
                for item in value
            ):
                view[str(key)] = list(value)
                continue
            if isinstance(value, dict) and len(value) <= 20 and all(
                item is None or isinstance(item, (str, int, float, bool))
                for item in value.values()
            ):
                view[str(key)] = dict(value)
        agent_location = self._agent_location_from_result(result)
        if agent_location:
            view["agent_location"] = agent_location
        return view

    def _task_graph_summary(self, state: AgentState) -> JsonDict:
        status_counts: JsonDict = {}
        classification_counts: JsonDict = {}
        for task in state.tasks.values():
            status_counts[task.status] = int(status_counts.get(task.status, 0)) + 1
            classification = str(task.scheduling.get("classification") or "unknown")
            classification_counts[classification] = (
                int(classification_counts.get(classification, 0)) + 1
            )
        return {
            "root_task_count": sum(
                1 for task in state.tasks.values() if not task.parent_task_id
            ),
            "runnable_task_count": sum(
                1
                for task in state.tasks.values()
                if task.status == "runnable" and task.scheduling.get("can_run")
            ),
            "waiting_task_count": sum(
                1 for task in state.tasks.values() if task.status == "waiting"
            ),
            "blocked_task_count": sum(
                1 for task in state.tasks.values() if task.status == "blocked"
            ),
            "terminal_task_count": sum(
                1
                for task in state.tasks.values()
                if task.status in TERMINAL_TASK_STATES
            ),
            "status_counts": status_counts,
            "classification_counts": classification_counts,
        }

    def _tool_call_allowed(
        self,
        event: AgentEvent,
        focus_task: Optional[AgentTask],
        focus_runs: List[ActionRun],
    ) -> bool:
        if event.type in {"action.completed", "action.failed", "action.cancelled"}:
            if focus_task is None or focus_task.status in TERMINAL_TASK_STATES:
                return False
            has_active_run = any(run.status in {"created", "running"} for run in focus_runs)
            return not has_active_run and not focus_task.waiting_on
        if event.type == "runtime.continue":
            if focus_task is None:
                return True
            has_active_run = any(run.status in {"created", "running"} for run in focus_runs)
            return not has_active_run and not focus_task.waiting_on
        if event.type == "agent.thought":
            has_active_run = any(run.status in {"created", "running"} for run in focus_runs)
            return not has_active_run
        if event.type == "protocol.tool_specification":
            return focus_task is not None
        if event.type in {"protocol.event", "protocol.action", "timer.fired"}:
            return True
        if event.type == "user.message":
            has_active_run = any(run.status in {"created", "running"} for run in focus_runs)
            return not has_active_run
        if event.type == "conversation.decision.requested":
            has_active_run = any(run.status in {"created", "running"} for run in focus_runs)
            return not has_active_run
        return False

    def _select_action_specs(
        self,
        *,
        action_specs: List[ActionSpec],
        event: AgentEvent,
        focus_task: Optional[AgentTask],
        state: AgentState,
    ) -> List[ActionSpec]:
        if not action_specs:
            return []

        text = self._selection_text(event, focus_task, state)
        scored = [
            (self._action_score(spec, text, event), index, spec)
            for index, spec in enumerate(action_specs)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        score_by_name = {spec.name: score for score, _, spec in scored}

        mandatory_specs = [
            spec
            for spec in action_specs
            if spec.source == "internal_runtime"
            or spec.name in CONTEXT_RETRIEVAL_ACTION_NAMES
        ]
        local_specs = [
            spec
            for score, _, spec in scored
            if spec.source not in {"internal_runtime", "star_protocol"}
            and spec.name not in CONTEXT_RETRIEVAL_ACTION_NAMES
            and (score > 0 or not any(item.source == "star_protocol" for item in action_specs))
        ]
        external_ranked = [
            spec for _, _, spec in scored if spec.source == "star_protocol"
        ]
        external_positive = [
            spec for spec in external_ranked if score_by_name.get(spec.name, 0) > 0
        ]
        external_discovery = [
            spec
            for spec in external_ranked
            if self._looks_like_discovery_action(spec)
            and spec not in external_positive
        ]
        external_fallback = [
            spec
            for spec in external_ranked
            if spec not in external_positive and spec not in external_discovery
        ]
        external_specs = (
            external_positive + external_discovery + external_fallback
        )[: self.policy.max_external_tools]

        ordered_specs: List[ActionSpec] = []
        for spec in mandatory_specs + local_specs + external_specs:
            if spec not in ordered_specs:
                ordered_specs.append(spec)

        complete_catalog_fits = (
            len(external_ranked) <= self.policy.max_external_tools
            and estimate_tokens(
                [self._action_spec_to_model_tool(spec) for spec in ordered_specs],
                self.policy,
            )
            <= self.policy.tool_budget_tokens
        )
        if complete_catalog_fits:
            return ordered_specs

        tool_candidates = [
            ContextCandidate(
                ref={"type": "action_spec", "id": spec.name},
                value=self._action_spec_to_model_tool(spec),
                priority=(
                    1000
                    if spec in mandatory_specs
                    else 800 + score_by_name.get(spec.name, 0)
                    if spec.source == "star_protocol"
                    else 700 + score_by_name.get(spec.name, 0)
                ),
                order=index,
                reason=(
                    "runtime_or_retrieval_tool"
                    if spec in mandatory_specs
                    else "relevant_external_tool"
                    if spec.source == "star_protocol"
                    else "relevant_local_tool"
                ),
                mandatory=spec in mandatory_specs,
            )
            for index, spec in enumerate(ordered_specs)
        ]
        selection = select_candidates(
            tool_candidates,
            policy=self.policy,
            budget_tokens=self.policy.tool_budget_tokens,
        )
        selected_names = {
            str(candidate.ref.get("id")) for candidate in selection.selected
        }
        return [spec for spec in ordered_specs if spec.name in selected_names]

    def _looks_like_discovery_action(self, spec: ActionSpec) -> bool:
        text = f"{spec.name} {spec.description}".lower()
        return any(
            marker in text
            for marker in (
                "list",
                "search",
                "discover",
                "inspect",
                "observe",
                "status",
                "scan",
                "查看",
                "列出",
                "搜索",
                "发现",
                "观察",
                "状态",
            )
        )

    def _selection_text(
        self,
        event: AgentEvent,
        focus_task: Optional[AgentTask],
        state: AgentState,
    ) -> str:
        parts = [
            event.type,
            str(event.payload),
            str(event.payload.get("content", "")),
            str(event.payload.get("action_name", "")),
            str(event.payload.get("result", "")),
            str(event.payload.get("error", "")),
            state.workspace.last_decision_summary,
        ]
        if focus_task:
            parts.extend([focus_task.title, focus_task.goal, focus_task.purpose, focus_task.status])
            parts.extend(
                str(run.result or run.error or "")
                for run in self._focus_action_runs(state, event, focus_task)
            )
        return " ".join(part for part in parts if part).lower()

    def _action_score(self, spec: ActionSpec, text: str, event: AgentEvent) -> int:
        target = f"{spec.name} {spec.description}".lower()
        score = 0
        if spec.name.lower() in text:
            score += 20
        for token in set(self._tokens(text)):
            if len(token) >= 3 and token in target:
                score += 1
        if self._status_intent(text) and any(word in target for word in ["status", "progress", "状态", "进度"]):
            score += 4
        if self._analysis_intent(text) and any(word in target for word in ["analysis", "analyze", "分析", "报告"]):
            score += 4
        if (
            spec.source == "star_protocol"
            and any(word in text for word in ["star", "environment", "external", "环境", "外部"])
        ):
            score += 3
        if (
            spec.source == "star_protocol"
            and any(word in target for word in ["observe", "inspect", "scan", "list", "task", "environment"])
        ):
            score += 2
        if spec.name == "query_task_status" and spec.source != "star_protocol" and event.source == "star_protocol":
            score -= 6
        return score

    def _tokens(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def _status_intent(self, text: str) -> bool:
        return any(word in text for word in ["status", "progress", "状态", "进度", "running", "wait"])

    def _analysis_intent(self, text: str) -> bool:
        return any(word in text for word in ["analysis", "analyze", "report", "分析", "报告"])

    def _action_spec_to_model_tool(self, spec: ActionSpec) -> JsonDict:
        parameters = dict(spec.input_schema or {})
        if not parameters:
            parameters = {"type": "object", "properties": {}}
        elif "type" not in parameters:
            parameters = {"type": "object", "properties": parameters}
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": parameters,
            },
        }

    def _action_result_view(self, result: Any) -> Any:
        if not isinstance(result, dict):
            return self._compact_value(result)
        view: JsonDict = {
            str(key): self._compact_value(value)
            for key, value in result.items()
        }
        agent_location = self._agent_location_from_result(result)
        if agent_location:
            view["agent_location"] = agent_location
        if agent_location and result.get("location") and result.get("location") != agent_location:
            view["location_note"] = (
                "agent_location is the current agent position; "
                "location may be the inspected or scanned target."
            )
        return view

    def _agent_location_from_result(self, result: JsonDict) -> Optional[str]:
        agent_id = result.get("agent_id")
        agents = result.get("agents")
        if isinstance(agent_id, str) and isinstance(agents, list):
            for agent in agents:
                if not isinstance(agent, dict) or agent.get("agent_id") != agent_id:
                    continue
                location = agent.get("location")
                if isinstance(location, str) and location:
                    return location
        return None

    def _compact_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return {
                str(key): self._compact_value(item_value)
                for key, item_value in value.items()
            }
        if isinstance(value, list):
            return [self._compact_value(item) for item in value]
        return str(value)
