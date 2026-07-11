from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from ...protocols import (
    ActionRun,
    ActionSpec,
    AgentEvent,
    AgentState,
    AgentTask,
    JsonDict,
    ensure_json_dict,
    new_id,
    utc_now,
)
from ..console import trace_line
from ..interfaces.protocol import ProtocolInterface
from ..kernel.event_bus import EventBus
from .task_system import MULTI_STEP_OBJECTIVE_PURPOSE, TaskSystem

if TYPE_CHECKING:
    from ..state_systems import MemorySystem


INTERNAL_RUNTIME_ACTION_TO_COMMAND_TYPE = {
    "runtime_create_task": "create_task",
    "runtime_wait": "wait",
    "runtime_update_task": "update_task",
    "runtime_complete_task": "complete_task",
    "runtime_cancel_task": "cancel_task",
}
UNFINISHED_EXTERNAL_STATUS_VALUES = {
    "in_progress",
    "in progress",
    "running",
    "pending",
    "open",
    "unfinished",
    "not_done",
    "not done",
}


def action_idempotency_key(
    *,
    task_id: str,
    action_name: str,
    args: JsonDict,
) -> str:
    """Return a deterministic identity for one logical task action."""

    canonical_payload = json.dumps(
        {
            "task_id": task_id,
            "action_name": action_name,
            "args": args,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


class ActionRegistry:
    """Action capability registry exposed to Generator."""

    def __init__(self) -> None:
        self._specs: Dict[str, ActionSpec] = {
            "project_analysis": ActionSpec(
                name="project_analysis",
                description="Local long-running project analysis. Emits progress events and then a report.",
                mode="async",
                timeout_ms=60_000,
                cancelable=True,
                side_effect_level="read",
                input_schema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "depth": {"type": "string", "enum": ["quick", "normal", "deep"]},
                    },
                    "required": ["target"],
                },
            ),
            "query_task_status": ActionSpec(
                name="query_task_status",
                description=(
                    "Synchronously summarize this agent runtime's internal task and "
                    "action-run statuses. This does not inspect any external Star "
                    "Protocol environment."
                ),
                mode="sync",
                timeout_ms=1000,
                cancelable=False,
                side_effect_level="read",
                input_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "offset": {"type": "integer", "minimum": 0},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "additionalProperties": False,
                },
                metadata={"bypass_task_readiness": True, "context_retrieval": True},
            ),
            "search_actions": ActionSpec(
                name="search_actions",
                description=(
                    "Search the complete local and Star Protocol action catalog. Use this when the "
                    "needed environment tool is not present in the current model-request tools. "
                    "Results are explicitly paginated and include full ActionSpec metadata."
                ),
                mode="sync",
                timeout_ms=1000,
                cancelable=False,
                side_effect_level="read",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "source": {"type": "string"},
                        "offset": {"type": "integer", "minimum": 0},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                metadata={"bypass_task_readiness": True, "context_retrieval": True},
            ),
            "read_task": ActionSpec(
                name="read_task",
                description=(
                    "Read one exact persisted Agent task by task_id. Optionally include descendants "
                    "and action runs. Use ids from context selection, task scheduling, or search results."
                ),
                mode="sync",
                timeout_ms=1000,
                cancelable=False,
                side_effect_level="read",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "include_descendants": {"type": "boolean"},
                        "include_action_runs": {"type": "boolean"},
                    },
                    "required": ["task_id"],
                    "additionalProperties": False,
                },
                metadata={"bypass_task_readiness": True, "context_retrieval": True},
            ),
            "read_action_run": ActionSpec(
                name="read_action_run",
                description="Read one exact persisted ActionRun by action_run_id.",
                mode="sync",
                timeout_ms=1000,
                cancelable=False,
                side_effect_level="read",
                input_schema={
                    "type": "object",
                    "properties": {"action_run_id": {"type": "string"}},
                    "required": ["action_run_id"],
                    "additionalProperties": False,
                },
                metadata={"bypass_task_readiness": True, "context_retrieval": True},
            ),
            "search_memory": ActionSpec(
                name="search_memory",
                description=(
                    "Search durable episodic and semantic Markdown memories. Results include "
                    "confidence and source references; use read_memory for exact content."
                ),
                mode="sync",
                timeout_ms=1000,
                cancelable=False,
                side_effect_level="read",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "kind": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                metadata={"bypass_task_readiness": True, "context_retrieval": True},
            ),
            "read_memory": ActionSpec(
                name="read_memory",
                description="Read one exact durable Markdown memory by memory_id.",
                mode="sync",
                timeout_ms=1000,
                cancelable=False,
                side_effect_level="read",
                input_schema={
                    "type": "object",
                    "properties": {"memory_id": {"type": "string"}},
                    "required": ["memory_id"],
                    "additionalProperties": False,
                },
                metadata={"bypass_task_readiness": True, "context_retrieval": True},
            ),
            "search_workspace": ActionSpec(
                name="search_workspace",
                description=(
                    "Search exact persisted transcript, notes, tasks, and action runs. Results are "
                    "explicitly paginated and retain references for targeted follow-up reads."
                ),
                mode="sync",
                timeout_ms=1000,
                cancelable=False,
                side_effect_level="read",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "kinds": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["transcript", "note", "task", "action_run"],
                            },
                        },
                        "offset": {"type": "integer", "minimum": 0},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                metadata={"bypass_task_readiness": True, "context_retrieval": True},
            ),
            "runtime_create_task": ActionSpec(
                name="runtime_create_task",
                description=(
                    "Internal runtime tool. Create a local Agent task when the model needs "
                    "a durable task boundary before future actions."
                ),
                mode="sync",
                timeout_ms=1000,
                cancelable=False,
                side_effect_level="write",
                source="internal_runtime",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_ref": {"type": "string"},
                        "title": {"type": "string"},
                        "goal": {"type": "string"},
                        "purpose": {"type": "string"},
                        "parent_task_id": {"type": "string"},
                        "dependencies": {"type": "array", "items": {"type": "string"}},
                        "continuation": {"type": "object"},
                    },
                    "required": ["title", "goal"],
                    "additionalProperties": False,
                },
            ),
            "runtime_wait": ActionSpec(
                name="runtime_wait",
                description=(
                    "Internal runtime tool. Mark a local Agent task as waiting on an "
                    "event/action condition instead of starting a new external action."
                ),
                mode="sync",
                timeout_ms=1000,
                cancelable=False,
                side_effect_level="write",
                source="internal_runtime",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "task_ref": {"type": "string"},
                        "condition": {"type": "object"},
                    },
                    "required": ["condition"],
                    "additionalProperties": False,
                },
            ),
            "runtime_update_task": ActionSpec(
                name="runtime_update_task",
                description=(
                    "Internal runtime tool. Patch local Agent task metadata, "
                    "progress, result, error, or continuation."
                ),
                mode="sync",
                timeout_ms=1000,
                cancelable=False,
                side_effect_level="write",
                source="internal_runtime",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "task_ref": {"type": "string"},
                        "patch": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "goal": {"type": "string"},
                                "purpose": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["created", "runnable", "waiting", "blocked"],
                                },
                                "dependencies": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "progress": {"type": "object"},
                                "result": {"type": "object"},
                                "error": {"type": "object"},
                                "continuation": {"type": "object"},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "required": ["patch"],
                    "additionalProperties": False,
                },
            ),
            "runtime_complete_task": ActionSpec(
                name="runtime_complete_task",
                description=(
                    "Internal runtime tool. Request completion of the focused local Agent task. "
                    "TaskSystem defers completion while dependencies, waits, actions, or descendant "
                    "tasks remain unresolved. Failed/cancelled descendants must be explicitly "
                    "acknowledged in the result after recovery or replanning."
                ),
                mode="sync",
                timeout_ms=1000,
                cancelable=False,
                side_effect_level="write",
                source="internal_runtime",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "task_ref": {"type": "string"},
                        "result": {"type": "object"},
                    },
                    "additionalProperties": False,
                },
            ),
            "runtime_cancel_task": ActionSpec(
                name="runtime_cancel_task",
                description="Internal runtime tool. Cancel the focused local Agent task and any active local action runs.",
                mode="sync",
                timeout_ms=1000,
                cancelable=False,
                side_effect_level="write",
                source="internal_runtime",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "task_ref": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            ),
        }

    def register(self, spec: ActionSpec) -> None:
        self._specs[spec.name] = spec

    def register_many(self, specs: Iterable[ActionSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def get(self, name: str) -> ActionSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown action: {name}")
        return self._specs[name]

    def list_specs(self) -> Iterable[ActionSpec]:
        return list(self._specs.values())


class ActionExecutor:
    """Executes sync actions immediately and async actions through workers."""

    def __init__(
        self,
        event_bus: EventBus,
        registry: ActionRegistry,
        *,
        protocol_interface: Optional[ProtocolInterface] = None,
        task_system: Optional[TaskSystem] = None,
        memory_system: Optional["MemorySystem"] = None,
        trace: bool = True,
    ) -> None:
        self.event_bus = event_bus
        self.registry = registry
        self.protocol_interface = protocol_interface
        self.task_system = task_system or TaskSystem()
        self.memory_system = memory_system
        self.trace = trace
        self._async_workers: Dict[str, asyncio.Task] = {}

    async def start_action(
        self,
        *,
        state: AgentState,
        task_id: Optional[str],
        action_name: str,
        args: JsonDict,
        mode_hint: Optional[str] = None,
        causation_id: Optional[str] = None,
        causation_event: Optional[AgentEvent] = None,
    ) -> List[AgentEvent]:
        args = ensure_json_dict(args)
        spec = self.registry.get(action_name)
        if spec.source == "internal_runtime":
            return await self._execute_internal_runtime_action(
                state=state,
                task_id=task_id,
                action_name=action_name,
                args=args,
                causation_id=causation_id,
                causation_event=causation_event,
            )
        mode = mode_hint or spec.mode
        if mode not in {"sync", "async"}:
            raise ValueError(f"MVP supports only sync/async actions, got: {mode}")
        if task_id is None:
            raise KeyError(f"Action {action_name} requires a task_id.")

        task = state.tasks[task_id]
        idempotency_key = action_idempotency_key(
            task_id=task_id,
            action_name=action_name,
            args=args,
        )
        matching_runs = self._matching_action_runs(
            state=state,
            task_id=task_id,
            action_name=action_name,
            args=args,
            idempotency_key=idempotency_key,
        )
        succeeded = next(
            (run for run in matching_runs if run.status == "succeeded"),
            None,
        )
        if succeeded is not None and self._replay_succeeded_action(spec):
            if self.trace:
                trace_line(
                    "action.executor",
                    "replay persisted success "
                    f"source=[cyan]{spec.source}[/cyan] "
                    f"action=[bold]{action_name}[/bold] "
                    f"task=[magenta]{task_id}[/magenta] "
                    f"run=[magenta]{succeeded.action_run_id}[/magenta]",
                )
            return [
                AgentEvent.make(
                    agent_id=state.agent_id,
                    type="action.completed",
                    source="action_executor",
                    task_id=task_id,
                    action_run_id=succeeded.action_run_id,
                    causation_id=causation_id,
                    idempotency_key=idempotency_key,
                    payload={
                        "action_name": action_name,
                        "result": ensure_json_dict(succeeded.result),
                        "deduplicated": True,
                    },
                )
            ]

        active = next(
            (run for run in matching_runs if run.status in {"created", "running"}),
            None,
        )
        if active is not None:
            if self.trace:
                trace_line(
                    "action.executor",
                    "skip duplicate active "
                    f"source=[cyan]{spec.source}[/cyan] "
                    f"action=[bold]{action_name}[/bold] "
                    f"task=[magenta]{task_id}[/magenta] "
                    f"run=[magenta]{active.action_run_id}[/magenta]",
                )
            return []

        scheduling_blockers = (
            []
            if spec.metadata.get("bypass_task_readiness")
            else self.task_system.action_start_blockers(state, task_id)
        )
        if scheduling_blockers:
            action_run_id = new_id("run")
            run = ActionRun(
                action_run_id=action_run_id,
                agent_id=state.agent_id,
                task_id=task_id,
                action_name=action_name,
                args=args,
                mode=mode,
                source=spec.source,
                status="created",
                idempotency_key=idempotency_key,
            )
            state.action_runs[action_run_id] = run
            error = {
                "type": "task_not_runnable",
                "message": "TaskSystem rejected action start because task constraints are unresolved.",
                "blockers": scheduling_blockers,
            }
            if self.trace:
                trace_line(
                    "action.executor",
                    "reject start "
                    f"action=[bold]{action_name}[/bold] "
                    f"task=[magenta]{task_id}[/magenta] "
                    f"blockers={json.dumps(scheduling_blockers, ensure_ascii=False)}",
                )
            return [
                AgentEvent.make(
                    agent_id=state.agent_id,
                    type="action.failed",
                    source="action_executor",
                    task_id=task_id,
                    action_run_id=action_run_id,
                    causation_id=causation_id,
                    payload={"action_name": action_name, "error": error},
                )
            ]
        action_run_id = new_id("run")
        run = ActionRun(
            action_run_id=action_run_id,
            agent_id=state.agent_id,
            task_id=task_id,
            action_name=action_name,
            args=args,
            mode=mode,
            source=spec.source,
            status="created",
            idempotency_key=idempotency_key,
        )
        state.action_runs[action_run_id] = run

        if action_run_id not in task.active_action_runs:
            task.active_action_runs.append(action_run_id)
        task.status = "running" if mode == "sync" else "waiting"
        task.touch()

        started = AgentEvent.make(
            agent_id=state.agent_id,
            type="action.started",
            source="action_executor",
            task_id=task_id,
            action_run_id=action_run_id,
            causation_id=causation_id,
            payload={"action_name": action_name, "mode": mode},
        )

        if self.trace:
            trace_line(
                "action.executor",
                "start "
                f"source=[cyan]{spec.source}[/cyan] mode=[cyan]{mode}[/cyan] "
                f"action=[bold]{action_name}[/bold] "
                f"task=[magenta]{task_id}[/magenta] "
                f"run=[magenta]{action_run_id}[/magenta]",
            )

        if spec.source == "star_protocol":
            if self.protocol_interface is None:
                raise RuntimeError(f"Action {action_name} requires a ProtocolInterface.")
            external_action_id = await self.protocol_interface.send_action(
                agent_id=state.agent_id,
                task_id=task_id,
                action_run_id=action_run_id,
                action_name=action_name,
                args=args,
                target=spec.target,
                causation_id=started.event_id,
            )
            run.progress = {"external_action_id": external_action_id}
            return [started]

        if mode == "sync":
            result = self._execute_sync(
                state=state,
                task_id=task_id,
                action_run_id=action_run_id,
                action_name=action_name,
                args=args,
            )
            completed = AgentEvent.make(
                agent_id=state.agent_id,
                type="action.completed",
                source="action_executor",
                task_id=task_id,
                action_run_id=action_run_id,
                causation_id=started.event_id,
                payload={"action_name": action_name, "result": result},
            )
            return [started, completed]

        worker = asyncio.create_task(
            self._run_async_worker(
                agent_id=state.agent_id,
                task_id=task_id,
                action_run_id=action_run_id,
                action_name=action_name,
                args=args,
                causation_id=started.event_id,
            )
        )
        self._async_workers[action_run_id] = worker
        return [started]

    def _matching_action_runs(
        self,
        *,
        state: AgentState,
        task_id: str,
        action_name: str,
        args: JsonDict,
        idempotency_key: str,
    ) -> List[ActionRun]:
        matches: List[ActionRun] = []
        for run in reversed(list(state.action_runs.values())):
            exact_key_match = run.idempotency_key == idempotency_key
            legacy_match = (
                run.task_id == task_id
                and run.action_name == action_name
                and run.args == args
            )
            if not exact_key_match and not legacy_match:
                continue
            if legacy_match and run.idempotency_key != idempotency_key:
                # Migrate records written by the old process-randomized hash scheme.
                run.idempotency_key = idempotency_key
            matches.append(run)
        return matches

    def _replay_succeeded_action(self, spec: ActionSpec) -> bool:
        configured = spec.metadata.get("replay_succeeded")
        if configured is not None:
            return bool(configured)
        # Read actions may legitimately return different data later. Successful
        # side-effecting actions are replayed to avoid repeating the side effect.
        return spec.side_effect_level != "read"

    async def _execute_internal_runtime_action(
        self,
        *,
        state: AgentState,
        task_id: Optional[str],
        action_name: str,
        args: JsonDict,
        causation_id: Optional[str],
        causation_event: Optional[AgentEvent],
    ) -> List[AgentEvent]:
        command_type = INTERNAL_RUNTIME_ACTION_TO_COMMAND_TYPE.get(action_name)
        if command_type is None:
            raise KeyError(f"Unknown internal runtime action: {action_name}")

        target_task_id, result = await self._apply_internal_runtime_command(
            state=state,
            task_id=task_id,
            command_type=command_type,
            args=args,
            causation_event=causation_event,
        )
        run = self._record_internal_action_run(
            state=state,
            task_id=target_task_id,
            action_name=action_name,
            args=args,
            result=result,
        )
        if self.trace:
            trace_line(
                "action.executor",
                "internal completed "
                f"action=[bold]{action_name}[/bold] "
                f"task=[magenta]{target_task_id}[/magenta] "
                f"run=[magenta]{run.action_run_id}[/magenta]",
            )
        return [
            AgentEvent.make(
                agent_id=state.agent_id,
                type="action.internal.completed",
                source="action_executor",
                task_id=target_task_id,
                action_run_id=run.action_run_id,
                causation_id=causation_id,
                payload={
                    "action_name": action_name,
                    "internal_command_type": command_type,
                    "result": result,
                    "deferred": bool(result.get("deferred")),
                },
            )
        ]

    async def _apply_internal_runtime_command(
        self,
        *,
        state: AgentState,
        task_id: Optional[str],
        command_type: str,
        args: JsonDict,
        causation_event: Optional[AgentEvent],
    ) -> tuple[str, JsonDict]:
        if command_type == "create_task":
            continuation = self._maybe_parse_json_value(args.get("continuation") or {})
            if not isinstance(continuation, dict):
                continuation = {}
            dependencies = args.get("dependencies") or []
            if not isinstance(dependencies, list):
                dependencies = [str(dependencies)]
            task = self.task_system.create_task(
                state,
                title=str(args.get("title", "Untitled task")),
                goal=str(args.get("goal", "")),
                purpose=str(
                    args.get(
                        "purpose",
                        "Runtime-created task through internal runtime action.",
                    )
                ),
                parent_task_id=self._parent_task_id_for_internal_create(
                    state,
                    task_id=task_id,
                    args=args,
                ),
                dependencies=[str(item) for item in dependencies],
                continuation=continuation,
            )
            result: JsonDict = {
                "task_id": task.task_id,
                "title": task.title,
                "status": task.status,
            }
            task_ref = args.get("task_ref")
            if isinstance(task_ref, str) and task_ref:
                result["task_ref"] = task_ref
            return task.task_id, result

        target_task_id = self._resolve_internal_task_id(state, task_id, args)

        if command_type == "wait":
            condition = self._maybe_parse_json_value(args.get("condition", {}))
            if not isinstance(condition, dict):
                condition = {"value": condition}
            self.task_system.add_wait(state, target_task_id, condition)
            return target_task_id, {
                "task_id": target_task_id,
                "status": state.tasks[target_task_id].status,
                "condition": condition,
            }

        if command_type == "update_task":
            patch = self._maybe_parse_json_value(args.get("patch", {}))
            if not isinstance(patch, dict):
                patch = {"progress": {"value": patch}}
            update_result = self.task_system.update_task(state, target_task_id, patch)
            return target_task_id, {
                "task_id": target_task_id,
                "status": state.tasks[target_task_id].status,
                "patch": patch,
                **update_result,
            }

        if command_type == "complete_task":
            result = self._maybe_parse_json_value(args.get("result"))
            if self._should_defer_internal_completion(
                state=state,
                task_id=target_task_id,
                result=result,
                causation_event=causation_event,
            ):
                task = state.tasks[target_task_id]
                reason = "external evidence still indicates unfinished work"
                blockers = [{"kind": "unfinished_external_evidence"}]
                deferred = self.task_system.defer_completion(
                    state,
                    target_task_id,
                    reason=reason,
                    blockers=blockers,
                )
                return target_task_id, {
                    "task_id": target_task_id,
                    "status": task.status,
                    "deferred": True,
                    "reason": reason,
                    "blockers": blockers,
                    "attempt_count": deferred["attempt_count"],
                    "result": result,
                }
            task_result = result if isinstance(result, dict) else None
            if result is not None and not isinstance(result, dict):
                task_result = {"value": result}
            completion = self.task_system.complete_task(
                state,
                target_task_id,
                task_result,
            )
            return target_task_id, {**completion, "result": result}

        if command_type == "cancel_task":
            reason = str(args.get("reason", "cancelled"))
            cancel_events = await self.cancel_action_runs(
                state=state,
                task_id=target_task_id,
                reason=reason,
                silent=True,
                include_descendants=True,
            )
            for event in cancel_events:
                self.task_system.apply_event(state, event)
            self.task_system.cancel_task(state, target_task_id, reason)
            return target_task_id, {
                "task_id": target_task_id,
                "status": state.tasks[target_task_id].status,
                "reason": reason,
                "cancelled_action_run_ids": [
                    event.action_run_id for event in cancel_events if event.action_run_id
                ],
            }

        raise KeyError(f"Unknown internal runtime command: {command_type}")

    def _record_internal_action_run(
        self,
        *,
        state: AgentState,
        task_id: str,
        action_name: str,
        args: JsonDict,
        result: JsonDict,
    ) -> ActionRun:
        action_run_id = new_id("run")
        now = utc_now()
        run = ActionRun(
            action_run_id=action_run_id,
            agent_id=state.agent_id,
            task_id=task_id,
            action_name=action_name,
            args=args,
            mode="sync",
            source="internal_runtime",
            status="succeeded",
            result=result,
            started_at=now,
            finished_at=now,
            idempotency_key=action_idempotency_key(
                task_id=task_id,
                action_name=action_name,
                args=args,
            ),
        )
        state.action_runs[action_run_id] = run
        return run

    def _resolve_internal_task_id(
        self,
        state: AgentState,
        task_id: Optional[str],
        args: JsonDict,
    ) -> str:
        explicit_task_id = args.get("task_id")
        if isinstance(explicit_task_id, str) and explicit_task_id:
            if explicit_task_id not in state.tasks:
                raise KeyError(f"Unknown task_id: {explicit_task_id}")
            return explicit_task_id

        task_ref = args.get("task_ref")
        if isinstance(task_ref, str) and task_ref in state.tasks:
            return task_ref

        if task_id:
            if task_id not in state.tasks:
                raise KeyError(f"Unknown task_id: {task_id}")
            return task_id

        current_task_id = state.workspace.current_task_id
        if current_task_id and current_task_id in state.tasks:
            return current_task_id

        raise KeyError("Internal runtime action requires an existing task.")

    def _parent_task_id_for_internal_create(
        self,
        state: AgentState,
        *,
        task_id: Optional[str],
        args: JsonDict,
    ) -> Optional[str]:
        explicit_parent_id = args.get("parent_task_id")
        if isinstance(explicit_parent_id, str) and explicit_parent_id in state.tasks:
            return explicit_parent_id

        if task_id and task_id in state.tasks:
            task = state.tasks[task_id]
            if self._is_multi_step_objective_task(task):
                return task.task_id

        current_task_id = state.workspace.current_task_id
        current = state.tasks.get(current_task_id) if current_task_id else None
        if current and self._is_multi_step_objective_task(current):
            return current.task_id
        return None

    def _should_defer_internal_completion(
        self,
        *,
        state: AgentState,
        task_id: str,
        result: Any,
        causation_event: Optional[AgentEvent],
    ) -> bool:
        task = state.tasks.get(task_id)
        if task is None or not self._is_multi_step_objective_task(task):
            return False
        evidence_values: list[Any] = [result]
        if causation_event is not None:
            evidence_values.append(causation_event.payload)
            if isinstance(causation_event.payload, dict):
                evidence_values.append(causation_event.payload.get("result"))
        return any(
            self._contains_unfinished_external_status(value)
            for value in evidence_values
        )

    def _contains_unfinished_external_status(self, value: Any, *, depth: int = 0) -> bool:
        if value is None or depth > 4:
            return False
        if isinstance(value, str):
            return value.strip().lower() in UNFINISHED_EXTERNAL_STATUS_VALUES
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key).lower()
                if (
                    key_text in {"task_status", "status", "state", "phase"}
                    and isinstance(item, str)
                    and item.strip().lower() in UNFINISHED_EXTERNAL_STATUS_VALUES
                ):
                    return True
                if key_text in {"done", "completed", "finished"} and item is False:
                    return True
                if self._contains_unfinished_external_status(item, depth=depth + 1):
                    return True
            return False
        if isinstance(value, list):
            return any(
                self._contains_unfinished_external_status(item, depth=depth + 1)
                for item in value
            )
        return False

    def _is_multi_step_objective_task(self, task: AgentTask) -> bool:
        return (
            task.purpose == MULTI_STEP_OBJECTIVE_PURPOSE
            or task.continuation.get("kind") == "multi_step_objective"
        )

    def _maybe_parse_json_value(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped or stripped[0] not in "{[":
            return value
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value

    async def cancel_action_runs(
        self,
        *,
        state: AgentState,
        task_id: str,
        reason: str,
        silent: bool = True,
        include_descendants: bool = False,
    ) -> List[AgentEvent]:
        events: List[AgentEvent] = []
        task = state.tasks.get(task_id)
        if not task:
            return events

        tasks = [task]
        if include_descendants:
            tasks.extend(self.task_system.descendants(state, task_id))
        for target_task in tasks:
            for action_run_id in list(target_task.active_action_runs):
                run = state.action_runs.get(action_run_id)
                if not run or run.status in {"succeeded", "failed", "cancelled"}:
                    continue
                worker = self._async_workers.pop(action_run_id, None)
                if worker and not worker.done():
                    worker.cancel()
                events.append(
                    AgentEvent.make(
                        agent_id=state.agent_id,
                        type="action.cancelled",
                        source="action_executor",
                        task_id=target_task.task_id,
                        action_run_id=action_run_id,
                        payload={"reason": reason, "silent": silent},
                    )
                )
        return events

    def _execute_sync(
        self,
        *,
        state: AgentState,
        task_id: str,
        action_run_id: str,
        action_name: str,
        args: JsonDict,
    ) -> JsonDict:
        if action_name == "query_task_status":
            self.task_system.reconcile(state)
            tasks = []
            status_filter = str(args.get("status") or "").strip()
            for task in state.tasks.values():
                # Do not let the introspection task dominate its own answer.
                if task.task_id == task_id:
                    continue
                if status_filter and task.status != status_filter:
                    continue
                tasks.append(
                    {
                        "task_id": task.task_id,
                        "title": task.title,
                        "status": task.status,
                        "parent_task_id": task.parent_task_id,
                        "child_task_ids": task.child_task_ids,
                        "dependencies": task.dependencies,
                        "scheduling": task.scheduling,
                        "progress": task.progress,
                        "active_action_runs": task.active_action_runs,
                        "waiting_on": task.waiting_on,
                    }
                )
            page = self._paginated_result(tasks, args)
            return {
                "summary": "当前任务状态已读取。",
                **{key: value for key, value in page.items() if key != "items"},
                "tasks": page["items"],
            }
        if action_name == "search_actions":
            query = str(args.get("query") or "").strip().lower()
            source = str(args.get("source") or "").strip()
            specs = []
            for spec in self.registry.list_specs():
                if source and spec.source != source:
                    continue
                searchable = json.dumps(spec.to_dict(), ensure_ascii=False).lower()
                if query and not self._searchable_matches(query, searchable):
                    continue
                specs.append(spec.to_dict())
            return {
                "summary": "Action catalog search completed.",
                "query": query,
                "source": source or None,
                **self._paginated_result(specs, args),
            }
        if action_name == "read_task":
            target_task_id = str(args.get("task_id") or "")
            target = state.tasks.get(target_task_id)
            if target is None:
                raise KeyError(f"Unknown task_id: {target_task_id}")
            tasks = [target]
            if bool(args.get("include_descendants")):
                tasks.extend(self.task_system.descendants(state, target_task_id))
            result: JsonDict = {
                "summary": "Persisted task record read.",
                "tasks": [item.to_dict() for item in tasks],
            }
            if bool(args.get("include_action_runs")):
                selected_task_ids = {item.task_id for item in tasks}
                result["action_runs"] = [
                    run.to_dict()
                    for run in state.action_runs.values()
                    if run.task_id in selected_task_ids
                ]
            return result
        if action_name == "read_action_run":
            target_run_id = str(args.get("action_run_id") or "")
            run = state.action_runs.get(target_run_id)
            if run is None:
                raise KeyError(f"Unknown action_run_id: {target_run_id}")
            return {
                "summary": "Persisted action run record read.",
                "action_run": run.to_dict(),
            }
        if action_name == "search_memory":
            if self.memory_system is None:
                return {"summary": "MemorySystem unavailable.", "results": []}
            raw_tags = args.get("tags")
            tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else None
            results = self.memory_system.search(
                agent_id=state.agent_id,
                query=str(args.get("query") or ""),
                kind=str(args.get("kind") or "") or None,
                tags=tags,
                limit=max(1, min(50, int(args.get("limit", 10)))),
            )
            return {
                "summary": "Durable memory search completed.",
                "results": results,
                "count": len(results),
            }
        if action_name == "read_memory":
            if self.memory_system is None:
                return {"summary": "MemorySystem unavailable.", "memory": None}
            memory_id = str(args.get("memory_id") or "")
            record = self.memory_system.read(
                agent_id=state.agent_id,
                memory_id=memory_id,
            )
            if record is None:
                raise KeyError(f"Unknown memory_id: {memory_id}")
            return {"summary": "Durable memory read completed.", "memory": record}
        if action_name == "search_workspace":
            query = str(args.get("query") or "").strip().lower()
            raw_kinds = args.get("kinds")
            kinds = (
                {str(kind) for kind in raw_kinds}
                if isinstance(raw_kinds, list) and raw_kinds
                else {"transcript", "note", "task", "action_run"}
            )
            matches: List[JsonDict] = []
            if "transcript" in kinds:
                for index, message in enumerate(state.workspace.transcript):
                    if self._workspace_value_matches(query, message):
                        matches.append(
                            {
                                "ref": {
                                    "type": "transcript",
                                    "id": message.get("event_id") or index,
                                },
                                "content": message,
                            }
                        )
            if "note" in kinds:
                for index, note in enumerate(state.workspace.notes):
                    if self._workspace_value_matches(query, note):
                        matches.append(
                            {
                                "ref": {"type": "workspace_note", "id": index},
                                "content": note,
                            }
                        )
            if "task" in kinds:
                for task in state.tasks.values():
                    value = task.to_dict()
                    if self._workspace_value_matches(query, value):
                        matches.append(
                            {
                                "ref": {"type": "task", "id": task.task_id},
                                "content": value,
                            }
                        )
            if "action_run" in kinds:
                for run in state.action_runs.values():
                    value = run.to_dict()
                    if self._workspace_value_matches(query, value):
                        matches.append(
                            {
                                "ref": {
                                    "type": "action_run",
                                    "id": run.action_run_id,
                                },
                                "content": value,
                            }
                        )
            return {
                "summary": "Workspace search completed.",
                "query": query,
                "kinds": sorted(kinds),
                **self._paginated_result(matches, args),
            }
        raise KeyError(f"No sync implementation for action: {action_name}")

    def _paginated_result(self, items: List[Any], args: JsonDict) -> JsonDict:
        offset = max(0, int(args.get("offset", 0)))
        limit = min(100, max(1, int(args.get("limit", 20))))
        page = items[offset : offset + limit]
        return {
            "total": len(items),
            "offset": offset,
            "limit": limit,
            "count": len(page),
            "has_more": offset + len(page) < len(items),
            "next_offset": offset + len(page) if offset + len(page) < len(items) else None,
            "items": page,
        }

    def _workspace_value_matches(self, query: str, value: Any) -> bool:
        if not query:
            return True
        searchable = json.dumps(value, ensure_ascii=False).lower()
        return self._searchable_matches(query, searchable)

    def _searchable_matches(self, query: str, searchable: str) -> bool:
        if query in searchable:
            return True
        terms = re.findall(r"[a-z0-9_]{2,}", query.lower())
        for chunk in re.findall(r"[\u4e00-\u9fff]+", query):
            terms.extend(
                chunk[index : index + 2]
                for index in range(max(1, len(chunk) - 1))
            )
        unique_terms = list(dict.fromkeys(term for term in terms if term))
        if not unique_terms:
            return False
        matched = sum(1 for term in unique_terms if term in searchable)
        required = len(unique_terms) if len(unique_terms) <= 3 else max(2, len(unique_terms) // 2)
        return matched >= required

    async def _run_async_worker(
        self,
        *,
        agent_id: str,
        task_id: str,
        action_run_id: str,
        action_name: str,
        args: JsonDict,
        causation_id: str,
    ) -> None:
        try:
            if action_name != "project_analysis":
                raise KeyError(f"No async implementation for action: {action_name}")

            target = args.get("target", "unknown project")
            depth = str(args.get("depth", "normal"))
            step_delay = {"quick": 0.35, "normal": 0.8, "deep": 1.2}.get(depth, 0.8)
            steps = [
                (20, "读取项目结构"),
                (45, "扫描依赖和配置"),
                (70, "执行静态分析"),
                (90, "整理发现并生成报告"),
            ]
            for percent, message in steps:
                await asyncio.sleep(step_delay)
                progress_event = AgentEvent.make(
                    agent_id=agent_id,
                    type="action.progress",
                    source="local_tool_worker",
                    task_id=task_id,
                    action_run_id=action_run_id,
                    causation_id=causation_id,
                    payload={
                        "action_name": action_name,
                        "progress": {"percent": percent, "message": message, "target": target},
                    },
                )
                await self.event_bus.publish(progress_event)

            await asyncio.sleep(0.3)
            result = {
                "target": target,
                "summary": "分析完成：发现 2 个中风险问题，0 个高风险问题。",
                "findings": [
                    {
                        "level": "medium",
                        "title": "依赖版本偏旧",
                        "recommendation": "升级 demo-lib 到 2.x，并增加锁文件审计。",
                    },
                    {
                        "level": "medium",
                        "title": "缺少 CI 安全扫描步骤",
                        "recommendation": "在 CI pipeline 中加入依赖扫描和静态分析。",
                    },
                ],
                "generated_at": utc_now(),
            }
            completed_event = AgentEvent.make(
                agent_id=agent_id,
                type="action.completed",
                source="local_tool_worker",
                task_id=task_id,
                action_run_id=action_run_id,
                causation_id=causation_id,
                payload={"action_name": action_name, "result": result},
            )
            await self.event_bus.publish(completed_event)
        except asyncio.CancelledError:
            if self.trace:
                trace_line(
                    "action.executor",
                    f"async worker cancelled run=[magenta]{action_run_id}[/magenta]",
                )
            raise
        except Exception as exc:  # Keep the demo resilient.
            failed_event = AgentEvent.make(
                agent_id=agent_id,
                type="action.failed",
                source="local_tool_worker",
                task_id=task_id,
                action_run_id=action_run_id,
                causation_id=causation_id,
                payload={"action_name": action_name, "error": {"message": str(exc)}},
            )
            await self.event_bus.publish(failed_event)
