from __future__ import annotations

from typing import Any, List, Optional

from ...protocols import AgentEvent, AgentState, AgentTask, JsonDict, new_id, utc_now


TERMINAL_TASK_STATES = {"completed", "failed", "cancelled"}
TERMINAL_ACTION_STATES = {"succeeded", "failed", "cancelled"}
MULTI_STEP_OBJECTIVE_PURPOSE = "Runtime task created for a multi-step model-request objective."


class TaskSystem:
    """Own task truth, task-tree readiness, and lifecycle transitions."""

    def apply_event(self, state: AgentState, event: AgentEvent) -> None:
        if event.type == "user.message":
            content = str(event.payload.get("content", ""))
            state.workspace.add_transcript("user", content, event_id=event.event_id)
        elif event.type == "action.started":
            self._apply_action_started(state, event)
        elif event.type == "action.progress":
            self._apply_action_progress(state, event)
        elif event.type == "action.completed":
            self._apply_action_completed(state, event)
        elif event.type == "action.failed":
            self._apply_action_failed(state, event)
        elif event.type == "action.cancelled":
            self._apply_action_cancelled(state, event)
        elif event.type == "timer.fired":
            if event.task_id and event.task_id in state.tasks:
                task = state.tasks[event.task_id]
                if task.status not in TERMINAL_TASK_STATES:
                    task.status = "runnable"
                    task.scheduling.pop("status_override", None)
                    task.scheduling.pop("status_override_reason", None)
                task.progress["timer"] = event.payload
                task.touch()
        self.reconcile(state)

    def create_task(
        self,
        state: AgentState,
        *,
        title: str,
        goal: str,
        purpose: str,
        parent_task_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        continuation: Optional[JsonDict] = None,
    ) -> AgentTask:
        task = AgentTask(
            task_id=new_id("task"),
            agent_id=state.agent_id,
            title=title,
            goal=goal,
            purpose=purpose,
            status="runnable",
            parent_task_id=parent_task_id,
            dependencies=self._unique_ids(dependencies or []),
            workspace_ref=state.workspace.workspace_id,
            continuation=continuation or {},
        )
        state.tasks[task.task_id] = task
        if parent_task_id and parent_task_id in state.tasks:
            parent = state.tasks[parent_task_id]
            if task.task_id not in parent.child_task_ids:
                parent.child_task_ids.append(task.task_id)
                parent.touch()
        state.workspace.current_task_id = task.task_id
        state.workspace.note(f"created task {task.task_id}: {title}")
        self.reconcile(state)
        return task

    def add_wait(self, state: AgentState, task_id: str, condition: JsonDict) -> None:
        task = state.tasks[task_id]
        if self._wait_condition_satisfied(state, condition):
            if task.status not in TERMINAL_TASK_STATES and not task.active_action_runs:
                task.status = "runnable"
                task.scheduling.pop("status_override", None)
                task.scheduling.pop("status_override_reason", None)
            task.touch()
        elif condition not in task.waiting_on:
            task.waiting_on.append(condition)
            if task.status not in TERMINAL_TASK_STATES:
                task.status = "waiting"
            task.touch()
        self.reconcile(state)

    def update_task(self, state: AgentState, task_id: str, patch: JsonDict) -> JsonDict:
        task = state.tasks[task_id]
        allowed = {
            "title",
            "goal",
            "purpose",
            "status",
            "dependencies",
            "progress",
            "result",
            "error",
            "continuation",
        }
        applied: JsonDict = {}
        rejected: JsonDict = {}
        for key, value in patch.items():
            if key not in allowed:
                rejected[key] = "field is not mutable through update_task"
                continue
            if key == "status":
                status = str(value)
                if status in TERMINAL_TASK_STATES or status == "running":
                    rejected[key] = (
                        "terminal states require complete_task/cancel_task; running is owned by ActionSystem"
                    )
                    continue
                if status not in {"created", "runnable", "waiting", "blocked"}:
                    rejected[key] = f"unsupported task status: {status}"
                    continue
                task.status = status
                if status in {"waiting", "blocked"}:
                    task.scheduling["status_override"] = status
                    task.scheduling["status_override_reason"] = "runtime_update_task"
                else:
                    task.scheduling.pop("status_override", None)
                    task.scheduling.pop("status_override_reason", None)
                applied[key] = status
                continue
            if key == "dependencies":
                if not isinstance(value, list):
                    rejected[key] = "dependencies must be a list of task ids"
                    continue
                task.dependencies = self._unique_ids(value)
                applied[key] = list(task.dependencies)
                continue
            setattr(task, key, value)
            applied[key] = value
        if applied or rejected:
            task.touch()
        self.reconcile(state)
        return {"applied": applied, "rejected": rejected}

    def complete_task(
        self,
        state: AgentState,
        task_id: str,
        result: Optional[JsonDict] = None,
    ) -> JsonDict:
        self.reconcile(state)
        task = state.tasks[task_id]
        if task.status == "completed":
            return {
                "task_id": task_id,
                "status": task.status,
                "completed": True,
                "deferred": False,
                "already_completed": True,
                "blockers": [],
            }
        blockers = self._completion_blockers(state, task, result=result)
        if blockers:
            self.defer_completion(
                state,
                task_id,
                reason="task tree is not ready for completion",
                blockers=blockers,
            )
            return {
                "task_id": task_id,
                "status": task.status,
                "completed": False,
                "deferred": True,
                "reason": "task tree is not ready for completion",
                "blockers": blockers,
            }

        task.status = "completed"
        task.result = result if result is not None else task.result
        task.waiting_on.clear()
        task.active_action_runs.clear()
        task.progress.pop("completion_deferred", None)
        task.progress["percent"] = 100
        task.scheduling.pop("status_override", None)
        task.scheduling.pop("status_override_reason", None)
        task.touch()
        state.workspace.note(f"completed task {task_id}: {task.title}")
        self._resume_parent_after_child_terminal(state, task, reason="child_task_completed")
        self.reconcile(state)
        return {
            "task_id": task_id,
            "status": task.status,
            "completed": True,
            "deferred": False,
            "blockers": [],
        }

    def defer_completion(
        self,
        state: AgentState,
        task_id: str,
        *,
        reason: str,
        blockers: List[JsonDict],
    ) -> JsonDict:
        task = state.tasks[task_id]
        previous = task.progress.get("completion_deferred")
        attempt_count = 1
        if (
            isinstance(previous, dict)
            and previous.get("reason") == reason
            and previous.get("blockers") == blockers
        ):
            attempt_count = int(previous.get("attempt_count", 0)) + 1
        record: JsonDict = {
            "reason": reason,
            "attempt_count": attempt_count,
            "blockers": blockers,
        }
        task.progress["completion_deferred"] = record
        task.touch()
        state.workspace.note(
            f"deferred completion for task {task_id}; reason={reason}; blockers={blockers}"
        )
        self.reconcile(state)
        return record

    def mark_stalled(
        self,
        state: AgentState,
        task_id: str,
        *,
        reason: str,
        decision_summary: str,
    ) -> None:
        task = state.tasks[task_id]
        previous = task.progress.get("scheduler_stall")
        count = 1
        if isinstance(previous, dict) and previous.get("reason") == reason:
            count = int(previous.get("count", 0)) + 1
        task.progress["scheduler_stall"] = {
            "reason": reason,
            "count": count,
            "decision_summary": decision_summary,
        }
        task.status = "blocked"
        task.scheduling["status_override"] = "blocked"
        task.scheduling["status_override_reason"] = "decision_stalled"
        task.touch()
        state.workspace.note(
            f"task {task_id} stalled after a runtime.continue decision: {reason}"
        )
        self.reconcile(state)

    def cancel_task(self, state: AgentState, task_id: str, reason: str) -> None:
        task = state.tasks[task_id]
        targets = sorted(
            [*self.descendants(state, task_id), task],
            key=lambda item: self.task_depth(state, item.task_id),
            reverse=True,
        )
        for target in targets:
            if target.status in TERMINAL_TASK_STATES:
                continue
            target.status = "cancelled"
            target.error = {"reason": reason, "cancelled_at": utc_now()}
            target.waiting_on.clear()
            target.active_action_runs.clear()
            target.scheduling.pop("status_override", None)
            target.scheduling.pop("status_override_reason", None)
            target.touch()
            state.workspace.note(f"cancelled task {target.task_id}: {reason}")
        self._resume_parent_after_child_terminal(
            state,
            task,
            reason="child_task_cancelled",
            error=task.error,
        )
        self.reconcile(state)

    def reconcile(self, state: AgentState) -> None:
        """Recompute operational status from actions, waits, dependencies, and children."""

        self._repair_parent_links(state)
        for task in state.tasks.values():
            task.dependencies = self._unique_ids(task.dependencies)
            task.child_task_ids = self._unique_ids(task.child_task_ids)
            self._repair_multi_step_action_state(state, task)
            active_run_ids = [
                run_id
                for run_id in task.active_action_runs
                if run_id not in state.action_runs
                or state.action_runs[run_id].status not in TERMINAL_ACTION_STATES
            ]
            waits = [
                condition
                for condition in task.waiting_on
                if not self._wait_condition_satisfied(state, condition)
            ]
            if active_run_ids != task.active_action_runs or waits != task.waiting_on:
                task.active_action_runs = active_run_ids
                task.waiting_on = waits
                task.touch()

        for task in state.tasks.values():
            if task.status != "completed":
                continue
            blockers = self._completion_blockers(state, task, result=task.result)
            if not blockers:
                continue
            task.status = "runnable"
            task.progress["reopened_by_task_system"] = {
                "reason": "completed task still had unresolved subtree constraints",
                "blockers": blockers,
            }
            task.touch()
            state.workspace.note(
                f"reopened inconsistent completed task {task.task_id}; blockers={blockers}"
            )

        ordered = sorted(
            state.tasks.values(),
            key=lambda item: self.task_depth(state, item.task_id),
            reverse=True,
        )
        for task in ordered:
            desired_status, scheduling = self._derive_task_scheduling(state, task)
            if task.status != desired_status or task.scheduling != scheduling:
                task.status = desired_status
                task.scheduling = scheduling
                task.touch()
        self._refresh_current_task(state)

    def active_or_current_task(self, state: AgentState) -> Optional[AgentTask]:
        self.reconcile(state)
        next_task_id = self.next_runnable_task_id(
            state,
            preferred_task_id=state.workspace.current_task_id,
            reconcile=False,
        )
        if next_task_id:
            return state.tasks[next_task_id]
        current_id = state.workspace.current_task_id
        if current_id and current_id in state.tasks:
            current = state.tasks[current_id]
            if current.status not in TERMINAL_TASK_STATES:
                return current
        for task in reversed(list(state.tasks.values())):
            if task.status not in TERMINAL_TASK_STATES:
                return task
        return None

    def next_runnable_task_id(
        self,
        state: AgentState,
        *,
        preferred_task_id: Optional[str] = None,
        reconcile: bool = True,
    ) -> Optional[str]:
        if reconcile:
            self.reconcile(state)
        candidates = [
            task
            for task in state.tasks.values()
            if task.status == "runnable"
            and bool(task.scheduling.get("can_run"))
            and not task.active_action_runs
            and not task.waiting_on
        ]
        if not candidates:
            return None
        if preferred_task_id and preferred_task_id in state.tasks:
            preferred_ids = {
                preferred_task_id,
                *[task.task_id for task in self.descendants(state, preferred_task_id)],
            }
            preferred = [task for task in candidates if task.task_id in preferred_ids]
            if preferred:
                candidates = preferred
        insertion_order = {task_id: index for index, task_id in enumerate(state.tasks)}
        selected = max(
            candidates,
            key=lambda task: (
                self.task_depth(state, task.task_id),
                insertion_order.get(task.task_id, 0),
            ),
        )
        return selected.task_id

    def root_task_id(self, state: AgentState, task_id: str) -> Optional[str]:
        task = state.tasks.get(task_id)
        if task is None:
            return None
        seen: set[str] = set()
        while task.parent_task_id and task.parent_task_id in state.tasks:
            if task.task_id in seen:
                return task.task_id
            seen.add(task.task_id)
            task = state.tasks[task.parent_task_id]
        return task.task_id

    def task_depth(self, state: AgentState, task_id: str) -> int:
        depth = 0
        task = state.tasks.get(task_id)
        seen: set[str] = set()
        while task and task.parent_task_id and task.parent_task_id in state.tasks:
            if task.task_id in seen:
                break
            seen.add(task.task_id)
            depth += 1
            task = state.tasks.get(task.parent_task_id)
        return depth

    def descendants(self, state: AgentState, task_id: str) -> List[AgentTask]:
        selected: List[AgentTask] = []
        seen = {task_id}
        pending = [task_id]
        while pending:
            parent_id = pending.pop(0)
            for child in self._children(state, parent_id):
                if child.task_id in seen:
                    continue
                seen.add(child.task_id)
                selected.append(child)
                pending.append(child.task_id)
        return selected

    def completion_blockers(
        self,
        state: AgentState,
        task_id: str,
        *,
        result: Optional[JsonDict] = None,
    ) -> List[JsonDict]:
        self.reconcile(state)
        return self._completion_blockers(state, state.tasks[task_id], result=result)

    def action_start_blockers(self, state: AgentState, task_id: str) -> List[JsonDict]:
        self.reconcile(state)
        task = state.tasks[task_id]
        if task.status in TERMINAL_TASK_STATES:
            return [{"kind": "terminal_task_state", "status": task.status}]
        scheduling = task.scheduling
        recovery_action_allowed = (
            bool(scheduling.get("can_run"))
            or scheduling.get("status_override_reason") == "decision_stalled"
        )
        blockers: List[JsonDict] = []
        for field, kind in (
            ("missing_dependency_ids", "missing_dependencies"),
            ("cyclic_dependency_ids", "cyclic_dependencies"),
            ("failed_dependency_ids", "failed_dependencies"),
            ("pending_dependency_ids", "pending_dependencies"),
            ("missing_child_ids", "missing_children"),
            ("nonterminal_child_ids", "nonterminal_children"),
            ("missing_action_run_ids", "missing_action_runs"),
        ):
            values = scheduling.get(field)
            if field == "nonterminal_child_ids" and recovery_action_allowed:
                continue
            if isinstance(values, list) and values:
                blockers.append({"kind": kind, "ids": values})
        if task.waiting_on:
            blockers.append({"kind": "unresolved_waits", "conditions": task.waiting_on})
        if (
            scheduling.get("status_override") in {"waiting", "blocked"}
            and scheduling.get("status_override_reason") != "decision_stalled"
        ):
            blockers.append(
                {
                    "kind": "explicit_status_override",
                    "status": scheduling.get("status_override"),
                }
            )
        return blockers

    def _derive_task_scheduling(
        self,
        state: AgentState,
        task: AgentTask,
    ) -> tuple[str, JsonDict]:
        children = self._children(state, task.task_id)
        child_ids = [child.task_id for child in children]
        nonterminal_children = [
            child.task_id for child in children if child.status not in TERMINAL_TASK_STATES
        ]
        completed_children = [child.task_id for child in children if child.status == "completed"]
        failed_children = [child.task_id for child in children if child.status == "failed"]
        cancelled_children = [child.task_id for child in children if child.status == "cancelled"]
        missing_children = [
            child_id for child_id in task.child_task_ids if child_id not in state.tasks
        ]

        missing_dependencies = [
            dependency_id
            for dependency_id in task.dependencies
            if dependency_id not in state.tasks
        ]
        failed_dependencies = [
            dependency_id
            for dependency_id in task.dependencies
            if dependency_id in state.tasks
            and state.tasks[dependency_id].status in {"failed", "cancelled"}
        ]
        pending_dependencies = [
            dependency_id
            for dependency_id in task.dependencies
            if dependency_id in state.tasks
            and state.tasks[dependency_id].status not in TERMINAL_TASK_STATES
        ]
        cyclic_dependencies = [
            dependency_id
            for dependency_id in task.dependencies
            if dependency_id == task.task_id
            or self._dependency_reaches(state, dependency_id, task.task_id, set())
        ]
        missing_action_runs = [
            run_id for run_id in task.active_action_runs if run_id not in state.action_runs
        ]
        active_action_runs = [
            run_id
            for run_id in task.active_action_runs
            if run_id in state.action_runs
            and state.action_runs[run_id].status not in TERMINAL_ACTION_STATES
        ]
        status_override = task.scheduling.get("status_override")
        status_override_reason = task.scheduling.get("status_override_reason")

        desired_status = "runnable"
        classification = "runnable"
        reason = "task has no unresolved execution constraints"
        can_run = True

        if task.status in TERMINAL_TASK_STATES:
            desired_status = task.status
            classification = "terminal"
            reason = f"task is {task.status}"
            can_run = False
        elif missing_children or missing_dependencies or cyclic_dependencies or missing_action_runs:
            desired_status = "blocked"
            classification = "blocked_invalid_graph"
            reason = "task graph contains missing or cyclic references"
            can_run = False
        elif active_action_runs:
            desired_status = "running"
            classification = "running_action"
            reason = "an action run is active"
            can_run = False
        elif task.waiting_on:
            desired_status = "waiting"
            classification = "waiting_event"
            reason = "task has unresolved event/action wait conditions"
            can_run = False
        elif failed_dependencies:
            desired_status = "blocked"
            classification = "blocked_dependency"
            reason = "a required dependency failed or was cancelled"
            can_run = False
        elif pending_dependencies:
            desired_status = "waiting"
            classification = "waiting_dependency"
            reason = "required dependencies are not completed"
            can_run = False
        elif status_override in {"waiting", "blocked"}:
            desired_status = str(status_override)
            classification = f"explicit_{status_override}"
            reason = str(
                status_override_reason
                or f"task status was explicitly set to {status_override}"
            )
            can_run = False
        elif nonterminal_children:
            blocked_children = [
                child.task_id
                for child in children
                if child.task_id in nonterminal_children and child.status == "blocked"
            ]
            if blocked_children and len(blocked_children) == len(nonterminal_children):
                desired_status = "runnable"
                classification = "child_requires_resolution"
                reason = "all unfinished children are blocked; parent must decide recovery"
                can_run = True
            else:
                desired_status = "waiting"
                classification = "waiting_children"
                reason = "child tasks are still progressing or waiting"
                can_run = False
        elif failed_children or cancelled_children:
            desired_status = "runnable"
            classification = "review_child_outcomes"
            reason = "terminal child failures/cancellations require explicit review"
            can_run = True

        completion_blockers = self._completion_blockers(
            state,
            task,
            result=task.result if task.status == "completed" else None,
        )
        scheduling: JsonDict = {
            "managed_by": "task_system",
            "root_task_id": self.root_task_id(state, task.task_id),
            "depth": self.task_depth(state, task.task_id),
            "classification": classification,
            "reason": reason,
            "can_run": can_run,
            "can_complete": not completion_blockers,
            "dependency_ids": list(task.dependencies),
            "pending_dependency_ids": pending_dependencies,
            "failed_dependency_ids": failed_dependencies,
            "missing_dependency_ids": missing_dependencies,
            "cyclic_dependency_ids": cyclic_dependencies,
            "child_task_ids": child_ids,
            "nonterminal_child_ids": nonterminal_children,
            "completed_child_ids": completed_children,
            "failed_child_ids": failed_children,
            "cancelled_child_ids": cancelled_children,
            "missing_child_ids": missing_children,
            "active_action_run_ids": active_action_runs,
            "missing_action_run_ids": missing_action_runs,
            "unresolved_wait_count": len(task.waiting_on),
            "completion_blockers": completion_blockers,
        }
        if status_override in {"waiting", "blocked"}:
            scheduling["status_override"] = status_override
            scheduling["status_override_reason"] = status_override_reason
        return desired_status, scheduling

    def _completion_blockers(
        self,
        state: AgentState,
        task: AgentTask,
        *,
        result: Optional[JsonDict],
    ) -> List[JsonDict]:
        if task.status in {"failed", "cancelled"}:
            return [{"kind": "terminal_task_state", "status": task.status}]

        subtree = [task, *self.descendants(state, task.task_id)]
        descendants = subtree[1:]
        blockers: List[JsonDict] = []

        nonterminal_descendants = [
            item.task_id for item in descendants if item.status not in TERMINAL_TASK_STATES
        ]
        if nonterminal_descendants:
            blockers.append(
                {"kind": "nonterminal_descendants", "task_ids": nonterminal_descendants}
            )

        active_runs: List[str] = []
        unresolved_wait_tasks: List[str] = []
        incomplete_dependencies: List[JsonDict] = []
        invalid_references: List[JsonDict] = []
        for item in subtree:
            if item.waiting_on:
                unresolved_wait_tasks.append(item.task_id)
            for run_id in item.active_action_runs:
                run = state.action_runs.get(run_id)
                if run is None or run.status not in TERMINAL_ACTION_STATES:
                    active_runs.append(run_id)
            for dependency_id in item.dependencies:
                dependency = state.tasks.get(dependency_id)
                if dependency is None or dependency.status != "completed":
                    incomplete_dependencies.append(
                        {
                            "task_id": item.task_id,
                            "dependency_id": dependency_id,
                            "dependency_status": dependency.status if dependency else "missing",
                        }
                    )
            for child_id in item.child_task_ids:
                if child_id not in state.tasks:
                    invalid_references.append(
                        {"task_id": item.task_id, "missing_child_task_id": child_id}
                    )
        if active_runs:
            blockers.append({"kind": "active_action_runs", "action_run_ids": active_runs})
        if unresolved_wait_tasks:
            blockers.append({"kind": "unresolved_waits", "task_ids": unresolved_wait_tasks})
        if incomplete_dependencies:
            blockers.append(
                {"kind": "incomplete_dependencies", "dependencies": incomplete_dependencies}
            )
        if invalid_references:
            blockers.append({"kind": "invalid_task_references", "references": invalid_references})

        accepted_ids: set[str] = set()
        accept_all_terminal_outcomes = False
        if isinstance(result, dict):
            raw_ids = result.get("accepted_terminal_task_ids")
            if isinstance(raw_ids, list):
                accepted_ids = {str(item) for item in raw_ids}
            accept_all_terminal_outcomes = bool(
                result.get("accept_terminal_descendant_outcomes")
            )
        terminal_issues = [
            {"task_id": item.task_id, "status": item.status}
            for item in descendants
            if item.status in {"failed", "cancelled"}
            and not accept_all_terminal_outcomes
            and item.task_id not in accepted_ids
        ]
        if terminal_issues:
            blockers.append(
                {
                    "kind": "unacknowledged_terminal_descendant_outcomes",
                    "tasks": terminal_issues,
                    "resolution": (
                        "recover/replan, or acknowledge exact task ids in "
                        "result.accepted_terminal_task_ids"
                    ),
                }
            )
        return blockers

    def _apply_action_started(self, state: AgentState, event: AgentEvent) -> None:
        if not event.action_run_id or event.action_run_id not in state.action_runs:
            return
        run = state.action_runs[event.action_run_id]
        run.status = "running"
        run.started_at = run.started_at or utc_now()
        if run.task_id in state.tasks:
            task = state.tasks[run.task_id]
            if run.action_run_id not in task.active_action_runs:
                task.active_action_runs.append(run.action_run_id)
            task.scheduling.pop("status_override", None)
            task.scheduling.pop("status_override_reason", None)
            if task.status not in TERMINAL_TASK_STATES:
                task.status = "running"
            task.touch()

    def _apply_action_progress(self, state: AgentState, event: AgentEvent) -> None:
        if not event.action_run_id or event.action_run_id not in state.action_runs:
            return
        run = state.action_runs[event.action_run_id]
        run.progress = dict(event.payload.get("progress", {}))
        run.status = "running"
        if run.task_id in state.tasks:
            task = state.tasks[run.task_id]
            task.progress["active_action"] = {
                "action_run_id": run.action_run_id,
                "action_name": run.action_name,
                "progress": dict(run.progress),
            }
            if task.status not in TERMINAL_TASK_STATES:
                task.status = "running"
            task.touch()

    def _apply_action_completed(self, state: AgentState, event: AgentEvent) -> None:
        if not event.action_run_id or event.action_run_id not in state.action_runs:
            return
        run = state.action_runs[event.action_run_id]
        if run.status == "cancelled":
            return
        run.status = "succeeded"
        run.finished_at = utc_now()
        run.result = dict(event.payload.get("result", {}))
        run.progress = {"percent": 100, "message": "completed"}
        if run.task_id in state.tasks:
            task = state.tasks[run.task_id]
            if run.action_run_id in task.active_action_runs:
                task.active_action_runs.remove(run.action_run_id)
            task.waiting_on = [
                condition
                for condition in task.waiting_on
                if not self._wait_condition_satisfied(
                    state,
                    condition,
                    action_run_id=run.action_run_id,
                )
            ]
            if task.status not in TERMINAL_TASK_STATES:
                if self._should_complete_action_wrapper_task(state, task):
                    task.progress = dict(run.progress)
                    task.result = run.result
                    task.status = "completed"
                    task.waiting_on.clear()
                    task.active_action_runs.clear()
                    state.workspace.note(f"completed task {task.task_id}: {task.title}")
                else:
                    self._record_task_action_outcome(task, run)
                    task.status = "runnable"
            task.touch()
            self._resume_parent_after_child_terminal(
                state,
                task,
                reason="child_action_completed",
            )

    def _apply_action_failed(self, state: AgentState, event: AgentEvent) -> None:
        if not event.action_run_id or event.action_run_id not in state.action_runs:
            return
        run = state.action_runs[event.action_run_id]
        if run.status in {"succeeded", "cancelled"}:
            return
        run.status = "failed"
        run.finished_at = utc_now()
        run.error = dict(event.payload.get("error", {}))
        if run.task_id in state.tasks:
            task = state.tasks[run.task_id]
            if run.action_run_id in task.active_action_runs:
                task.active_action_runs.remove(run.action_run_id)
            task.waiting_on = [
                condition
                for condition in task.waiting_on
                if not self._wait_condition_satisfied(
                    state,
                    condition,
                    action_run_id=run.action_run_id,
                )
            ]
            scheduling_rejection = run.error.get("type") == "task_not_runnable"
            if scheduling_rejection:
                self._record_task_action_outcome(task, run)
                task.status = self._open_task_status(task)
            elif self._is_multi_step_objective_task(task):
                self._record_task_action_outcome(task, run)
                task.status = self._open_task_status(task)
            else:
                task.error = run.error
                task.status = "failed"
            task.touch()
            self._resume_parent_after_child_terminal(
                state,
                task,
                reason="child_action_failed",
                error=run.error,
            )

    def _apply_action_cancelled(self, state: AgentState, event: AgentEvent) -> None:
        if not event.action_run_id or event.action_run_id not in state.action_runs:
            return
        run = state.action_runs[event.action_run_id]
        if run.status in {"succeeded", "failed"}:
            return
        run.status = "cancelled"
        run.finished_at = utc_now()
        if run.task_id in state.tasks:
            task = state.tasks[run.task_id]
            if run.action_run_id in task.active_action_runs:
                task.active_action_runs.remove(run.action_run_id)
            if task.status not in TERMINAL_TASK_STATES:
                if self._is_multi_step_objective_task(task):
                    self._record_task_action_outcome(task, run)
                    task.status = self._open_task_status(task)
                else:
                    task.status = "cancelled"
            task.touch()
            self._resume_parent_after_child_terminal(
                state,
                task,
                reason="child_action_cancelled",
            )

    def _record_task_action_outcome(self, task: AgentTask, run: Any) -> None:
        task.progress.pop("active_action", None)
        if (
            task.progress.get("percent") == 100
            and task.progress.get("message") == "completed"
        ):
            task.progress.pop("percent", None)
            task.progress.pop("message", None)
        task.progress["last_action"] = {
            "action_run_id": run.action_run_id,
            "action_name": run.action_name,
            "status": run.status,
        }

    def _repair_multi_step_action_state(
        self,
        state: AgentState,
        task: AgentTask,
    ) -> None:
        if not self._is_multi_step_objective_task(task):
            return
        if task.status in TERMINAL_TASK_STATES:
            return
        repaired = False
        if (
            task.progress.get("percent") == 100
            and task.progress.get("message") == "completed"
        ):
            task.progress.pop("percent", None)
            task.progress.pop("message", None)
            repaired = True
        if task.result is not None and any(
            run.task_id == task.task_id and run.result == task.result
            for run in state.action_runs.values()
        ):
            task.result = None
            repaired = True
        if repaired:
            task.touch()

    def _refresh_current_task(self, state: AgentState) -> None:
        next_task_id = self.next_runnable_task_id(
            state,
            preferred_task_id=state.workspace.current_task_id,
            reconcile=False,
        )
        if next_task_id:
            state.workspace.current_task_id = next_task_id
            return
        current_id = state.workspace.current_task_id
        if current_id and current_id in state.tasks:
            if state.tasks[current_id].status not in TERMINAL_TASK_STATES:
                return
        nonterminal = [
            task for task in state.tasks.values() if task.status not in TERMINAL_TASK_STATES
        ]
        if nonterminal:
            selected = max(
                nonterminal,
                key=lambda task: self.task_depth(state, task.task_id),
            )
            state.workspace.current_task_id = selected.task_id
            return
        state.workspace.current_task_id = None

    def _wait_condition_satisfied(
        self,
        state: AgentState,
        condition: JsonDict,
        *,
        action_run_id: Optional[str] = None,
    ) -> bool:
        if not condition:
            return False
        single_run_id = condition.get("action_run_id")
        if isinstance(single_run_id, str) and single_run_id:
            if action_run_id and single_run_id != action_run_id:
                return False
            return self._action_run_terminal(state, single_run_id)
        run_ids = condition.get("action_run_ids")
        if isinstance(run_ids, list) and run_ids:
            normalized = [str(run_id) for run_id in run_ids if run_id]
            if action_run_id and action_run_id not in normalized:
                return False
            return all(self._action_run_terminal(state, run_id) for run_id in normalized)
        task_id = condition.get("task_id")
        if isinstance(task_id, str) and task_id:
            task = state.tasks.get(task_id)
            required_status = str(condition.get("status") or "completed")
            return bool(task and task.status == required_status)
        kind = condition.get("kind")
        if kind == "action_completed" and action_run_id:
            return condition.get("action_run_id") == action_run_id
        if kind == "task_completed":
            waited_task_id = condition.get("task_id")
            waited_task = state.tasks.get(str(waited_task_id))
            return bool(waited_task and waited_task.status == "completed")
        return False

    def _action_run_terminal(self, state: AgentState, action_run_id: str) -> bool:
        run = state.action_runs.get(action_run_id)
        return bool(run and run.status in TERMINAL_ACTION_STATES)

    def _is_multi_step_objective_task(self, task: AgentTask) -> bool:
        return (
            task.purpose == MULTI_STEP_OBJECTIVE_PURPOSE
            or task.continuation.get("kind") == "multi_step_objective"
        )

    def _open_task_status(self, task: AgentTask) -> str:
        return "waiting" if task.active_action_runs or task.waiting_on else "runnable"

    def _parent_task(self, state: AgentState, task: AgentTask) -> Optional[AgentTask]:
        if not task.parent_task_id:
            return None
        return state.tasks.get(task.parent_task_id)

    def _resume_parent_after_child_terminal(
        self,
        state: AgentState,
        task: AgentTask,
        *,
        reason: str,
        error: Optional[JsonDict] = None,
    ) -> None:
        if task.status not in TERMINAL_TASK_STATES:
            return
        parent = self._parent_task(state, task)
        if parent is None or parent.status in TERMINAL_TASK_STATES:
            return
        parent.progress["last_child_task_id"] = task.task_id
        parent.progress["last_child_event"] = reason
        if error is not None:
            parent.error = {
                "child_task_id": task.task_id,
                "child_error": error,
            }
        parent.touch()

    def _should_complete_action_wrapper_task(
        self,
        state: AgentState,
        task: AgentTask,
    ) -> bool:
        if task.active_action_runs or task.waiting_on or self._children(state, task.task_id):
            return False
        if any(
            dependency_id not in state.tasks
            or state.tasks[dependency_id].status != "completed"
            for dependency_id in task.dependencies
        ):
            return False
        return task.purpose in {
            "Runtime task created for a model-request tool call.",
            (
                "Runtime-created task because the generator started an action "
                "without a resolvable task."
            ),
        }

    def _children(self, state: AgentState, task_id: str) -> List[AgentTask]:
        parent = state.tasks.get(task_id)
        child_ids = list(parent.child_task_ids) if parent else []
        child_ids.extend(
            task.task_id
            for task in state.tasks.values()
            if task.parent_task_id == task_id and task.task_id not in child_ids
        )
        return [state.tasks[child_id] for child_id in child_ids if child_id in state.tasks]

    def _repair_parent_links(self, state: AgentState) -> None:
        for task in state.tasks.values():
            if not task.parent_task_id or task.parent_task_id not in state.tasks:
                continue
            parent = state.tasks[task.parent_task_id]
            if task.task_id not in parent.child_task_ids:
                parent.child_task_ids.append(task.task_id)
                parent.touch()

    def _dependency_reaches(
        self,
        state: AgentState,
        current_task_id: str,
        target_task_id: str,
        seen: set[str],
    ) -> bool:
        if current_task_id == target_task_id:
            return True
        if current_task_id in seen:
            return False
        seen.add(current_task_id)
        current = state.tasks.get(current_task_id)
        if current is None:
            return False
        return any(
            self._dependency_reaches(state, dependency_id, target_task_id, seen)
            for dependency_id in current.dependencies
        )

    def _unique_ids(self, values: List[Any]) -> List[str]:
        selected: List[str] = []
        for value in values:
            value_id = str(value)
            if value_id and value_id not in selected:
                selected.append(value_id)
        return selected
