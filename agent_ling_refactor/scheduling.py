from __future__ import annotations

from agent.protocols import AgentEvent, AgentState, JsonDict, ensure_json_dict
from agent.runtime.action_systems.task_system import TERMINAL_TASK_STATES, TaskSystem


_HUMAN_WAIT_TYPES = {"human_response", "user_message", "user.response"}


class RefactoredTaskSystem(TaskSystem):
    """TaskSystem fixes local to the refactored runtime."""

    def add_wait(self, state: AgentState, task_id: str, condition: JsonDict) -> None:
        normalized = ensure_json_dict(condition)
        if not normalized:
            # An empty condition has no event that could satisfy it.  Treat it
            # as "do not wait" instead of creating a permanent empty blocker.
            task = state.tasks[task_id]
            task.waiting_on = [item for item in task.waiting_on if item]
            if task.status not in TERMINAL_TASK_STATES and not task.active_action_runs:
                task.status = "runnable"
                task.scheduling.pop("status_override", None)
                task.scheduling.pop("status_override_reason", None)
            task.touch()
            self.reconcile(state)
            return
        super().add_wait(state, task_id, normalized)

    def apply_event(self, state: AgentState, event: AgentEvent) -> None:
        self.sanitize_waits(state)
        super().apply_event(state, event)
        if event.type == "user.message":
            self.resolve_human_waits(state, event)

    def sanitize_waits(self, state: AgentState) -> list[str]:
        repaired: list[str] = []
        for task in state.tasks.values():
            valid = [condition for condition in task.waiting_on if condition]
            if valid == task.waiting_on:
                continue
            task.waiting_on = valid
            task.touch()
            repaired.append(task.task_id)
        if repaired:
            self.reconcile(state)
        return repaired

    def resolve_human_waits(self, state: AgentState, event: AgentEvent) -> list[str]:
        resolved: list[str] = []
        for task in state.tasks.values():
            if task.status in TERMINAL_TASK_STATES:
                continue
            remaining = [
                condition
                for condition in task.waiting_on
                if condition and condition.get("awaiting") not in _HUMAN_WAIT_TYPES
            ]
            if len(remaining) == len(task.waiting_on):
                continue
            task.waiting_on = remaining
            task.continuation.pop("awaiting", None)
            task.scheduling.pop("status_override", None)
            task.scheduling.pop("status_override_reason", None)
            task.progress["last_wait_resolution"] = {
                "reason": "new user message satisfied human wait",
                "event_id": event.event_id,
            }
            if not task.active_action_runs:
                task.status = "runnable"
            task.touch()
            resolved.append(task.task_id)
        if resolved:
            self.reconcile(state)
        return resolved
