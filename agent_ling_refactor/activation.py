from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from agent.protocols import AgentEvent, AgentState, JsonDict, utc_now

from .settings import ActivationSettings


_STATE_KEY = "model_activation"
_RETRYABLE_MARKERS = (
    "429",
    "rate limit",
    "ratelimit",
    "too many requests",
    "timeout",
    "temporarily",
)


@dataclass(frozen=True)
class ActivationDecision:
    activate: bool
    reason: str
    fingerprint: str = ""


class ModelActivationGate:
    """Admit only events that add decision-relevant information.

    Event ingestion remains lossless.  This gate controls only the expensive
    transition from persisted state to a model request.
    """

    def __init__(self, settings: ActivationSettings) -> None:
        self.settings = settings

    def begin_user_turn(self, state: AgentState, event: AgentEvent) -> ActivationDecision:
        decision = self._begin_chain(state, event)
        if not decision.activate:
            return decision
        return ActivationDecision(True, "new user message always has unique conversational value")

    def begin_objective(self, state: AgentState, event: AgentEvent) -> ActivationDecision:
        decision = self._begin_chain(state, event)
        if not decision.activate:
            return decision
        return ActivationDecision(True, "new runtime objective starts a bounded decision chain")

    def _begin_chain(self, state: AgentState, event: AgentEvent) -> ActivationDecision:
        if self.in_backoff(state):
            return ActivationDecision(False, "model provider backoff is active")
        data = self._state(state)
        data["decision_hops"] = 0
        data["chain_root_event_id"] = event.event_id
        data["last_external_event_at"] = utc_now()
        for task in state.tasks.values():
            paused = task.progress.get("model_activation_paused")
            if not isinstance(paused, dict):
                continue
            task.progress.pop("model_activation_paused", None)
            if task.scheduling.get("status_override_reason") == "model_activation_budget":
                task.scheduling.pop("status_override", None)
                task.scheduling.pop("status_override_reason", None)
                if task.status == "blocked" and not task.waiting_on:
                    task.status = "runnable"
            task.touch()
        return ActivationDecision(True, "new activation chain")

    def evaluate(self, state: AgentState, event: AgentEvent) -> ActivationDecision:
        if self.in_backoff(state):
            return ActivationDecision(False, "model provider backoff is active")

        if event.type == "action.internal.completed":
            return ActivationDecision(
                False,
                "internal task mutation is the result of an existing decision",
            )
        if event.type == "protocol.tool_specification":
            return ActivationDecision(False, "tool catalog refresh is context-only")

        if event.type == "action.failed" and self._not_runnable_failure(event):
            return ActivationDecision(
                False,
                "action was rejected by deterministic task constraints",
            )

        task = state.tasks.get(event.task_id or "")
        if task is not None and task.status == "waiting" and task.waiting_on:
            return ActivationDecision(
                False,
                "task is explicitly waiting for an external condition",
            )

        if event.type == "action.completed" and event.payload.get("deduplicated"):
            return ActivationDecision(False, "replayed success contains no new evidence")

        if event.type == "protocol.event" and self._is_self_broadcast(state, event):
            return ActivationDecision(
                False,
                "self-authored protocol broadcast duplicates an action outcome",
            )

        if event.type == "agent.thought" and self._has_open_tasks(state):
            return ActivationDecision(
                False,
                "idle reflection cannot interrupt an open task",
            )

        fingerprint = self._fingerprint(event)
        if fingerprint and self._seen_recently(state, fingerprint):
            return ActivationDecision(False, "equivalent evidence was already evaluated", fingerprint)

        data = self._state(state)
        hops = int(data.get("decision_hops") or 0)
        if event.type != "user.message" and hops >= self.settings.max_decision_hops:
            if task is not None and task.status not in {"completed", "failed", "cancelled"}:
                task.status = "blocked"
                task.scheduling["status_override"] = "blocked"
                task.scheduling["status_override_reason"] = "model_activation_budget"
                task.progress["model_activation_paused"] = {
                    "reason": "decision chain exhausted its model activation budget",
                    "event_id": event.event_id,
                    "max_decision_hops": self.settings.max_decision_hops,
                    "created_at": utc_now(),
                }
                task.touch()
            return ActivationDecision(
                False,
                f"decision chain reached its {self.settings.max_decision_hops}-hop budget",
                fingerprint,
            )

        if fingerprint:
            self._remember(state, fingerprint)
        data["decision_hops"] = hops + 1
        data["last_activation_at"] = utc_now()
        data["last_activation_event_id"] = event.event_id
        data["last_activation_event_type"] = event.type
        return ActivationDecision(True, "event adds new decision-relevant evidence", fingerprint)

    def record_success(self, state: AgentState) -> None:
        data = self._state(state)
        data["consecutive_model_errors"] = 0
        data.pop("backoff_until", None)
        data.pop("last_model_error", None)

    def record_error(self, state: AgentState, exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".casefold()
        retryable = any(marker in text for marker in _RETRYABLE_MARKERS)
        data = self._state(state)
        count = int(data.get("consecutive_model_errors") or 0) + 1
        data["consecutive_model_errors"] = count
        data["last_model_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "retryable": retryable,
            "created_at": utc_now(),
        }
        if retryable:
            seconds = min(
                self.settings.backoff_max_seconds,
                self.settings.backoff_initial_seconds * (2 ** max(0, count - 1)),
            )
            until = datetime.now(timezone.utc) + timedelta(seconds=seconds)
            data["backoff_until"] = until.isoformat()
            data["backoff_seconds"] = seconds
        return retryable

    def in_backoff(self, state: AgentState) -> bool:
        value = self._state(state).get("backoff_until")
        if not isinstance(value, str) or not value:
            return False
        try:
            until = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        return datetime.now(timezone.utc) < until

    def suppression_audit(
        self,
        event: AgentEvent,
        decision: ActivationDecision,
    ) -> JsonDict:
        return {
            "model_activation": "suppressed",
            "event_type": event.type,
            "reason": decision.reason,
            "fingerprint": decision.fingerprint,
        }

    def _state(self, state: AgentState) -> JsonDict:
        value = state.workspace.variables.setdefault(
            _STATE_KEY,
            {
                "decision_hops": 0,
                "recent_fingerprints": {},
                "consecutive_model_errors": 0,
            },
        )
        if not isinstance(value, dict):
            value = {}
            state.workspace.variables[_STATE_KEY] = value
        value.setdefault("recent_fingerprints", {})
        return value

    def _fingerprint(self, event: AgentEvent) -> str:
        if event.type not in {
            "action.completed",
            "action.failed",
            "action.cancelled",
            "protocol.event",
            "agent.thought",
            "timer.fired",
        }:
            return ""
        payload = self._stable_value(event.payload)
        canonical = json.dumps(
            {"type": event.type, "task_id": event.task_id, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

    def _stable_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self._stable_value(item)
                for key, item in value.items()
                if key
                not in {
                    "id",
                    "event_id",
                    "external_action_id",
                    "created_at",
                    "updated_at",
                    "started_at",
                    "finished_at",
                    "action_run_id",
                    "action_run_ids",
                    "active_action_runs",
                    "active_action_run_ids",
                    "version",
                }
            }
        if isinstance(value, list):
            return [self._stable_value(item) for item in value]
        return value

    def _seen_recently(self, state: AgentState, fingerprint: str) -> bool:
        recent = self._state(state).get("recent_fingerprints")
        if not isinstance(recent, dict):
            return False
        value = recent.get(fingerprint)
        if not isinstance(value, str):
            return False
        try:
            seen_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        age = (datetime.now(timezone.utc) - seen_at).total_seconds()
        return age < self.settings.duplicate_ttl_seconds

    def _remember(self, state: AgentState, fingerprint: str) -> None:
        data = self._state(state)
        recent = data.get("recent_fingerprints")
        if not isinstance(recent, dict):
            recent = {}
        recent[fingerprint] = utc_now()
        if len(recent) > 128:
            recent = dict(sorted(recent.items(), key=lambda item: item[1])[-128:])
        data["recent_fingerprints"] = recent

    def _not_runnable_failure(self, event: AgentEvent) -> bool:
        error = event.payload.get("error")
        return isinstance(error, dict) and error.get("type") == "task_not_runnable"

    def _is_self_broadcast(self, state: AgentState, event: AgentEvent) -> bool:
        if not event.payload.get("broadcast"):
            return False
        content = event.payload.get("content")
        if not isinstance(content, dict):
            return False
        data = content.get("data")
        return isinstance(data, dict) and data.get("actor") == state.agent_id

    def _has_open_tasks(self, state: AgentState) -> bool:
        return any(
            task.status not in {"completed", "failed", "cancelled"}
            for task in state.tasks.values()
        )
