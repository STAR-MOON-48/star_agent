from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import TYPE_CHECKING, Any

from ....config import ContextPolicyConfig, MemoryConfig
from ....protocols import AgentEvent, AgentState, JsonDict, ensure_json_dict, utc_now
from ...persistence_system import MemoryRecord, MemoryStore

if TYPE_CHECKING:
    from ...kernel.generator_runtime import GeneratorRuntime


MEMORY_VARIABLE_KEY = "memory_system"
CAPTURED_EVENT_TYPES = {
    "user.message",
    "action.completed",
    "action.failed",
    "action.cancelled",
    "action.internal.completed",
    "conversation.understanding.ready",
    "conversation.utterance.sent",
    "agent.thought",
}
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
}


@dataclass(frozen=True)
class MemoryReflection:
    trigger_event: AgentEvent
    context: JsonDict
    public_context: JsonDict
    record: MemoryRecord
    model_trace: JsonDict

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", ensure_json_dict(self.context))
        object.__setattr__(self, "public_context", ensure_json_dict(self.public_context))
        object.__setattr__(self, "model_trace", ensure_json_dict(self.model_trace))


class MemorySystem:
    """Captures episodes, retrieves experience, and consolidates it with the model."""

    def __init__(
        self,
        config: MemoryConfig,
        *,
        store: MemoryStore,
        context_policy: ContextPolicyConfig | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.context_policy = context_policy or ContextPolicyConfig.empty()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def poll_interval_seconds(self) -> float:
        return max(5.0, min(self.config.reflection_interval_seconds / 4.0, 30.0))

    def observe_event(self, state: AgentState, event: AgentEvent) -> MemoryRecord | None:
        if not self.enabled or not self.config.auto_capture:
            return None
        if event.type not in CAPTURED_EVENT_TYPES:
            return None
        memory_id = "mem_evt_" + hashlib.sha256(
            f"{state.agent_id}:{event.event_id}".encode("utf-8")
        ).hexdigest()[:20]
        existing = self.store.get(state.agent_id, memory_id)
        if existing is not None:
            return existing
        payload = self._redact(event.payload)
        content = "\n".join(
            [
                f"# {self._event_title(event)}",
                "",
                "## 事件",
                f"- 类型: `{event.type}`",
                f"- 来源: `{event.source}`",
                f"- 时间: `{event.created_at}`",
                f"- Task: `{event.task_id or 'none'}`",
                f"- ActionRun: `{event.action_run_id or 'none'}`",
                "",
                "## 事实载荷",
                "```json",
                json.dumps(payload, ensure_ascii=False, indent=2)[:8000],
                "```",
            ]
        )
        record = self.store.save(
            MemoryRecord(
                memory_id=memory_id,
                agent_id=state.agent_id,
                kind="episodic",
                title=self._event_title(event),
                content=content,
                tags=self._event_tags(event),
                source_refs=[
                    {
                        "type": "event",
                        "id": event.event_id,
                        "event_type": event.type,
                    }
                ],
            )
        )
        memory_state = self._state(state)
        pending = memory_state.setdefault("pending_reflection_ids", [])
        if not isinstance(pending, list):
            pending = []
        if record.memory_id not in pending:
            pending.append(record.memory_id)
        memory_state["pending_reflection_ids"] = pending[-self.config.max_pending_events :]
        memory_state["last_captured_at"] = utc_now()
        memory_state["captured_count"] = int(memory_state.get("captured_count", 0)) + 1
        return record

    def search(
        self,
        *,
        agent_id: str,
        query: str = "",
        kind: str | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> list[JsonDict]:
        if not self.enabled:
            return []
        records = self.store.search(
            agent_id,
            query=query,
            kind=kind,
            tags=tags,
            limit=limit or self.config.retrieval_limit,
        )
        return [self._context_record(record) for record in records]

    def read(self, *, agent_id: str, memory_id: str) -> JsonDict | None:
        record = self.store.get(agent_id, memory_id)
        return record.to_dict() if record is not None else None

    def context_for(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        limit: int,
    ) -> JsonDict:
        query = self._event_query(state, event)
        records = self.search(
            agent_id=state.agent_id,
            query=query,
            limit=limit,
        )
        return {
            "query": query,
            "records": records,
            "count": len(records),
            "source_of_truth": "markdown_memory_store",
        }

    async def maybe_reflect(
        self,
        *,
        state: AgentState,
        generator_runtime: "GeneratorRuntime",
    ) -> MemoryReflection | None:
        if not self._should_reflect(state):
            return None
        memory_state = self._state(state)
        pending_ids = list(memory_state.get("pending_reflection_ids") or [])
        records = [
            record
            for memory_id in pending_ids
            if (record := self.store.get(state.agent_id, str(memory_id))) is not None
        ]
        if not records:
            memory_state["pending_reflection_ids"] = []
            return None
        trigger = AgentEvent.make(
            agent_id=state.agent_id,
            type="memory.reflection.requested",
            source="memory_system",
            payload={"memory_ids": [record.memory_id for record in records]},
            priority=140,
        )
        context: JsonDict = {
            "context_kind": "memory_reflection_pack",
            "agent": state.profile.to_dict(),
            "memory": {
                "pending_count": len(records),
                "episodes": [self._context_record(record, content_limit=3000) for record in records],
            },
            "cognition": {
                "emotion": state.workspace.variables.get("emotion_state"),
                "last_decision_summary": state.workspace.last_decision_summary,
            },
            "_generator_session": "memory_reflection",
            "instruction": (
                "Consolidate only reusable, evidence-backed experience. Return Markdown; "
                "do not call tools and do not invent facts."
            ),
        }
        result, trace = await generator_runtime.generate_text_with_trace(
            context,
            session_id="memory_reflection",
        )
        content = (result.text or "").strip()
        if not content:
            raise ValueError("MemoryReflectionSystem returned empty Markdown.")
        reflection_id = "mem_ref_" + hashlib.sha256(
            ("|".join(record.memory_id for record in records) + content).encode("utf-8")
        ).hexdigest()[:20]
        title = self._markdown_title(content) or "经验反思"
        reflection_record = self.store.save(
            MemoryRecord(
                memory_id=reflection_id,
                agent_id=state.agent_id,
                kind="semantic",
                title=title,
                content=content,
                tags=["reflection", "reusable_experience"],
                source_refs=[
                    {"type": "memory", "id": record.memory_id}
                    for record in records
                ],
                confidence=0.8,
            )
        )
        memory_state["pending_reflection_ids"] = []
        memory_state["last_reflection_at"] = utc_now()
        memory_state["last_reflection_id"] = reflection_record.memory_id
        memory_state["reflection_count"] = int(memory_state.get("reflection_count", 0)) + 1
        memory_state.pop("last_error", None)
        state.version += 1
        state.updated_at = utc_now()
        return MemoryReflection(
            trigger_event=trigger,
            context=context,
            public_context=generator_runtime.public_context(context),
            record=reflection_record,
            model_trace=trace,
        )

    def record_error(self, state: AgentState, exc: Exception) -> None:
        memory_state = self._state(state)
        attempted_at = utc_now()
        memory_state["last_reflection_attempt_at"] = attempted_at
        memory_state["last_reflection_at"] = attempted_at
        memory_state["last_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    def _should_reflect(self, state: AgentState) -> bool:
        if not self.enabled or not self.config.reflection_enabled:
            return False
        pending = self._state(state).get("pending_reflection_ids")
        if not isinstance(pending, list) or len(pending) < self.config.reflection_min_events:
            return False
        if any(run.status in {"created", "running"} for run in state.action_runs.values()):
            return False
        last = self._parse_datetime(self._state(state).get("last_reflection_at"))
        if last is None:
            return True
        return (
            datetime.now(timezone.utc) - last
        ).total_seconds() >= self.config.reflection_interval_seconds

    def _state(self, state: AgentState) -> JsonDict:
        value = state.workspace.variables.setdefault(MEMORY_VARIABLE_KEY, {})
        if not isinstance(value, dict):
            value = {}
            state.workspace.variables[MEMORY_VARIABLE_KEY] = value
        return value

    def _event_title(self, event: AgentEvent) -> str:
        return f"{event.type} · {event.task_id or event.action_run_id or event.event_id}"

    def _event_tags(self, event: AgentEvent) -> list[str]:
        tags = [event.type, event.source]
        if event.task_id:
            tags.append("task")
        if event.action_run_id:
            tags.append("action")
        if event.type.startswith("conversation.") or event.type == "user.message":
            tags.append("conversation")
        return list(dict.fromkeys(tags))

    def _event_query(self, state: AgentState, event: AgentEvent) -> str:
        values = [
            str(event.payload.get("content") or ""),
            str(event.payload.get("decision_request") or ""),
            str(event.payload.get("action_name") or ""),
        ]
        if event.task_id and event.task_id in state.tasks:
            task = state.tasks[event.task_id]
            values.extend([task.title, task.goal])
        return " ".join(value for value in values if value).strip() or event.type

    def _context_record(
        self,
        record: MemoryRecord,
        *,
        content_limit: int = 1200,
    ) -> JsonDict:
        return {
            "memory_id": record.memory_id,
            "kind": record.kind,
            "title": record.title,
            "content": record.content[:content_limit],
            "tags": record.tags,
            "confidence": record.confidence,
            "source_refs": record.source_refs,
            "updated_at": record.updated_at,
        }

    def _redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): "[REDACTED]"
                if str(key).casefold() in SENSITIVE_KEYS
                else self._redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        return value

    def _markdown_title(self, content: str) -> str:
        for line in content.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return ""

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
