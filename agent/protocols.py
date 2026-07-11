"""Core protocol objects for the event-driven agent MVP.

This file intentionally uses only Python stdlib dataclasses so the MVP can be
run without installing any dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


JsonDict = Dict[str, Any]


def ensure_json_dict(value: Any, *, scalar_key: str = "value") -> JsonDict:
    """Normalize one structured runtime value to a JSON object.

    Runtime state is persisted across versions and may also be patched by model
    tool calls.  Keeping this normalization at the protocol boundary prevents a
    scalar legacy/model value from breaking later mapping operations.
    """

    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    return {scalar_key: value}


def ensure_json_dict_list(
    value: Any,
    *,
    scalar_key: str = "value",
) -> List[JsonDict]:
    """Normalize a structured collection and every item inside it."""

    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    return [ensure_json_dict(item, scalar_key=scalar_key) for item in items]


def ensure_string_list(value: Any) -> List[str]:
    """Normalize a scalar or collection to a stable list of strings."""

    if value is None:
        return []
    items = value if isinstance(value, (list, tuple, set)) else [value]
    return [str(item) for item in items]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass
class AgentEvent:
    """Everything that can wake or update an agent enters as an event."""

    event_id: str
    agent_id: str
    type: str
    source: str
    payload: JsonDict = field(default_factory=dict)
    task_id: Optional[str] = None
    action_run_id: Optional[str] = None
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    priority: int = 100
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.payload = ensure_json_dict(self.payload)

    @staticmethod
    def make(
        *,
        agent_id: str,
        type: str,
        source: str,
        payload: Optional[JsonDict] = None,
        task_id: Optional[str] = None,
        action_run_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        priority: int = 100,
    ) -> "AgentEvent":
        return AgentEvent(
            event_id=new_id("evt"),
            agent_id=agent_id,
            type=type,
            source=source,
            payload=payload or {},
            task_id=task_id,
            action_run_id=action_run_id,
            correlation_id=correlation_id,
            causation_id=causation_id,
            idempotency_key=idempotency_key,
            priority=priority,
        )

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @staticmethod
    def from_dict(data: JsonDict) -> "AgentEvent":
        return AgentEvent(**data)


@dataclass
class AgentProfile:
    """Stable identity/configuration for the agent."""

    agent_id: str
    name: str = "Agent"
    system_profile: str = ""
    persona_profile: str = ""
    behavior_profile: str = ""
    identity_profile: str = ""
    background_profile: str = ""
    values_profile: str = ""
    voice_profile: str = ""
    speech_profile: str = ""
    relationship_profile: str = ""
    self_boundaries: str = ""

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @staticmethod
    def from_dict(data: JsonDict) -> "AgentProfile":
        return AgentProfile(**data)


@dataclass
class Workspace:
    """Working memory of the agent.

    This is the MVP version of working memory. It also owns the context-builder
    input material: short transcript, notes, variables, and current task focus.
    """

    workspace_id: str
    current_task_id: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    variables: JsonDict = field(default_factory=dict)
    transcript: List[JsonDict] = field(default_factory=list)
    last_decision_summary: str = ""
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.notes = ensure_string_list(self.notes)
        self.variables = ensure_json_dict(self.variables)
        self.transcript = ensure_json_dict_list(self.transcript, scalar_key="content")

    def add_transcript(self, role: str, content: str, *, event_id: Optional[str] = None) -> None:
        self.transcript.append(
            {
                "role": role,
                "content": content,
                "event_id": event_id,
                "created_at": utc_now(),
            }
        )
        self.updated_at = utc_now()

    def note(self, text: str) -> None:
        self.notes.append(f"{utc_now()} {text}")
        self.updated_at = utc_now()

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @staticmethod
    def from_dict(data: JsonDict) -> "Workspace":
        normalized = dict(data) if isinstance(data, dict) else {}
        normalized.setdefault("workspace_id", new_id("ws"))
        return Workspace(**normalized)


@dataclass
class AgentTask:
    """Task is the basic unit of agent activity, like a PCB for the agent."""

    task_id: str
    agent_id: str
    title: str
    goal: str
    purpose: str
    status: str = "created"  # created/runnable/running/waiting/blocked/completed/failed/cancelled
    parent_task_id: Optional[str] = None
    child_task_ids: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    active_action_runs: List[str] = field(default_factory=list)
    waiting_on: List[JsonDict] = field(default_factory=list)
    scheduling: JsonDict = field(default_factory=dict)
    progress: JsonDict = field(default_factory=dict)
    result_ref: Optional[str] = None
    result: JsonDict = field(default_factory=dict)
    error: JsonDict = field(default_factory=dict)
    workspace_ref: Optional[str] = None
    continuation: JsonDict = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    version: int = 0

    def __post_init__(self) -> None:
        self.child_task_ids = ensure_string_list(self.child_task_ids)
        self.dependencies = ensure_string_list(self.dependencies)
        self.active_action_runs = ensure_string_list(self.active_action_runs)
        self.waiting_on = ensure_json_dict_list(self.waiting_on)
        self.scheduling = ensure_json_dict(self.scheduling)
        self.progress = ensure_json_dict(self.progress, scalar_key="message")
        self.result = ensure_json_dict(self.result)
        self.error = ensure_json_dict(self.error, scalar_key="message")
        self.continuation = ensure_json_dict(self.continuation)

    def touch(self) -> None:
        self.updated_at = utc_now()
        self.version += 1

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @staticmethod
    def from_dict(data: JsonDict) -> "AgentTask":
        normalized = dict(data)
        normalized["scheduling"] = ensure_json_dict(normalized.get("scheduling"))
        normalized["progress"] = ensure_json_dict(
            normalized.get("progress"),
            scalar_key="message",
        )
        normalized["result"] = ensure_json_dict(normalized.get("result"))
        normalized["error"] = ensure_json_dict(
            normalized.get("error"),
            scalar_key="message",
        )
        normalized["continuation"] = ensure_json_dict(
            normalized.get("continuation")
        )
        return AgentTask(**normalized)


@dataclass
class ActionRun:
    """One concrete execution of an action/tool under a task."""

    action_run_id: str
    agent_id: str
    task_id: str
    action_name: str
    args: JsonDict
    mode: str  # sync/async/stream/subscription; MVP implements sync + async.
    source: str = "local"
    status: str = "created"  # created/running/succeeded/failed/cancelled
    progress: JsonDict = field(default_factory=dict)
    result: JsonDict = field(default_factory=dict)
    error: JsonDict = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    idempotency_key: Optional[str] = None

    def __post_init__(self) -> None:
        self.args = ensure_json_dict(self.args)
        self.progress = ensure_json_dict(self.progress, scalar_key="message")
        self.result = ensure_json_dict(self.result)
        self.error = ensure_json_dict(self.error, scalar_key="message")

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @staticmethod
    def from_dict(data: JsonDict) -> "ActionRun":
        normalized = dict(data)
        normalized["args"] = ensure_json_dict(normalized.get("args"))
        normalized["progress"] = ensure_json_dict(
            normalized.get("progress"),
            scalar_key="message",
        )
        normalized["result"] = ensure_json_dict(normalized.get("result"))
        normalized["error"] = ensure_json_dict(
            normalized.get("error"),
            scalar_key="message",
        )
        return ActionRun(**normalized)


@dataclass
class ActionSpec:
    """Capability descriptor exposed to the Generator through context."""

    name: str
    description: str
    input_schema: JsonDict
    mode: str = "sync"
    timeout_ms: int = 5000
    cancelable: bool = True
    requires_approval: bool = False
    side_effect_level: str = "read"
    source: str = "local"
    target: Optional[str] = None
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.input_schema = ensure_json_dict(self.input_schema)
        self.metadata = ensure_json_dict(self.metadata)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class ConversationUnderstanding:
    """Wernicke's attributed interpretation of one external utterance."""

    understanding_id: str
    turn_id: str
    speaker_id: str
    semantic_summary: str
    speech_act: str = "unknown"
    intents: List[str] = field(default_factory=list)
    key_information: List[JsonDict] = field(default_factory=list)
    entities: List[JsonDict] = field(default_factory=list)
    affect_cues: JsonDict = field(default_factory=dict)
    dialogue_obligations: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)
    task_relevance: str = ""
    response_needed: bool = True
    decision_needed: bool = False
    decision_request: str = ""
    confidence: float = 0.5
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.intents = ensure_string_list(self.intents)
        self.key_information = ensure_json_dict_list(self.key_information)
        self.entities = ensure_json_dict_list(self.entities)
        self.affect_cues = ensure_json_dict(self.affect_cues)
        self.dialogue_obligations = ensure_string_list(self.dialogue_obligations)
        self.ambiguities = ensure_string_list(self.ambiguities)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @staticmethod
    def from_dict(data: JsonDict) -> "ConversationUnderstanding":
        return ConversationUnderstanding(**data)


@dataclass
class ConversationTurn:
    """Durable lifecycle for one incoming utterance and its response."""

    turn_id: str
    conversation_id: str
    agent_id: str
    speaker_id: str
    recipient_id: str
    channel: str
    source_event_id: str
    utterance: str
    speaker_context: JsonDict = field(default_factory=dict)
    scene_context: JsonDict = field(default_factory=dict)
    status: str = "received"
    understanding: JsonDict = field(default_factory=dict)
    decision: JsonDict = field(default_factory=dict)
    speech_intent: JsonDict = field(default_factory=dict)
    response_text: Optional[str] = None
    response_event_id: Optional[str] = None
    outbound_utterances: List[JsonDict] = field(default_factory=list)
    suppressed_speech_intents: List[JsonDict] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.speaker_context = ensure_json_dict(self.speaker_context)
        self.scene_context = ensure_json_dict(self.scene_context)
        self.understanding = ensure_json_dict(self.understanding)
        self.decision = ensure_json_dict(self.decision)
        self.speech_intent = ensure_json_dict(self.speech_intent)
        self.outbound_utterances = ensure_json_dict_list(self.outbound_utterances)
        self.suppressed_speech_intents = ensure_json_dict_list(
            self.suppressed_speech_intents
        )

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @staticmethod
    def from_dict(data: JsonDict) -> "ConversationTurn":
        return ConversationTurn(**data)


@dataclass
class ConversationSession:
    """Conversation identity and ordered turn references."""

    conversation_id: str
    agent_id: str
    participant_ids: List[str] = field(default_factory=list)
    turn_ids: List[str] = field(default_factory=list)
    status: str = "active"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.participant_ids = ensure_string_list(self.participant_ids)
        self.turn_ids = ensure_string_list(self.turn_ids)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @staticmethod
    def from_dict(data: JsonDict) -> "ConversationSession":
        return ConversationSession(**data)


@dataclass
class ConversationState:
    """Persisted conversation source of truth for one agent."""

    agent_id: str
    sessions: Dict[str, ConversationSession] = field(default_factory=dict)
    turns: Dict[str, ConversationTurn] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        raw_sessions = self.sessions if isinstance(self.sessions, dict) else {}
        raw_turns = self.turns if isinstance(self.turns, dict) else {}
        self.sessions = {
            str(key): (
                value
                if isinstance(value, ConversationSession)
                else ConversationSession.from_dict(value)
            )
            for key, value in raw_sessions.items()
            if isinstance(value, (ConversationSession, dict))
        }
        self.turns = {
            str(key): (
                value
                if isinstance(value, ConversationTurn)
                else ConversationTurn.from_dict(value)
            )
            for key, value in raw_turns.items()
            if isinstance(value, (ConversationTurn, dict))
        }

    def to_dict(self) -> JsonDict:
        return {
            "agent_id": self.agent_id,
            "sessions": {key: value.to_dict() for key, value in self.sessions.items()},
            "turns": {key: value.to_dict() for key, value in self.turns.items()},
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: JsonDict) -> "ConversationState":
        sessions = data.get("sessions") if isinstance(data.get("sessions"), dict) else {}
        turns = data.get("turns") if isinstance(data.get("turns"), dict) else {}
        return ConversationState(
            agent_id=str(data["agent_id"]),
            sessions={
                str(key): ConversationSession.from_dict(value)
                for key, value in sessions.items()
                if isinstance(value, dict)
            },
            turns={
                str(key): ConversationTurn.from_dict(value)
                for key, value in turns.items()
                if isinstance(value, dict)
            },
            updated_at=str(data.get("updated_at") or utc_now()),
        )


@dataclass
class AgentState:
    """Persisted state for one agent actor."""

    agent_id: str
    profile: AgentProfile
    workspace: Workspace
    tasks: Dict[str, AgentTask] = field(default_factory=dict)
    action_runs: Dict[str, ActionRun] = field(default_factory=dict)
    processed_event_ids: List[str] = field(default_factory=list)
    version: int = 0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        raw_tasks = self.tasks if isinstance(self.tasks, dict) else {}
        raw_runs = self.action_runs if isinstance(self.action_runs, dict) else {}
        self.tasks = {
            str(key): value if isinstance(value, AgentTask) else AgentTask.from_dict(value)
            for key, value in raw_tasks.items()
            if isinstance(value, (AgentTask, dict))
        }
        self.action_runs = {
            str(key): (
                value if isinstance(value, ActionRun) else ActionRun.from_dict(value)
            )
            for key, value in raw_runs.items()
            if isinstance(value, (ActionRun, dict))
        }
        self.processed_event_ids = ensure_string_list(self.processed_event_ids)

    @staticmethod
    def new(agent_id: str) -> "AgentState":
        return AgentState(
            agent_id=agent_id,
            profile=AgentProfile(agent_id=agent_id),
            workspace=Workspace(workspace_id=new_id("ws")),
        )

    def mark_processed(self, event_id: str) -> None:
        if event_id not in self.processed_event_ids:
            self.processed_event_ids.append(event_id)
        self.processed_event_ids = self.processed_event_ids[-500:]
        self.version += 1
        self.updated_at = utc_now()

    def to_dict(self) -> JsonDict:
        return {
            "agent_id": self.agent_id,
            "profile": self.profile.to_dict(),
            "workspace": self.workspace.to_dict(),
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "action_runs": {k: v.to_dict() for k, v in self.action_runs.items()},
            "processed_event_ids": list(self.processed_event_ids),
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: JsonDict) -> "AgentState":
        agent_id = str(data["agent_id"])
        profile_data = data.get("profile")
        workspace_data = data.get("workspace")
        tasks = data.get("tasks") if isinstance(data.get("tasks"), dict) else {}
        action_runs = (
            data.get("action_runs")
            if isinstance(data.get("action_runs"), dict)
            else {}
        )
        return AgentState(
            agent_id=agent_id,
            profile=(
                AgentProfile.from_dict(profile_data)
                if isinstance(profile_data, dict)
                else AgentProfile(agent_id=agent_id)
            ),
            workspace=Workspace.from_dict(
                workspace_data if isinstance(workspace_data, dict) else {}
            ),
            tasks={
                str(k): AgentTask.from_dict(v)
                for k, v in tasks.items()
                if isinstance(v, dict)
            },
            action_runs={
                str(k): ActionRun.from_dict(v)
                for k, v in action_runs.items()
                if isinstance(v, dict)
            },
            processed_event_ids=ensure_string_list(data.get("processed_event_ids")),
            version=data.get("version", 0),
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
        )


GeneratorDecision = JsonDict
Command = JsonDict
AwaitCondition = JsonDict
