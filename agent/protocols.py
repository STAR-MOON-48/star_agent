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
        return Workspace(**data)


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
    result: Optional[JsonDict] = None
    error: Optional[JsonDict] = None
    workspace_ref: Optional[str] = None
    continuation: JsonDict = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    version: int = 0

    def touch(self) -> None:
        self.updated_at = utc_now()
        self.version += 1

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @staticmethod
    def from_dict(data: JsonDict) -> "AgentTask":
        return AgentTask(**data)


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
    result: Optional[JsonDict] = None
    error: Optional[JsonDict] = None
    created_at: str = field(default_factory=utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    idempotency_key: Optional[str] = None

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @staticmethod
    def from_dict(data: JsonDict) -> "ActionRun":
        return ActionRun(**data)


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
    understanding: Optional[JsonDict] = None
    decision: Optional[JsonDict] = None
    speech_intent: Optional[JsonDict] = None
    response_text: Optional[str] = None
    response_event_id: Optional[str] = None
    outbound_utterances: List[JsonDict] = field(default_factory=list)
    suppressed_speech_intents: List[JsonDict] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

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

    def to_dict(self) -> JsonDict:
        return {
            "agent_id": self.agent_id,
            "sessions": {key: value.to_dict() for key, value in self.sessions.items()},
            "turns": {key: value.to_dict() for key, value in self.turns.items()},
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: JsonDict) -> "ConversationState":
        return ConversationState(
            agent_id=str(data["agent_id"]),
            sessions={
                key: ConversationSession.from_dict(value)
                for key, value in data.get("sessions", {}).items()
            },
            turns={
                key: ConversationTurn.from_dict(value)
                for key, value in data.get("turns", {}).items()
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
        return AgentState(
            agent_id=data["agent_id"],
            profile=AgentProfile.from_dict(data["profile"]),
            workspace=Workspace.from_dict(data["workspace"]),
            tasks={k: AgentTask.from_dict(v) for k, v in data.get("tasks", {}).items()},
            action_runs={k: ActionRun.from_dict(v) for k, v in data.get("action_runs", {}).items()},
            processed_event_ids=list(data.get("processed_event_ids", [])),
            version=data.get("version", 0),
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
        )


GeneratorDecision = JsonDict
Command = JsonDict
AwaitCondition = JsonDict
