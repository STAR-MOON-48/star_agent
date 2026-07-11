from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocols import AgentProfile, JsonDict, ensure_json_dict


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    system_profile: str
    persona_profile: str
    behavior_profile: str
    identity_profile: str
    background_profile: str
    values_profile: str
    voice_profile: str
    speech_profile: str
    relationship_profile: str
    self_boundaries: str

    def to_agent_profile(self, agent_id: str) -> AgentProfile:
        return AgentProfile(
            agent_id=agent_id,
            name=self.name,
            system_profile=self.system_profile,
            persona_profile=self.persona_profile,
            behavior_profile=self.behavior_profile,
            identity_profile=self.identity_profile,
            background_profile=self.background_profile,
            values_profile=self.values_profile,
            voice_profile=self.voice_profile,
            speech_profile=self.speech_profile,
            relationship_profile=self.relationship_profile,
            self_boundaries=self.self_boundaries,
        )


@dataclass(frozen=True)
class ContextPolicyConfig:
    max_context_tokens: int
    reserve_output_tokens: int
    safety_margin_tokens: int
    compaction_trigger_tokens: int
    compaction_target_tokens: int
    fixed_prompt_reserve_tokens: int
    tool_budget_tokens: int
    summary_trigger_tokens: int
    summary_source_tokens: int
    summary_min_interval_seconds: float
    max_external_tools: int
    preferred_recent_transcript_messages: int
    preferred_recent_notes: int
    preferred_recent_action_runs: int
    chars_per_token: float
    model_summary_enabled: bool

    @staticmethod
    def from_dict(data: JsonDict) -> "ContextPolicyConfig":
        max_context_tokens = max(4096, int(data.get("max_context_tokens", 1_000_000)))
        reserve_output_tokens = max(512, int(data.get("reserve_output_tokens", 4096)))
        safety_margin_tokens = max(256, int(data.get("safety_margin_tokens", 2048)))
        available_input_tokens = max(
            1024,
            max_context_tokens - reserve_output_tokens - safety_margin_tokens,
        )
        compaction_trigger_tokens = max(
            1024,
            min(
                available_input_tokens,
                int(
                    data.get(
                        "compaction_trigger_tokens",
                        available_input_tokens * 0.9,
                    )
                ),
            ),
        )
        compaction_target_tokens = max(
            1024,
            min(
                compaction_trigger_tokens,
                int(
                    data.get(
                        "compaction_target_tokens",
                        available_input_tokens * 0.3,
                    )
                ),
            ),
        )
        return ContextPolicyConfig(
            max_context_tokens=max_context_tokens,
            reserve_output_tokens=reserve_output_tokens,
            safety_margin_tokens=safety_margin_tokens,
            compaction_trigger_tokens=compaction_trigger_tokens,
            compaction_target_tokens=compaction_target_tokens,
            fixed_prompt_reserve_tokens=max(
                1024,
                int(data.get("fixed_prompt_reserve_tokens", 6000)),
            ),
            tool_budget_tokens=max(1024, int(data.get("tool_budget_tokens", 8000))),
            summary_trigger_tokens=max(
                1024,
                int(data.get("summary_trigger_tokens", 8000)),
            ),
            summary_source_tokens=max(
                1024,
                int(data.get("summary_source_tokens", 12000)),
            ),
            summary_min_interval_seconds=max(
                0.0,
                float(data.get("summary_min_interval_seconds", 300.0)),
            ),
            max_external_tools=max(1, int(data.get("max_external_tools", 24))),
            preferred_recent_transcript_messages=max(
                0,
                int(data.get("preferred_recent_transcript_messages", 4)),
            ),
            preferred_recent_notes=max(0, int(data.get("preferred_recent_notes", 3))),
            preferred_recent_action_runs=max(
                0,
                int(data.get("preferred_recent_action_runs", 3)),
            ),
            chars_per_token=max(0.5, float(data.get("chars_per_token", 2.0))),
            model_summary_enabled=bool(data.get("model_summary_enabled", True)),
        )

    @staticmethod
    def empty() -> "ContextPolicyConfig":
        return ContextPolicyConfig.from_dict({"model_summary_enabled": False})

    @property
    def available_input_tokens(self) -> int:
        return max(
            1024,
            self.max_context_tokens
            - self.reserve_output_tokens
            - self.safety_margin_tokens,
        )

    @property
    def target_input_tokens(self) -> int:
        return min(self.available_input_tokens, self.compaction_target_tokens)


@dataclass(frozen=True)
class GeneratorPromptConfig:
    session_id: str
    system_name: str
    description: str
    system_prompt: str
    profile_context_template: str
    user_payload_prefix: str
    context_policy: ContextPolicyConfig

    @staticmethod
    def from_dict(session_id: str, data: JsonDict) -> "GeneratorPromptConfig":
        return GeneratorPromptConfig(
            session_id=session_id,
            system_name=str(data.get("system_name", session_id)),
            description=str(data.get("description", "")),
            system_prompt=str(data.get("system_prompt", "")),
            profile_context_template=str(data.get("profile_context_template", "")),
            user_payload_prefix=str(data.get("user_payload_prefix", "")),
            context_policy=ContextPolicyConfig.from_dict(
                data.get("context", {}) if isinstance(data.get("context"), dict) else {}
            ),
        )

    @staticmethod
    def empty(session_id: str = "decision") -> "GeneratorPromptConfig":
        return GeneratorPromptConfig(
            session_id=session_id,
            system_name=session_id,
            description="",
            system_prompt="",
            profile_context_template="",
            user_payload_prefix="",
            context_policy=ContextPolicyConfig.empty(),
        )


@dataclass(frozen=True)
class GeneratorConfig:
    default_session: str
    sessions: dict[str, GeneratorPromptConfig]

    @staticmethod
    def from_dict(data: JsonDict) -> "GeneratorConfig":
        default_session = str(data.get("default_session") or data.get("default_session_id") or "decision")
        session_configs = data.get("sessions")
        base_prompt = {
            key: value
            for key in (
                "system_name",
                "description",
                "system_prompt",
                "profile_context_template",
                "user_payload_prefix",
                "context",
            )
            if (value := data.get(key)) is not None
        }

        sessions: dict[str, GeneratorPromptConfig] = {}
        if isinstance(session_configs, dict):
            for session_id, session_data in session_configs.items():
                if not isinstance(session_data, dict):
                    continue
                prompt_data = merge_config(base_prompt, session_data)
                sessions[str(session_id)] = GeneratorPromptConfig.from_dict(str(session_id), prompt_data)
        elif base_prompt:
            sessions[default_session] = GeneratorPromptConfig.from_dict(default_session, base_prompt)

        if not sessions:
            sessions[default_session] = GeneratorPromptConfig.empty(default_session)
        if default_session not in sessions:
            default_session = next(iter(sessions))
        return GeneratorConfig(default_session=default_session, sessions=sessions)

    @staticmethod
    def empty(default_session: str = "decision") -> "GeneratorConfig":
        return GeneratorConfig(
            default_session=default_session,
            sessions={default_session: GeneratorPromptConfig.empty(default_session)},
        )

    def prompt_for(self, session_id: str | None = None) -> GeneratorPromptConfig:
        selected = session_id or self.default_session
        return self.sessions.get(selected) or self.sessions[self.default_session]

    @property
    def system_prompt(self) -> str:
        return self.prompt_for().system_prompt

    @property
    def profile_context_template(self) -> str:
        return self.prompt_for().profile_context_template

    @property
    def user_payload_prefix(self) -> str:
        return self.prompt_for().user_payload_prefix


@dataclass(frozen=True)
class StarEntryConfig:
    startup_objective: str


@dataclass(frozen=True)
class DmnConfig:
    enabled: bool
    interval_seconds: float
    idle_after_seconds: float
    unchanged_interval_seconds: float
    max_events_per_cycle: int
    thought_priority: int

    @staticmethod
    def from_dict(data: JsonDict) -> "DmnConfig":
        return DmnConfig(
            enabled=bool(data.get("enabled", False)),
            interval_seconds=float(data.get("interval_seconds", 120.0)),
            idle_after_seconds=float(data.get("idle_after_seconds", 30.0)),
            unchanged_interval_seconds=float(data.get("unchanged_interval_seconds", 900.0)),
            max_events_per_cycle=int(data.get("max_events_per_cycle", 1)),
            thought_priority=int(data.get("thought_priority", 80)),
        )

    @staticmethod
    def empty() -> "DmnConfig":
        return DmnConfig.from_dict({})


@dataclass(frozen=True)
class ConversationConfig:
    enabled: bool
    recent_turn_limit: int
    verbatim_turn_limit: int
    compact_turn_limit: int
    max_wernicke_tool_rounds: int
    workspace_search_limit: int
    proactive_enabled: bool
    proactive_interval_seconds: float
    proactive_burst_messages: int

    @staticmethod
    def from_dict(data: JsonDict) -> "ConversationConfig":
        recent_turn_limit = max(1, int(data.get("recent_turn_limit", 48)))
        verbatim_turn_limit = max(
            1,
            min(recent_turn_limit, int(data.get("verbatim_turn_limit", 6))),
        )
        compact_turn_limit = max(
            verbatim_turn_limit,
            min(recent_turn_limit, int(data.get("compact_turn_limit", 18))),
        )
        return ConversationConfig(
            enabled=bool(data.get("enabled", True)),
            recent_turn_limit=recent_turn_limit,
            verbatim_turn_limit=verbatim_turn_limit,
            compact_turn_limit=compact_turn_limit,
            max_wernicke_tool_rounds=max(
                1,
                int(data.get("max_wernicke_tool_rounds", 6)),
            ),
            workspace_search_limit=max(
                1,
                int(data.get("workspace_search_limit", 12)),
            ),
            proactive_enabled=bool(data.get("proactive_enabled", True)),
            proactive_interval_seconds=max(
                0.1,
                float(data.get("proactive_interval_seconds", 8.0)),
            ),
            proactive_burst_messages=max(
                1,
                min(12, int(data.get("proactive_burst_messages", 3))),
            ),
        )

    @staticmethod
    def empty() -> "ConversationConfig":
        return ConversationConfig.from_dict({})


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool
    auto_capture: bool
    reflection_enabled: bool
    reflection_interval_seconds: float
    reflection_min_events: int
    retrieval_limit: int
    max_pending_events: int

    @staticmethod
    def from_dict(data: JsonDict) -> "MemoryConfig":
        return MemoryConfig(
            enabled=bool(data.get("enabled", True)),
            auto_capture=bool(data.get("auto_capture", True)),
            reflection_enabled=bool(data.get("reflection_enabled", True)),
            reflection_interval_seconds=max(
                30.0,
                float(data.get("reflection_interval_seconds", 300.0)),
            ),
            reflection_min_events=max(1, int(data.get("reflection_min_events", 6))),
            retrieval_limit=max(1, int(data.get("retrieval_limit", 6))),
            max_pending_events=max(10, int(data.get("max_pending_events", 100))),
        )

    @staticmethod
    def empty() -> "MemoryConfig":
        return MemoryConfig.from_dict({"enabled": False})


@dataclass(frozen=True)
class EmotionConfig:
    enabled: bool
    decay_half_life_seconds: float
    sensitivity: float
    history_limit: int

    @staticmethod
    def from_dict(data: JsonDict) -> "EmotionConfig":
        return EmotionConfig(
            enabled=bool(data.get("enabled", True)),
            decay_half_life_seconds=max(
                60.0,
                float(data.get("decay_half_life_seconds", 1800.0)),
            ),
            sensitivity=max(0.1, min(2.0, float(data.get("sensitivity", 1.0)))),
            history_limit=max(5, int(data.get("history_limit", 50))),
        )

    @staticmethod
    def empty() -> "EmotionConfig":
        return EmotionConfig.from_dict({"enabled": False})


@dataclass(frozen=True)
class DecisionConfig:
    enabled: bool
    memory_retrieval_limit: int
    history_limit: int

    @staticmethod
    def from_dict(data: JsonDict) -> "DecisionConfig":
        return DecisionConfig(
            enabled=bool(data.get("enabled", True)),
            memory_retrieval_limit=max(
                1,
                int(data.get("memory_retrieval_limit", 6)),
            ),
            history_limit=max(10, int(data.get("history_limit", 100))),
        )

    @staticmethod
    def empty() -> "DecisionConfig":
        return DecisionConfig.from_dict({"enabled": False})


@dataclass(frozen=True)
class AgentConfig:
    profile: ProfileConfig
    generator: GeneratorConfig
    star: StarEntryConfig
    dmn: DmnConfig
    conversation: ConversationConfig
    memory: MemoryConfig
    emotion: EmotionConfig
    decision: DecisionConfig
    source: str

    @staticmethod
    def from_dict(data: JsonDict, *, source: str) -> "AgentConfig":
        data = ensure_json_dict(data)
        agent = ensure_json_dict(data.get("agent"))
        entrypoints = ensure_json_dict(data.get("entrypoints"))
        cognition = ensure_json_dict(data.get("cognition"))
        state = ensure_json_dict(data.get("state"))
        profile = ensure_json_dict(agent.get("profile"))
        generator = ensure_json_dict(data.get("generator"))
        star = ensure_json_dict(entrypoints.get("star"))
        dmn = ensure_json_dict(cognition.get("dmn"))
        conversation = ensure_json_dict(cognition.get("conversation"))
        memory = ensure_json_dict(state.get("memory"))
        emotion = ensure_json_dict(cognition.get("emotion"))
        decision = ensure_json_dict(cognition.get("decision"))
        return AgentConfig(
            profile=ProfileConfig(
                name=str(profile.get("name", "")),
                system_profile=str(profile.get("system_profile", "")),
                persona_profile=str(profile.get("persona_profile", "")),
                behavior_profile=str(profile.get("behavior_profile", "")),
                identity_profile=str(profile.get("identity_profile", "")),
                background_profile=str(profile.get("background_profile", "")),
                values_profile=str(profile.get("values_profile", "")),
                voice_profile=str(profile.get("voice_profile", "")),
                speech_profile=str(profile.get("speech_profile", "")),
                relationship_profile=str(profile.get("relationship_profile", "")),
                self_boundaries=str(profile.get("self_boundaries", "")),
            ),
            generator=GeneratorConfig.from_dict(generator),
            star=StarEntryConfig(
                startup_objective=str(star.get("startup_objective", "")),
            ),
            dmn=DmnConfig.from_dict(dmn),
            conversation=ConversationConfig.from_dict(conversation),
            memory=MemoryConfig.from_dict(memory),
            emotion=EmotionConfig.from_dict(emotion),
            decision=DecisionConfig.from_dict(decision),
            source=source,
        )

    @staticmethod
    def empty(*, source: str = "empty") -> "AgentConfig":
        return AgentConfig(
            profile=ProfileConfig(
                name="Agent",
                system_profile="",
                persona_profile="",
                behavior_profile="",
                identity_profile="",
                background_profile="",
                values_profile="",
                voice_profile="",
                speech_profile="",
                relationship_profile="",
                self_boundaries="",
            ),
            generator=GeneratorConfig.empty(),
            star=StarEntryConfig(startup_objective=""),
            dmn=DmnConfig.empty(),
            conversation=ConversationConfig.empty(),
            memory=MemoryConfig.empty(),
            emotion=EmotionConfig.empty(),
            decision=DecisionConfig.empty(),
            source=source,
        )


def load_agent_config(path: str | Path) -> AgentConfig:
    config_path = Path(path)
    with config_path.open("rb") as f:
        return AgentConfig.from_dict(tomllib.load(f), source=str(config_path))


def merge_config(base: JsonDict, override: JsonDict) -> JsonDict:
    merged: JsonDict = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = merge_config(current, value)
        else:
            merged[key] = value
    return merged


class SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return ""
