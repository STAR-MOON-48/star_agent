from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.config import DmnConfig, EmotionConfig, MemoryConfig
from agent.protocols import AgentProfile

from .messages import MessagePurpose


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config") / "default_agent.toml"


@dataclass(frozen=True)
class RuntimeSettings:
    model_id: str
    max_context_chars: int
    recent_transcript_items: int
    recent_action_runs: int
    memory_retrieval_limit: int
    max_model_tools: int
    send_action_ack: bool
    model_timeout_seconds: float


@dataclass(frozen=True)
class ConversationSettings:
    recent_turns: int
    proactive_enabled: bool
    proactive_interval_seconds: float
    proactive_burst_messages: int


@dataclass(frozen=True)
class ActivationSettings:
    duplicate_ttl_seconds: float
    max_decision_hops: int
    backoff_initial_seconds: float
    backoff_max_seconds: float


@dataclass(frozen=True)
class ControlSettings:
    enabled: bool
    poll_interval_seconds: float


@dataclass(frozen=True)
class PromptSettings:
    common_rules: str
    role_descriptions: dict[MessagePurpose, str]

    def description_for(self, purpose: MessagePurpose) -> str:
        return self.role_descriptions[purpose]


@dataclass(frozen=True)
class StarSettings:
    startup_objective: str


@dataclass(frozen=True)
class RefactorSettings:
    profile: AgentProfile
    runtime: RuntimeSettings
    conversation: ConversationSettings
    activation: ActivationSettings
    control: ControlSettings
    prompts: PromptSettings
    memory: MemoryConfig
    emotion: EmotionConfig
    dmn: DmnConfig
    star: StarSettings
    source: str


def _table(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _text(value: Any, default: str = "") -> str:
    return str(value if value is not None else default).strip()


def load_refactor_settings(path: str | Path | None = None) -> RefactorSettings:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("rb") as file:
        data = tomllib.load(file)

    profile_data = _table(data.get("profile"))
    runtime_data = _table(data.get("runtime"))
    conversation_data = _table(data.get("conversation"))
    activation_data = _table(data.get("activation"))
    control_data = _table(data.get("control"))
    prompt_data = _table(data.get("prompts"))
    role_data = _table(prompt_data.get("roles"))
    memory_data = _table(data.get("memory"))
    emotion_data = _table(data.get("emotion"))
    dmn_data = _table(data.get("dmn"))
    star_data = _table(data.get("star"))

    profile = AgentProfile(
        agent_id="configured-at-runtime",
        name=_text(profile_data.get("name"), "Ling"),
        system_profile=_text(profile_data.get("system_profile")),
        identity_profile=_text(profile_data.get("identity_profile")),
        background_profile=_text(profile_data.get("background_profile")),
        persona_profile=_text(profile_data.get("persona_profile")),
        values_profile=_text(profile_data.get("values_profile")),
        behavior_profile=_text(profile_data.get("behavior_profile")),
        voice_profile=_text(profile_data.get("voice_profile")),
        speech_profile=_text(profile_data.get("speech_profile")),
        relationship_profile=_text(profile_data.get("relationship_profile")),
        self_boundaries=_text(profile_data.get("self_boundaries")),
    )
    role_descriptions = {
        purpose: _text(role_data.get(purpose.value))
        for purpose in MessagePurpose
    }
    missing_roles = [purpose.value for purpose, text in role_descriptions.items() if not text]
    if missing_roles:
        raise ValueError(f"Missing prompt role descriptions: {', '.join(missing_roles)}")

    return RefactorSettings(
        profile=profile,
        runtime=RuntimeSettings(
            model_id=_text(runtime_data.get("model_id"), "xiaomi/mimo-v2.5"),
            max_context_chars=max(4_000, int(runtime_data.get("max_context_chars", 24_000))),
            recent_transcript_items=max(2, int(runtime_data.get("recent_transcript_items", 10))),
            recent_action_runs=max(1, int(runtime_data.get("recent_action_runs", 5))),
            memory_retrieval_limit=max(1, int(runtime_data.get("memory_retrieval_limit", 4))),
            max_model_tools=max(1, int(runtime_data.get("max_model_tools", 20))),
            send_action_ack=bool(runtime_data.get("send_action_ack", True)),
            model_timeout_seconds=max(1.0, float(runtime_data.get("model_timeout_seconds", 90))),
        ),
        conversation=ConversationSettings(
            recent_turns=max(1, int(conversation_data.get("recent_turns", 12))),
            proactive_enabled=bool(conversation_data.get("proactive_enabled", True)),
            proactive_interval_seconds=max(
                0.1, float(conversation_data.get("proactive_interval_seconds", 8))
            ),
            proactive_burst_messages=max(
                1, min(8, int(conversation_data.get("proactive_burst_messages", 3)))
            ),
        ),
        activation=ActivationSettings(
            duplicate_ttl_seconds=max(
                1.0, float(activation_data.get("duplicate_ttl_seconds", 120))
            ),
            max_decision_hops=max(
                1, int(activation_data.get("max_decision_hops", 6))
            ),
            backoff_initial_seconds=max(
                1.0, float(activation_data.get("backoff_initial_seconds", 15))
            ),
            backoff_max_seconds=max(
                1.0, float(activation_data.get("backoff_max_seconds", 300))
            ),
        ),
        control=ControlSettings(
            enabled=bool(control_data.get("enabled", True)),
            poll_interval_seconds=max(
                0.05, float(control_data.get("poll_interval_seconds", 0.25))
            ),
        ),
        prompts=PromptSettings(
            common_rules=_text(prompt_data.get("common_rules")),
            role_descriptions=role_descriptions,
        ),
        memory=MemoryConfig.from_dict(memory_data),
        emotion=EmotionConfig.from_dict(emotion_data),
        dmn=DmnConfig.from_dict(dmn_data),
        star=StarSettings(startup_objective=_text(star_data.get("startup_objective"))),
        source=str(config_path),
    )
