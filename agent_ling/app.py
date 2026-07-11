from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent.config import AgentConfig
from agent.runtime import (
    AgentRuntime,
    ConversationSystem,
    DecisionSystem,
    EmotionSystem,
    JsonStateStore,
    MemorySystem,
    PerceptionSystem,
)
from agent.runtime.interfaces.model import ModelInterface
from agent.runtime.interfaces.protocol import ProtocolInterface
from agent.runtime.kernel.event_bus import EventBus
from agent.runtime.kernel.generator_runtime import GeneratorRuntime

from .config import load_agent_config


@dataclass
class AgentApplication:
    """Concrete agent application assembly: profile/config + runtime."""

    agent_id: str
    config: AgentConfig
    runtime: AgentRuntime


def create_agent_runtime(
    *,
    agent_id: str,
    store: JsonStateStore,
    event_bus: Optional[EventBus] = None,
    generator_runtime: Optional[GeneratorRuntime] = None,
    model_interface: Optional[ModelInterface] = None,
    model_id: Optional[str] = None,
    model_config_path: Optional[str] = None,
    agent_config: Optional[AgentConfig] = None,
    agent_config_path: Optional[str] = None,
    protocol_interface: Optional[ProtocolInterface] = None,
    perception_system: Optional[PerceptionSystem] = None,
    conversation_system: Optional[ConversationSystem] = None,
    memory_system: Optional[MemorySystem] = None,
    emotion_system: Optional[EmotionSystem] = None,
    decision_system: Optional[DecisionSystem] = None,
    trace: bool = True,
) -> AgentApplication:
    config = agent_config or load_agent_config(agent_config_path)
    runtime = AgentRuntime(
        agent_id=agent_id,
        store=store,
        event_bus=event_bus,
        generator_runtime=generator_runtime,
        model_interface=model_interface,
        model_id=model_id,
        model_config_path=model_config_path,
        agent_config=config,
        protocol_interface=protocol_interface,
        perception_system=perception_system,
        conversation_system=conversation_system,
        memory_system=memory_system,
        emotion_system=emotion_system,
        decision_system=decision_system,
        trace=trace,
    )
    return AgentApplication(agent_id=agent_id, config=config, runtime=runtime)
