from __future__ import annotations

from dataclasses import dataclass, replace

from agent.runtime.interfaces.model import ModelInterface
from agent.runtime.interfaces.protocol import ProtocolInterface
from agent.runtime.interfaces.star_model import StarModel
from agent.runtime.kernel.event_bus import EventBus
from agent.runtime.persistence_system import JsonStateStore

from .runtime import RefactoredRuntime, ReplyHandler
from .settings import RefactorSettings, load_refactor_settings


@dataclass(frozen=True)
class RefactoredApplication:
    agent_id: str
    settings: RefactorSettings
    runtime: RefactoredRuntime


def create_refactored_runtime(
    *,
    agent_id: str,
    store: JsonStateStore,
    model: ModelInterface | None = None,
    model_id: str | None = None,
    model_config_path: str | None = None,
    settings: RefactorSettings | None = None,
    settings_path: str | None = None,
    protocol: ProtocolInterface | None = None,
    event_bus: EventBus | None = None,
    on_reply: ReplyHandler | None = None,
    trace: bool = True,
) -> RefactoredApplication:
    selected = settings or load_refactor_settings(settings_path)
    if model_id:
        selected = replace(
            selected,
            runtime=replace(selected.runtime, model_id=model_id),
        )
    model_interface = model or StarModel(
        default_model_id=selected.runtime.model_id,
        config_path=model_config_path,
    )
    runtime = RefactoredRuntime(
        agent_id=agent_id,
        store=store,
        settings=selected,
        model=model_interface,
        protocol=protocol,
        event_bus=event_bus,
        on_reply=on_reply,
        trace=trace,
    )
    return RefactoredApplication(
        agent_id=agent_id,
        settings=selected,
        runtime=runtime,
    )
