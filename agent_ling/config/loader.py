from __future__ import annotations

import tomllib
from importlib.resources import files
from pathlib import Path

from agent.config import AgentConfig, merge_config
from agent.protocols import JsonDict


def load_agent_config(path: str | Path | None = None) -> AgentConfig:
    data = _load_packaged_default()
    source = "agent_ling/config/default_agent.toml"
    if path:
        override_path = Path(path)
        with override_path.open("rb") as f:
            data = merge_config(data, tomllib.load(f))
        source = str(override_path)
    return AgentConfig.from_dict(data, source=source)


def _load_packaged_default() -> JsonDict:
    with (files("agent_ling.config") / "default_agent.toml").open("rb") as f:
        return tomllib.load(f)
