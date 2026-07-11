from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from ...protocols import AgentState, JsonDict, utc_now


class JsonStateStore:
    """Tiny JSON persistence layer for AgentState and checkpoints."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)

    def _state_path(self, agent_id: str) -> Path:
        return self.root / f"{agent_id}.state.json"

    def load_state(self, agent_id: str) -> AgentState:
        path = self._state_path(agent_id)
        if not path.exists():
            return AgentState.new(agent_id)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return AgentState.from_dict(data) if isinstance(data, dict) else AgentState.new(agent_id)

    def save_state(self, state: AgentState) -> None:
        path = self._state_path(state.agent_id)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def append_checkpoint(
        self,
        *,
        agent_id: str,
        event: JsonDict,
        decision: JsonDict | None,
        state_version: int,
        comment: str = "",
    ) -> None:
        cp_path = self.root / "checkpoints" / f"{agent_id}.jsonl"
        record: Dict[str, Any] = {
            "created_at": utc_now(),
            "agent_id": agent_id,
            "state_version": state_version,
            "event": event,
            "decision": decision,
            "comment": comment,
        }
        with cp_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append_generator_log(
        self,
        *,
        agent_id: str,
        event: JsonDict,
        context: JsonDict,
        decision: JsonDict | None,
        state_version: int,
        model_tools: list[JsonDict] | None = None,
        model_trace: JsonDict | None = None,
        comment: str = "",
    ) -> None:
        log_path = self.root / "logs" / f"{agent_id}.generator.jsonl"
        record: Dict[str, Any] = {
            "created_at": utc_now(),
            "agent_id": agent_id,
            "state_version": state_version,
            "event": event,
            "context": context,
            "model_tools": model_tools or [],
            "model_trace": model_trace or {},
            "decision": decision,
            "comment": comment,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
