from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable

from agent.protocols import ActionSpec, AgentProfile, JsonDict, ensure_json_dict_list
from agent.runtime.interfaces.model import ModelInterface, ModelMessage
from agent.runtime.kernel.generator_session import model_response_trace

from .messages import NaturalMessage
from .prompts import PromptCompiler
from .settings import RuntimeSettings


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelTurn:
    text: str
    tool_calls: tuple[ToolCall, ...]
    trace: JsonDict
    elapsed_seconds: float


class ModelGateway:
    """A small model boundary shared by independent cognitive regions."""

    def __init__(
        self,
        *,
        model: ModelInterface,
        prompts: PromptCompiler,
        settings: RuntimeSettings,
    ) -> None:
        self.model = model
        self.prompts = prompts
        self.settings = settings

    async def respond(
        self,
        *,
        profile: AgentProfile,
        message: NaturalMessage,
        context: str,
        action_specs: Iterable[ActionSpec] = (),
    ) -> ModelTurn:
        specs = list(action_specs)
        tools = [action_spec_to_tool(spec) for spec in specs]
        system = self.prompts.system_prompt(
            profile=profile,
            message=message,
            tools_available=bool(tools),
        )
        messages = [
            ModelMessage(role="system", content=system),
            ModelMessage(role="user", content=context),
        ]
        kwargs: JsonDict = {"model": self.settings.model_id}
        if tools:
            kwargs["tools"] = tools
        started = perf_counter()
        result = await asyncio.wait_for(
            self.model.chat(messages, **kwargs),
            timeout=self.settings.model_timeout_seconds,
        )
        elapsed = perf_counter() - started
        trace = model_response_trace(result)
        calls = tuple(
            ToolCall(
                call_id=str(call.get("id") or ""),
                name=str(call.get("name") or ""),
                arguments=(
                    dict(call.get("arguments"))
                    if isinstance(call.get("arguments"), dict)
                    else {}
                ),
            )
            for call in ensure_json_dict_list(trace.get("tool_calls"))
            if call.get("name")
        )
        return ModelTurn(
            text=(result.text or "").strip(),
            tool_calls=calls,
            trace={
                "model": result.model,
                "usage": result.usage,
                "response": trace,
                "elapsed_seconds": round(elapsed, 4),
                "model_call_count": 1,
            },
            elapsed_seconds=elapsed,
        )


def action_spec_to_tool(spec: ActionSpec) -> JsonDict:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }
