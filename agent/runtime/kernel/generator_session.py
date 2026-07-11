from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from ...protocols import GeneratorDecision, JsonDict
from ..interfaces.model import ModelInterface, ModelResult


@dataclass(frozen=True)
class GeneratorRequest:
    """One event-driven request handled by a generator session actor."""

    request_id: str
    context: Any
    runtime_context: JsonDict = field(default_factory=dict)
    model_kwargs: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratorResult:
    """Generator decision plus audit trace for the model boundary."""

    decision: GeneratorDecision
    trace: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequest:
    """Generic model request routed through the generator session actor."""

    request_id: str
    context: Any
    kwargs: JsonDict = field(default_factory=dict)


class GeneratorSession(ABC):
    """Actor-style generator session.

    Public methods enqueue requests. A single worker consumes them in order, so
    model-facing session state has one owner.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, Any, asyncio.Future[Any]]] = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._worker and not self._worker.done():
            return
        self._stopping.clear()
        self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopping.set()
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass

    async def generate(
        self,
        *,
        context: Any,
        runtime_context: JsonDict,
        model_kwargs: Optional[JsonDict] = None,
    ) -> GeneratorDecision:
        return (
            await self.generate_with_trace(
                context=context,
                runtime_context=runtime_context,
                model_kwargs=model_kwargs,
            )
        ).decision

    async def generate_with_trace(
        self,
        *,
        context: Any,
        runtime_context: JsonDict,
        model_kwargs: Optional[JsonDict] = None,
    ) -> GeneratorResult:
        request = GeneratorRequest(
            request_id=f"gr_{uuid4().hex[:12]}",
            context=context,
            runtime_context=runtime_context,
            model_kwargs=model_kwargs or {},
        )
        return await self._enqueue("generate_with_trace", request)

    async def chat(self, *, context: Any, **kwargs: Any) -> ModelResult:
        request = ModelRequest(
            request_id=f"gm_{uuid4().hex[:12]}",
            context=context,
            kwargs=dict(kwargs),
        )
        return await self._enqueue("chat", request)

    async def _enqueue(self, kind: str, request: Any) -> Any:
        if not self._worker or self._worker.done():
            await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        await self._queue.put((kind, request, future))
        return await future

    async def _run(self) -> None:
        while not self._stopping.is_set():
            kind, request, future = await self._queue.get()
            try:
                if kind == "generate_with_trace":
                    result = await self.handle_generate_with_trace(request)
                elif kind == "chat":
                    result = await self.handle_chat(request)
                else:
                    raise ValueError(f"Unknown generator request kind: {kind}")
                if not future.done():
                    future.set_result(result)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            finally:
                self._queue.task_done()

    @abstractmethod
    async def handle_generate(self, request: GeneratorRequest) -> GeneratorDecision:
        """Handle one runtime decision request."""

    async def handle_generate_with_trace(self, request: GeneratorRequest) -> GeneratorResult:
        return GeneratorResult(
            decision=await self.handle_generate(request),
            trace={
                "request_id": request.request_id,
                "model_request": model_request_trace(request),
            },
        )

    async def handle_chat(self, request: ModelRequest) -> ModelResult:
        raise NotImplementedError("This generator session does not support generic chat requests.")


class LLMGeneratorSession(GeneratorSession):
    """Real generator session backed by a model interface."""

    def __init__(self, model_interface: ModelInterface) -> None:
        super().__init__()
        self.model_interface = model_interface

    async def handle_generate(self, request: GeneratorRequest) -> GeneratorDecision:
        return (await self.handle_generate_with_trace(request)).decision

    async def handle_generate_with_trace(self, request: GeneratorRequest) -> GeneratorResult:
        result = await self.model_interface.chat(request.context, **request.model_kwargs)
        trace: JsonDict = {
            "request_id": request.request_id,
            "model_request": model_request_trace(request),
            "model_response": model_response_trace(result),
            "parse": {},
        }
        tool_calls = _extract_tool_calls(result.raw)
        if tool_calls:
            decision = tool_calls_to_generator_decision(
                tool_calls,
                runtime_context=request.runtime_context,
                assistant_text=result.text or "",
            )
            trace["parse"] = {"source": "model_tool_calls"}
        else:
            try:
                decision = parse_generator_decision(result.text or "")
                trace["parse"] = {"source": "json_text"}
            except ValueError as exc:
                recovered_decision = model_text_tool_intent_to_decision(
                    result.text or "",
                    runtime_context=request.runtime_context,
                )
                if recovered_decision is not None:
                    decision = recovered_decision
                    trace["parse"] = {
                        "source": "plain_text_tool_intent",
                        "json_error": str(exc),
                        "recovered_action_names": [
                            command.get("action_name")
                            for command in decision.get("commands", [])
                            if command.get("type") == "start_action"
                        ],
                    }
                else:
                    decision = model_text_to_reply_decision(result.text or "")
                    trace["parse"] = {
                        "source": "plain_text_reply",
                        "json_error": str(exc),
                    }
        trace["parsed_decision"] = decision
        return GeneratorResult(decision=decision, trace=trace)

    async def handle_chat(self, request: ModelRequest) -> ModelResult:
        return await self.model_interface.chat(request.context, **request.kwargs)


def parse_generator_decision(text: str) -> GeneratorDecision:
    payload = _extract_json(text)
    try:
        decision = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Generator returned invalid JSON: {exc}") from exc

    if not isinstance(decision, dict):
        raise ValueError("GeneratorDecision must be a JSON object.")
    commands = decision.get("commands", [])
    if not isinstance(commands, list):
        raise ValueError("GeneratorDecision.commands must be a list.")
    if "decision_summary" not in decision:
        decision["decision_summary"] = ""
    decision["commands"] = commands
    return decision


def model_text_to_reply_decision(text: str) -> GeneratorDecision:
    content = text.strip()
    if not content:
        return {
            "decision_summary": "Model returned no tool calls and no reply text.",
            "commands": [],
        }
    return {
        "decision_summary": "Converted model text into assistant reply.",
        "commands": [
            {
                "type": "reply",
                "content": content,
            }
        ],
    }


def model_text_tool_intent_to_decision(
    text: str,
    *,
    runtime_context: JsonDict,
) -> GeneratorDecision | None:
    content = text.strip()
    if not content:
        return None

    if not _tool_intent_recovery_allowed(runtime_context):
        return None

    candidate_names = [
        name
        for name in _candidate_tool_names(runtime_context)
        if not _tool_requires_arguments(runtime_context, name)
    ]
    mentioned_name = _first_mentioned_tool_intent(content, candidate_names)
    if not mentioned_name:
        return None

    decision = tool_calls_to_generator_decision(
        [
            {
                "id": "plain_text_tool_intent_1",
                "name": mentioned_name,
                "arguments": {},
            }
        ],
        runtime_context=runtime_context,
        assistant_text=content,
    )
    decision["decision_summary"] = (
        "Recovered explicit plain-text tool intent into runtime start_action command."
    )
    return decision


def _extract_json(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        return stripped[start : end + 1]
    return stripped


def model_request_trace(request: GeneratorRequest) -> JsonDict:
    return {
        "request_id": request.request_id,
        "messages": _serialize_messages(request.context),
        "kwargs": _jsonable(request.model_kwargs),
    }


def model_response_trace(result: ModelResult) -> JsonDict:
    return {
        "model": result.model,
        "usage": result.usage,
        "text": result.text,
        "tool_calls": [
            {
                "id": _tool_call_id(tool_call),
                "name": _tool_call_name(tool_call),
                "arguments": _tool_call_arguments(tool_call),
            }
            for tool_call in _extract_tool_calls(result.raw)
        ],
        "raw": _jsonable(result.raw),
    }


def _serialize_messages(context: Any) -> list[Any]:
    messages = getattr(context, "messages", None)
    if messages is None:
        return [_jsonable(context)]
    return [_jsonable(message) for message in messages]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json", exclude_none=True))
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)


def _extract_tool_calls(raw: Any) -> list[Any]:
    if raw is None:
        return []
    tool_calls = getattr(raw, "tool_calls", None)
    if tool_calls:
        return list(tool_calls)
    output = getattr(raw, "output", None)
    actions = getattr(output, "actions", None) if output is not None else None
    return list(actions or [])


def _tool_intent_recovery_allowed(runtime_context: JsonDict) -> bool:
    if runtime_context.get("_model_tools"):
        return True
    tooling = runtime_context.get("tooling")
    return isinstance(tooling, dict) and bool(tooling.get("model_tools_available"))


def _candidate_tool_names(runtime_context: JsonDict) -> list[str]:
    names: list[str] = []

    for tool in runtime_context.get("_model_tools") or []:
        name: Any = None
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict):
                name = function.get("name")
            if not name:
                name = tool.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())

    tooling = runtime_context.get("tooling")
    if isinstance(tooling, dict):
        for name in tooling.get("candidate_action_names") or []:
            if isinstance(name, str) and name.strip():
                names.append(name.strip())

    seen: set[str] = set()
    unique_names: list[str] = []
    for name in names:
        lowered = name.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique_names.append(name)
    return unique_names


def _tool_requires_arguments(runtime_context: JsonDict, name: str) -> bool:
    for tool in runtime_context.get("_model_tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        if function.get("name") != name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            return False
        required = parameters.get("required")
        return isinstance(required, list) and bool(required)
    return False


_TOOL_INTENT_MARKERS = (
    "调用",
    "使用",
    "执行",
    "运行",
    "查看",
    "获取",
    "检查",
    "观察",
    "扫描",
    "列出",
    "查询",
    "call",
    "use",
    "run",
    "execute",
    "invoke",
    "inspect",
    "observe",
    "scan",
    "list",
    "get",
    "fetch",
    "check",
)

_TOOL_INTENT_NEGATIONS = (
    "不要",
    "无需",
    "不需要",
    "不能",
    "避免",
    "没有必要",
    "not ",
    "don't",
    "do not",
    "no need",
    "should not",
    "won't",
    "cannot",
    "avoid",
)


def _first_mentioned_tool_intent(text: str, tool_names: list[str]) -> str | None:
    lowered = text.lower()
    matches: list[tuple[int, str]] = []
    for name in tool_names:
        index = lowered.find(name.lower())
        if index >= 0:
            matches.append((index, name))

    for index, name in sorted(matches, key=lambda item: item[0]):
        if _looks_like_tool_intent(text, index, name):
            return name
    return None


def _looks_like_tool_intent(text: str, index: int, name: str) -> bool:
    lowered = text.lower()
    name_end = index + len(name)
    intent_window = text[max(0, index - 48) : min(len(text), name_end + 48)]
    intent_window_lower = intent_window.lower()
    negation_window = lowered[max(0, index - 32) : index]

    if any(marker in negation_window for marker in _TOOL_INTENT_NEGATIONS):
        return False
    if any(marker in intent_window_lower for marker in _TOOL_INTENT_MARKERS):
        return True

    # Backticked tool names usually mean the model is explicitly talking about a
    # callable surface, but they are not sufficient without a nearby verb above.
    quoted = f"`{name}`"
    return quoted in intent_window and any(
        marker in lowered for marker in ("接下来", "下一步", "让我", "现在", "then", "next")
    )


def tool_calls_to_generator_decision(
    tool_calls: list[Any],
    *,
    runtime_context: JsonDict,
    assistant_text: str = "",
) -> GeneratorDecision:
    commands: list[JsonDict] = []
    reply_text = assistant_text.strip()
    if reply_text:
        commands.append({"type": "reply", "content": reply_text})

    valid_tool_names = set(_candidate_tool_names(runtime_context))
    action_tool_calls: list[Any] = []
    ignored_tool_names: list[str] = []

    for tool_call in tool_calls:
        name = _tool_call_name(tool_call)
        if valid_tool_names and name not in valid_tool_names:
            ignored_tool_names.append(name)
            continue
        action_tool_calls.append(tool_call)

    focus_task_id = runtime_context.get("decision", {}).get("focus_task_id")
    focus_task = runtime_context.get("focus", {}).get("task")
    if not focus_task_id and isinstance(focus_task, dict):
        focus_task_id = focus_task.get("task_id")

    runtime_tool_names = _internal_runtime_tool_names(runtime_context)
    internal_action_tool_calls = [
        tool_call
        for tool_call in action_tool_calls
        if _tool_call_name(tool_call) in runtime_tool_names
    ]
    task_bound_action_tool_calls = [
        tool_call
        for tool_call in action_tool_calls
        if _tool_call_name(tool_call) not in runtime_tool_names
    ]

    for tool_call in internal_action_tool_calls:
        name = _tool_call_name(tool_call)
        args = _tool_call_arguments(tool_call)
        task_ref = args.get("task_ref")
        task_id = args.get("task_id") or focus_task_id
        commands.append(
            {
                "type": "start_action",
                "task_id": task_id if isinstance(task_id, str) and task_id else None,
                "task_ref": (
                    task_ref
                    if name != "runtime_create_task" and isinstance(task_ref, str) and task_ref
                    else None
                ),
                "action_name": name,
                "args": args,
                "mode_hint": None,
            }
        )

    if (
        task_bound_action_tool_calls
        and not focus_task_id
        and _should_create_objective_task(runtime_context)
    ):
        task_ref = "multi_step_objective"
        objective = _objective_text(runtime_context)
        commands.append(
            {
                "type": "create_task",
                "task_ref": task_ref,
                "title": _objective_title(objective),
                "goal": objective or "Continue the Star Protocol objective until it is resolved.",
                "purpose": "Runtime task created for a multi-step model-request objective.",
                "continuation": {
                    "kind": "multi_step_objective",
                    "source": "model_tool_call",
                },
            }
        )
        for tool_call in task_bound_action_tool_calls:
            commands.append(
                {
                    "type": "start_action",
                    "task_ref": task_ref,
                    "task_id": None,
                    "action_name": _tool_call_name(tool_call),
                    "args": _tool_call_arguments(tool_call),
                    "mode_hint": None,
                }
            )
        return {
            "decision_summary": (
                _tool_call_decision_summary(
                    action_count=len(action_tool_calls),
                    ignored_tool_names=ignored_tool_names,
                    suffix="under a durable multi-step task.",
                )
            ),
            "commands": commands,
        }

    for index, tool_call in enumerate(task_bound_action_tool_calls, start=1):
        name = _tool_call_name(tool_call)
        args = _tool_call_arguments(tool_call)
        if focus_task_id:
            commands.append(
                {
                    "type": "start_action",
                    "task_id": focus_task_id,
                    "task_ref": None,
                    "action_name": name,
                    "args": args,
                    "mode_hint": None,
                }
            )
            continue

        task_ref = f"tool_call_{index}"
        commands.append(
            {
                "type": "create_task",
                "task_ref": task_ref,
                "title": f"Run {name}",
                "goal": f"Execute action {name} requested by the model tool call.",
                "purpose": "Runtime task created for a model-request tool call.",
                "continuation": {},
            }
        )
        commands.append(
            {
                "type": "start_action",
                "task_ref": task_ref,
                "task_id": None,
                "action_name": name,
                "args": args,
                "mode_hint": None,
            }
        )

    return {
        "decision_summary": _tool_call_decision_summary(
            action_count=len(action_tool_calls),
            ignored_tool_names=ignored_tool_names,
            suffix="into runtime command(s).",
        ),
        "commands": commands,
    }


def _internal_runtime_tool_names(runtime_context: JsonDict) -> set[str]:
    runtime = runtime_context.get("runtime")
    if not isinstance(runtime, dict):
        return set()
    action_guidance = runtime.get("action_guidance")
    if not isinstance(action_guidance, dict):
        return set()
    names = action_guidance.get("candidate_internal_runtime_action_names")
    if not isinstance(names, list):
        return set()
    return {name for name in names if isinstance(name, str)}


def _tool_call_decision_summary(
    *,
    action_count: int,
    ignored_tool_names: list[str],
    suffix: str,
) -> str:
    parts = []
    if action_count:
        parts.append(f"converted {action_count} model tool call(s) to action command(s)")
    if ignored_tool_names:
        parts.append(f"ignored unknown tool call(s): {', '.join(ignored_tool_names)}")
    if not parts:
        parts.append("model tool calls produced no runtime commands")
    return "Converted model tool call boundary: " + "; ".join(parts) + f" {suffix}"


def _should_create_objective_task(runtime_context: JsonDict) -> bool:
    runtime = runtime_context.get("runtime")
    if isinstance(runtime, dict) and runtime.get("mode") == "star_protocol":
        return True

    objective = _objective_text(runtime_context).lower()
    return any(
        marker in objective
        for marker in (
            "star protocol",
            "star_protocol",
            "持续",
            "直到",
            "长程",
            "连续",
            "推进",
            "目标完成",
        )
    )


def _objective_text(runtime_context: JsonDict) -> str:
    decision = runtime_context.get("decision")
    if isinstance(decision, dict):
        trigger = decision.get("trigger")
        if isinstance(trigger, dict):
            payload = trigger.get("payload")
            if isinstance(payload, dict):
                content = payload.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    for item in reversed(runtime_context.get("evidence") or []):
        if isinstance(item, dict) and item.get("type") == "transcript":
            if item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return ""


def _objective_title(objective: str) -> str:
    normalized = " ".join(objective.split())
    if not normalized:
        return "Continue multi-step objective"
    return normalized


def _tool_call_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("name") or tool_call.get("function", {}).get("name") or "")
    return str(getattr(tool_call, "name", "") or "")


def _tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("id", "") or "")
    return str(getattr(tool_call, "id", "") or "")


def _tool_call_arguments(tool_call: Any) -> JsonDict:
    if isinstance(tool_call, dict):
        arguments = tool_call.get("arguments")
    else:
        arguments = getattr(tool_call, "arguments", None)
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}
