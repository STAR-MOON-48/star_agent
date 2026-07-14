from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional
from uuid import uuid4

from menglong import Assistant, Context, System, Tool, User

from agent.protocols import ActionSpec, AgentEvent, JsonDict, ensure_json_dict
from agent.runtime.interfaces import ModelInterface, ProtocolInterface


DEFAULT_SYSTEM_PROMPT = """\
You are a minimal baseline tool-loop agent connected to a Star Protocol environment.
Use the available tools when the objective requires observing or changing the environment.
After each tool result, decide whether another tool call is needed. Continue until the
objective is complete or genuinely cannot be completed. Never invent tool results.
When no more tools are needed, return a concise final answer.
"""


TraceCallback = Callable[[str, JsonDict], None]


@dataclass(frozen=True)
class _ToolCall:
    call_id: str
    name: str
    arguments: JsonDict


@dataclass(frozen=True)
class _Request:
    content: str
    recipient: Optional[str] = None
    causation_id: Optional[str] = None
    conversation_id: Optional[str] = None


class ToolLoopAgent:
    """The smallest useful model-owned tool loop.

    One request is processed at a time. The model calls tools, Star Protocol executes
    them, outcomes are appended to the model context, and the loop stops when the
    model returns text without tool calls.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        model: ModelInterface,
        protocol: ProtocolInterface,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 20,
        max_tokens: int = 2048,
        trace: Optional[TraceCallback] = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.agent_id = agent_id
        self.model = model
        self.protocol = protocol
        self.system_prompt = system_prompt.strip()
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.trace = trace

        self._inbox: asyncio.Queue[_Request] = asyncio.Queue()
        self._pending_actions: dict[str, asyncio.Future[JsonDict]] = {}
        self._event_task: Optional[asyncio.Task[None]] = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self.protocol.start(agent_id=self.agent_id)
        self._started = True
        self._event_task = asyncio.create_task(
            self._route_protocol_events(),
            name=f"{self.agent_id}-baseline-star-events",
        )

    async def stop(self) -> None:
        self._started = False
        if self._event_task is not None:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
            self._event_task = None
        for future in self._pending_actions.values():
            if not future.done():
                future.cancel()
        self._pending_actions.clear()
        await self.protocol.stop()

    async def wait_for_tools(self, timeout: float = 5.0) -> bool:
        """Wait until Star discovery has supplied at least one tool."""

        deadline = asyncio.get_running_loop().time() + max(timeout, 0.0)
        while asyncio.get_running_loop().time() < deadline:
            if list(self.protocol.list_action_specs()):
                return True
            await asyncio.sleep(0.05)
        return bool(list(self.protocol.list_action_specs()))

    async def submit(
        self,
        content: str,
        *,
        recipient: Optional[str] = None,
        causation_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> None:
        await self._inbox.put(
            _Request(
                content=content,
                recipient=recipient,
                causation_id=causation_id,
                conversation_id=conversation_id,
            )
        )

    async def serve(self) -> None:
        """Process submitted and Star-originated user requests forever."""

        if not self._started:
            raise RuntimeError("ToolLoopAgent must be started before serve().")
        while True:
            request = await self._inbox.get()
            try:
                await self.run(
                    request.content,
                    recipient=request.recipient,
                    causation_id=request.causation_id,
                    conversation_id=request.conversation_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error_text = (
                    f"Baseline agent request failed: {type(exc).__name__}: {exc}"
                )
                self._emit_trace("request.failed", {"error": error_text})
                if request.recipient:
                    await self._send_reply(
                        error_text,
                        recipient=request.recipient,
                        causation_id=request.causation_id,
                        conversation_id=request.conversation_id,
                    )
            finally:
                self._inbox.task_done()

    async def run(
        self,
        objective: str,
        *,
        recipient: Optional[str] = None,
        causation_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> str:
        """Run one classic tool loop to a final model response."""

        if not self._started:
            raise RuntimeError("ToolLoopAgent must be started before run().")

        request_id = f"baseline_{uuid4().hex[:12]}"
        context = Context()
        context.add(System(self.system_prompt))
        context.add(User(objective))
        self._emit_trace(
            "request.started",
            {"request_id": request_id, "objective": objective},
        )

        for step in range(1, self.max_steps + 1):
            specs = {spec.name: spec for spec in self.protocol.list_action_specs()}
            model_kwargs: JsonDict = {"max_tokens": self.max_tokens}
            if specs:
                model_kwargs["tools"] = [
                    self._model_tool(spec) for spec in specs.values()
                ]

            result = await self.model.chat(context, **model_kwargs)
            tool_calls = self._extract_tool_calls(result.raw)
            self._emit_trace(
                "model.response",
                {
                    "request_id": request_id,
                    "step": step,
                    "text": result.text,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                        for call in tool_calls
                    ],
                },
            )

            if not tool_calls:
                answer = (
                    result.text or ""
                ).strip() or "The model returned no final answer."
                if recipient:
                    await self._send_reply(
                        answer,
                        recipient=recipient,
                        causation_id=causation_id,
                        conversation_id=conversation_id,
                    )
                self._emit_trace(
                    "request.completed",
                    {"request_id": request_id, "steps": step, "answer": answer},
                )
                return answer

            context.add(
                Assistant(
                    content=result.text or "",
                    actions=[
                        {
                            "id": call.call_id,
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                        for call in tool_calls
                    ],
                )
            )
            for call in tool_calls:
                outcome = await self._execute_tool(
                    call,
                    spec=specs.get(call.name),
                    task_id=request_id,
                    causation_id=causation_id,
                )
                context.add(
                    Tool(
                        tool_id=call.call_id,
                        name=call.name,
                        content=json.dumps(outcome, ensure_ascii=False, default=str),
                    )
                )

        answer = (
            f"Stopped after reaching the maximum of {self.max_steps} tool-loop steps."
        )
        if recipient:
            await self._send_reply(
                answer,
                recipient=recipient,
                causation_id=causation_id,
                conversation_id=conversation_id,
            )
        self._emit_trace(
            "request.max_steps",
            {"request_id": request_id, "max_steps": self.max_steps},
        )
        return answer

    async def _execute_tool(
        self,
        call: _ToolCall,
        *,
        spec: Optional[ActionSpec],
        task_id: str,
        causation_id: Optional[str],
    ) -> JsonDict:
        if spec is None:
            return {"success": False, "error": f"Unknown tool: {call.name}"}
        if spec.requires_approval:
            return {
                "success": False,
                "error": f"Tool requires approval, unsupported by baseline agent: {call.name}",
            }

        action_run_id = f"baseline_run_{uuid4().hex[:12]}"
        future: asyncio.Future[JsonDict] = asyncio.get_running_loop().create_future()
        self._pending_actions[action_run_id] = future
        self._emit_trace(
            "tool.started",
            {
                "action_run_id": action_run_id,
                "name": call.name,
                "arguments": call.arguments,
            },
        )
        try:
            external_id = await self.protocol.send_action(
                agent_id=self.agent_id,
                task_id=task_id,
                action_run_id=action_run_id,
                action_name=call.name,
                args=call.arguments,
                target=spec.target,
                causation_id=causation_id,
            )
            timeout = max(spec.timeout_ms / 1000.0, 0.001)
            outcome = await asyncio.wait_for(future, timeout=timeout)
            outcome.setdefault("external_action_id", external_id)
            self._emit_trace(
                "tool.completed",
                {"action_run_id": action_run_id, "name": call.name, **outcome},
            )
            return outcome
        except asyncio.TimeoutError:
            outcome = {
                "success": False,
                "error": f"Tool timed out after {spec.timeout_ms} ms: {call.name}",
            }
            self._emit_trace(
                "tool.failed",
                {"action_run_id": action_run_id, "name": call.name, **outcome},
            )
            return outcome
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            outcome = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._emit_trace(
                "tool.failed",
                {"action_run_id": action_run_id, "name": call.name, **outcome},
            )
            return outcome
        finally:
            self._pending_actions.pop(action_run_id, None)

    async def _route_protocol_events(self) -> None:
        while True:
            event = await self.protocol.next_event()
            if event.type in {"action.completed", "action.failed"}:
                future = self._pending_actions.get(event.action_run_id or "")
                if future is not None and not future.done():
                    if event.type == "action.completed":
                        future.set_result(
                            {
                                "success": True,
                                "result": ensure_json_dict(event.payload.get("result")),
                            }
                        )
                    else:
                        future.set_result(
                            {
                                "success": False,
                                "error": ensure_json_dict(event.payload.get("error")),
                            }
                        )
                continue

            if event.type == "user.message":
                content = str(event.payload.get("content") or "")
                if content:
                    await self.submit(
                        content,
                        recipient=self._string_or_none(event.payload.get("sender")),
                        causation_id=event.event_id,
                        conversation_id=self._string_or_none(
                            event.payload.get("conversation_id")
                        ),
                    )
                continue

            self._emit_trace(
                "protocol.event",
                {"type": event.type, "payload": event.payload},
            )

    async def _send_reply(
        self,
        content: str,
        *,
        recipient: str,
        causation_id: Optional[str],
        conversation_id: Optional[str],
    ) -> None:
        await self.protocol.send_event(
            agent_id=self.agent_id,
            recipient=recipient,
            event_name="assistant.message",
            data={"content": content, "conversation_id": conversation_id},
            causation_id=causation_id,
        )

    def _model_tool(self, spec: ActionSpec) -> JsonDict:
        parameters = ensure_json_dict(spec.input_schema)
        if not parameters:
            parameters = {"type": "object", "properties": {}}
        elif "type" not in parameters:
            parameters = {"type": "object", "properties": parameters}
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": parameters,
            },
        }

    def _extract_tool_calls(self, raw: Any) -> list[_ToolCall]:
        calls = self._raw_tool_calls(raw)
        parsed: list[_ToolCall] = []
        for index, call in enumerate(calls, start=1):
            function = self._value(call, "function")
            name = self._value(call, "name") or self._value(function, "name")
            if not name:
                continue
            arguments = self._value(call, "arguments")
            if arguments is None:
                arguments = self._value(function, "arguments")
            parsed.append(
                _ToolCall(
                    call_id=str(
                        self._value(call, "id")
                        or f"tool_call_{index}_{uuid4().hex[:8]}"
                    ),
                    name=str(name),
                    arguments=self._arguments(arguments),
                )
            )
        return parsed

    def _raw_tool_calls(self, raw: Any) -> list[Any]:
        if raw is None:
            return []
        calls = self._value(raw, "tool_calls")
        if calls:
            return list(calls)
        output = self._value(raw, "output")
        actions = self._value(output, "actions")
        return list(actions or [])

    def _arguments(self, value: Any) -> JsonDict:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {"raw": value}
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        return {}

    def _value(self, value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)

    def _string_or_none(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value)
        return text if text else None

    def _emit_trace(self, event: str, data: JsonDict) -> None:
        if self.trace is not None:
            self.trace(event, data)
