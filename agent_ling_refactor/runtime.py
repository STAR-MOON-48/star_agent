from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import suppress
from time import monotonic
from typing import Any

from agent.protocols import (
    ActionSpec,
    AgentEvent,
    AgentState,
    ConversationTurn,
    JsonDict,
    new_id,
    utc_now,
)
from agent.runtime.action_systems.actions import ActionExecutor, ActionRegistry
from agent.runtime.action_systems.task_system import (
    MULTI_STEP_OBJECTIVE_PURPOSE,
    TERMINAL_TASK_STATES,
)
from agent.runtime.cognition_system import EmotionSystem
from agent.runtime.console import trace_line, trace_text
from agent.runtime.interfaces.model import ModelInterface
from agent.runtime.interfaces.protocol import ProtocolInterface
from agent.runtime.kernel.event_bus import EventBus
from agent.runtime.persistence_system import (
    ConversationStore,
    JsonStateStore,
    MemoryRecord,
    MemoryStore,
)
from agent.runtime.state_systems import MemorySystem

from .activation import ActivationDecision, ModelActivationGate
from .context import ContextComposer, event_to_natural_text
from .control import ControlInbox
from .conversation import ConversationLedger
from .messages import MessagePurpose, NaturalMessage
from .model_gateway import ModelGateway, ModelTurn, ToolCall, action_spec_to_tool
from .prompts import PromptCompiler
from .scheduling import RefactoredTaskSystem
from .settings import RefactorSettings


ReplyHandler = Callable[[str], Any | Awaitable[Any]]


REQUEST_DECISION_SPEC = ActionSpec(
    name="request_decision",
    description="当话语涉及行动、承诺、重要判断或外部状态核实时，请求决策区域继续处理。",
    input_schema={
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "为什么需要进一步决策。"},
            "objective": {"type": "string", "description": "需要决定或推进的事项。"},
        },
        "required": ["reason"],
        "additionalProperties": False,
    },
    source="cognitive_signal",
    side_effect_level="read",
)

REQUEST_EXPRESSION_SPEC = ActionSpec(
    name="request_expression",
    description="当决策结果需要对外说明时，把自然语言表达意图交给表达区域。",
    input_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "希望对外表达的事实、状态或问题。"},
        },
        "required": ["content"],
        "additionalProperties": False,
    },
    source="cognitive_signal",
    side_effect_level="read",
)


class RefactoredRuntime:
    """Event runtime with distinct understanding, decision and expression regions.

    Ordinary conversation intentionally remains a two-stage cognitive path:
    understanding first, expression second.  When understanding also requests
    a decision, decision work starts concurrently with expression so Broca does
    not wait for unrelated action planning.
    """

    DECISION_EVENT_TYPES = {
        "action.completed",
        "action.failed",
        "action.cancelled",
        "action.internal.completed",
        "timer.fired",
        "runtime.continue",
        "runtime.objective",
        "operator.directive",
        "agent.thought",
        "protocol.event",
        "protocol.action",
        "protocol.tool_specification",
    }

    def __init__(
        self,
        *,
        agent_id: str,
        store: JsonStateStore,
        settings: RefactorSettings,
        model: ModelInterface,
        protocol: ProtocolInterface | None = None,
        event_bus: EventBus | None = None,
        on_reply: ReplyHandler | None = None,
        trace: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.store = store
        self.settings = settings
        self.model = model
        self.protocol = protocol
        self.trace = trace
        self.on_reply = on_reply
        self.event_bus = event_bus or EventBus(trace=trace)
        self.registry = ActionRegistry()
        if protocol is not None:
            self.registry.register_many(protocol.list_action_specs())
        self.task_system = RefactoredTaskSystem()
        self.activation = ModelActivationGate(settings.activation)
        self.memory_store = MemoryStore(store.root)
        self.memory_system = MemorySystem(settings.memory, store=self.memory_store)
        self.emotion_system = EmotionSystem(settings.emotion)
        self.action_executor = ActionExecutor(
            self.event_bus,
            self.registry,
            protocol_interface=protocol,
            task_system=self.task_system,
            memory_system=self.memory_system,
            trace=trace,
        )
        self.conversation = ConversationLedger(
            store=ConversationStore(store.root),
            settings=settings.conversation,
        )
        self.gateway = ModelGateway(
            model=model,
            prompts=PromptCompiler(settings.prompts),
            settings=settings.runtime,
        )
        self.context = ContextComposer(settings.runtime)
        self.control_inbox = ControlInbox(store.root, agent_id)
        self._worker: asyncio.Task[None] | None = None
        self._protocol_worker: asyncio.Task[None] | None = None
        self._dmn_worker: asyncio.Task[None] | None = None
        self._memory_worker: asyncio.Task[None] | None = None
        self._control_worker: asyncio.Task[None] | None = None
        self._delayed: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()
        self._stopping = asyncio.Event()
        self._last_activity = monotonic()
        self._reply_queue: asyncio.Queue[str] = asyncio.Queue()

    async def start(self) -> None:
        if self._worker is not None and not self._worker.done():
            return
        self._stopping.clear()
        state = self.store.load_state(self.agent_id)
        self._apply_profile(state)
        reflection_state = state.workspace.variables.get("reflection_runtime")
        if isinstance(reflection_state, dict):
            reflection_state.pop("inflight_event_id", None)
        memory_state = state.workspace.variables.get("memory_system")
        if isinstance(memory_state, dict):
            memory_state.pop("reflection_inflight_event_id", None)
        repaired_waits = self.task_system.sanitize_waits(state)
        if repaired_waits:
            state.workspace.note(
                "启动时移除了没有可满足条件的空等待项：" + "、".join(repaired_waits)
            )
        self.store.save_state(state)
        recovery_events = self._startup_recovery_events(state)
        if self.protocol is not None:
            await self.protocol.start(agent_id=self.agent_id)
            self.registry.register_many(self.protocol.list_action_specs())
            self._protocol_worker = asyncio.create_task(self._protocol_loop())
        self._worker = asyncio.create_task(self._event_loop())
        for recovery_event in recovery_events:
            await self.event_bus.publish(recovery_event)
        if self.settings.dmn.enabled:
            self._dmn_worker = asyncio.create_task(self._dmn_loop())
        if self.settings.memory.enabled and self.settings.memory.reflection_enabled:
            self._memory_worker = asyncio.create_task(self._memory_loop())
        if self.settings.control.enabled:
            self.control_inbox.recover()
            self._control_worker = asyncio.create_task(self._control_loop())

    async def stop(self) -> None:
        self._stopping.set()
        tasks = tuple(
            task
            for task in (
                self._worker,
                self._protocol_worker,
                self._dmn_worker,
                self._memory_worker,
                self._control_worker,
                *tuple(self._delayed),
            )
            if task is not None
        )
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        if self.protocol is not None:
            await self.protocol.stop()

    async def submit_user_message(
        self,
        content: str,
        *,
        sender: str = "local_user",
        conversation_id: str | None = None,
    ) -> AgentEvent:
        event = AgentEvent.make(
            agent_id=self.agent_id,
            type="user.message",
            source="local",
            payload={
                "content": content,
                "sender": sender,
                "conversation_id": conversation_id,
            },
            priority=10,
        )
        await self.event_bus.publish(event)
        return event

    async def submit_objective(self, content: str) -> AgentEvent:
        event = AgentEvent.make(
            agent_id=self.agent_id,
            type="runtime.objective",
            source="runtime",
            payload={"content": content},
            priority=20,
        )
        await self.event_bus.publish(event)
        return event

    async def next_reply(self, timeout: float | None = None) -> str:
        if timeout is None:
            return await self._reply_queue.get()
        return await asyncio.wait_for(self._reply_queue.get(), timeout=timeout)

    def _startup_recovery_events(self, state: AgentState) -> list[AgentEvent]:
        return [
            AgentEvent.make(
                agent_id=state.agent_id,
                type="action.failed",
                source="runtime.recovery",
                task_id=run.task_id,
                action_run_id=run.action_run_id,
                payload={
                    "action_name": run.action_name,
                    "error": {
                        "type": "runtime_restarted",
                        "message": (
                            "The previous process ended before this action reached a terminal "
                            "outcome. Current state must be observed again before retrying."
                        ),
                    },
                },
            )
            for run in state.action_runs.values()
            if run.status in {"created", "running"}
        ]

    async def handle_event(self, event: AgentEvent) -> None:
        emitted: list[AgentEvent] = []
        delayed: list[tuple[AgentEvent, float]] = []
        audit: JsonDict | None = None
        comment = "state-only event"
        model_requested = False
        turn_for_error: ConversationTurn | None = None
        async with self._lock:
            state = self.store.load_state(self.agent_id)
            if event.event_id in state.processed_event_ids:
                return
            self._last_activity = monotonic()
            self.task_system.apply_event(state, event)
            self._apply_profile(state)
            self.emotion_system.observe_event(state, event)
            self.memory_system.observe_event(state, event)
            try:
                if event.type == "user.message":
                    turn = self.conversation.receive(state, event)
                    turn_for_error = turn
                    admission = self.activation.begin_user_turn(state, event)
                    if admission.activate:
                        model_requested = True
                        emitted, delayed, audit = await self._handle_user_turn(
                            state=state,
                            event=event,
                            turn=turn,
                        )
                        comment = "understanding then expression; optional decision runs concurrently"
                    else:
                        await self._send_reply(
                            state=state,
                            event=event,
                            turn=turn,
                            content="我先停一下，当前模型服务正在退避，稍后有新消息时再继续。",
                            final=True,
                        )
                        audit = self.activation.suppression_audit(event, admission)
                        comment = "model activation suppressed"
                elif event.type == "conversation.proactive":
                    if (
                        self.conversation.proactive_is_current(state, event)
                        and not self.activation.in_backoff(state)
                    ):
                        turn = self._turn_from_event(state, event)
                        if turn is not None:
                            model_requested = True
                            delayed, audit = await self._handle_proactive(
                                state=state,
                                event=event,
                                turn=turn,
                            )
                    comment = "proactive expression"
                elif event.type == "reflection.requested":
                    if self.activation.in_backoff(state):
                        admission = ActivationDecision(False, "model provider backoff is active")
                        audit = self.activation.suppression_audit(event, admission)
                        comment = "model activation suppressed"
                    else:
                        model_requested = True
                        emitted, audit = await self._handle_reflection(state, event)
                        comment = "natural-language idle reflection"
                elif event.type == "memory.reflection.requested":
                    if self.activation.in_backoff(state):
                        admission = ActivationDecision(False, "model provider backoff is active")
                        audit = self.activation.suppression_audit(event, admission)
                        comment = "model activation suppressed"
                    else:
                        model_requested = True
                        audit = await self._handle_memory_reflection(state, event)
                        comment = "natural-language memory reflection"
                elif event.type == "operator.note":
                    content = str(event.payload.get("content") or "").strip()
                    if content:
                        state.workspace.note(f"操作员内部备注：{content}")
                    audit = {
                        "internal_control": "note_recorded",
                        "content": content,
                    }
                    comment = "operator note stored without model activation"
                elif self._is_decision_event(event):
                    admission = (
                        self.activation.begin_objective(state, event)
                        if event.type == "runtime.objective"
                        else (
                            self.activation.begin_operator_directive(state, event)
                            if event.type == "operator.directive"
                            else self.activation.evaluate(state, event)
                        )
                    )
                    if admission.activate:
                        model_requested = True
                        emitted, audit = await self._handle_decision_event(state, event)
                        if audit is None:
                            audit = {}
                        audit["activation_reason"] = admission.reason
                        comment = "decision then expression when outward speech is needed"
                    else:
                        audit = self.activation.suppression_audit(event, admission)
                        comment = "model activation suppressed"
                        if self.trace:
                            trace_line(
                                "model.activation",
                                f"suppressed type=[cyan]{event.type}[/cyan] reason={admission.reason}",
                            )
                if model_requested:
                    self.activation.record_success(state)
            except Exception as exc:
                retryable = self.activation.record_error(state, exc)
                error_text = (
                    "当前模型服务请求过于频繁，我已经暂停后续模型唤醒并进入退避。"
                    if retryable
                    else "这次处理没有完成，我已经停止继续触发，等待下一条有效信息。"
                )
                if (
                    event.type in {"user.message", "conversation.proactive"}
                    and (turn_for_error is None or turn_for_error.response_event_id is None)
                ):
                    await self._send_reply(
                        state=state,
                        event=event,
                        turn=turn_for_error,
                        content=error_text,
                        final=True,
                    )
                audit = {
                    "error": error_text,
                    "error_type": type(exc).__name__,
                    "retryable": retryable,
                }
                comment = "refactored runtime error"

            self._clear_background_inflight(state, event)
            self._finish_event(state, event, audit, comment)

        for item in emitted:
            await self.event_bus.publish(item)
        for item, seconds in delayed:
            self._publish_later(item, seconds)

    async def _handle_user_turn(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        turn: ConversationTurn,
    ) -> tuple[list[AgentEvent], list[tuple[AgentEvent, float]], JsonDict]:
        history = self.conversation.recent_context(turn)
        understanding_message = self.conversation.understanding_message(turn, event)
        understood = await self._run_region(
            state=state,
            event=event,
            message=understanding_message,
            history=history,
            specs=[REQUEST_DECISION_SPEC],
        )
        understanding = understood.text or f"对方表达了：{turn.utterance}"
        decision_calls = [
            call for call in understood.tool_calls if call.name == "request_decision"
        ]
        self.conversation.record_understanding(
            turn,
            text=understanding,
            decision_requested=bool(decision_calls),
        )
        # The same natural-language understanding is available to the cheap
        # stateful regions immediately.  They do not need to parse Wernicke's
        # private object schema or wait for Broca/Decision to finish.
        understanding_event = AgentEvent.make(
            agent_id=state.agent_id,
            type="conversation.understanding.ready",
            source="understanding",
            correlation_id=turn.conversation_id,
            causation_id=event.event_id,
            payload={
                "content": understanding,
                "turn_id": turn.turn_id,
                "speaker_id": turn.speaker_id,
            },
        )
        self.emotion_system.observe_event(state, understanding_event)
        self.memory_system.observe_event(state, understanding_event)

        decision_future: asyncio.Task[ModelTurn] | None = None
        if decision_calls:
            request = "；".join(
                str(call.arguments.get("objective") or call.arguments.get("reason") or "")
                for call in decision_calls
            ).strip("；")
            decision_message = self.conversation.decision_message(
                turn=turn,
                event=event,
                understanding=understanding,
                request=request,
            )
            decision_future = asyncio.create_task(
                self._run_region(
                    state=state,
                    event=event,
                    message=decision_message,
                    history=history,
                    specs=self._select_action_specs(decision_message),
                )
            )

        expression_message = self.conversation.expression_message(
            turn=turn,
            event=event,
            understanding=understanding,
        )
        expressed = await self._run_region(
            state=state,
            event=event,
            message=expression_message,
            history=history,
            specs=[],
        )
        reply = expressed.text or "我听到了。"
        self.conversation.record_model_intent(turn, text=reply, used_tools=False)
        await self._send_reply(
            state=state,
            event=event,
            turn=turn,
            content=reply,
            final=decision_future is None,
        )

        emitted: list[AgentEvent] = []
        decision_audit: JsonDict = {}
        if decision_future is not None:
            try:
                decided = await decision_future
            except Exception as exc:
                state.workspace.note(
                    f"决策区域未完成：{type(exc).__name__}: {exc}"
                )
                decision_audit = {"error": f"{type(exc).__name__}: {exc}"}
            else:
                executable_calls = tuple(
                    call
                    for call in decided.tool_calls
                    if call.name != REQUEST_EXPRESSION_SPEC.name
                )
                if executable_calls:
                    emitted.extend(
                        await self._execute_tool_calls(
                            state=state,
                            event=event,
                            turn=turn,
                            calls=executable_calls,
                        )
                    )
                elif decided.text:
                    state.workspace.note(f"决策意见：{decided.text}")
                if not executable_calls:
                    self.conversation.mark_complete(turn)
                decision_audit = {
                    "message": decided.text,
                    "called_capabilities": [call.name for call in decided.tool_calls],
                }

        delayed: list[tuple[AgentEvent, float]] = []
        if decision_future is None:
            follow_up = self.conversation.proactive_event_after_reply(
                state=state,
                turn=turn,
                causation_id=event.event_id,
            )
            if follow_up is not None:
                delayed.append(
                    (follow_up, self.settings.conversation.proactive_interval_seconds)
                )
        return emitted, delayed, {
            "understanding": understanding,
            "outward_reply": reply,
            "decision_requested": bool(decision_calls),
            "decision": decision_audit,
            "response_model_path": ["understanding", "expression"],
        }

    async def _handle_decision_event(
        self,
        state: AgentState,
        event: AgentEvent,
    ) -> tuple[list[AgentEvent], JsonDict]:
        turn = self.conversation.context_from_task(state, event)
        message = NaturalMessage(
            sender="runtime",
            recipient="decision",
            purpose=MessagePurpose.DECISION,
            text=event_to_natural_text(event),
            event_id=event.event_id,
            conversation_id=turn.conversation_id if turn is not None else event.correlation_id,
            turn_id=turn.turn_id if turn is not None else None,
            task_id=event.task_id,
        )
        history = self.conversation.recent_context(turn) if turn is not None else []
        decided = await self._run_region(
            state=state,
            event=event,
            message=message,
            history=history,
            specs=self._select_action_specs(message),
        )
        expression_requests = [
            call
            for call in decided.tool_calls
            if call.name == REQUEST_EXPRESSION_SPEC.name
        ]
        executable_calls = tuple(
            call
            for call in decided.tool_calls
            if call.name != REQUEST_EXPRESSION_SPEC.name
        )
        if executable_calls:
            emitted = await self._execute_tool_calls(
                state=state,
                event=event,
                turn=turn,
                calls=executable_calls,
            )
        else:
            emitted = []

        expression_content = "；".join(
            str(call.arguments.get("content") or "").strip()
            for call in expression_requests
            if str(call.arguments.get("content") or "").strip()
        )
        if not expression_content and not executable_calls:
            expression_content = decided.text
        if expression_content and turn is None:
            state.workspace.note(f"内部决策结果：{expression_content}")
        if expression_content and turn is not None:
            expression = NaturalMessage(
                sender="decision",
                recipient="expression",
                purpose=MessagePurpose.EXPRESSION,
                text=(
                    f"决策区域希望对外表达：{expression_content}。"
                    "请结合最近对话，形成 Ling 此刻自然说出口的话。"
                ),
                event_id=event.event_id,
                conversation_id=turn.conversation_id,
                turn_id=turn.turn_id,
                task_id=event.task_id,
            )
            expressed = await self._run_region(
                state=state,
                event=event,
                message=expression,
                history=history,
                specs=[],
            )
            if expressed.text:
                await self._send_reply(
                    state=state,
                    event=event,
                    turn=turn,
                    content=expressed.text,
                    final=True,
                )
                self._close_task(state, event, expressed.text)
                follow_up = self.conversation.proactive_event_after_reply(
                    state=state,
                    turn=turn,
                    causation_id=event.event_id,
                )
                if follow_up is not None:
                    self._publish_later(
                        follow_up,
                        self.settings.conversation.proactive_interval_seconds,
                    )
        return emitted, {
            "decision": decided.text,
            "called_capabilities": [call.name for call in decided.tool_calls],
        }

    async def _handle_proactive(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        turn: ConversationTurn,
    ) -> tuple[list[tuple[AgentEvent, float]], JsonDict]:
        message = NaturalMessage(
            sender="conversation",
            recipient="expression",
            purpose=MessagePurpose.EXPRESSION,
            text=str(event.payload.get("content") or "自然延续当前话题。"),
            event_id=event.event_id,
            conversation_id=turn.conversation_id,
            turn_id=turn.turn_id,
        )
        expressed = await self._run_region(
            state=state,
            event=event,
            message=message,
            history=self.conversation.recent_context(turn),
            specs=[],
        )
        if expressed.text:
            await self._send_reply(
                state=state,
                event=event,
                turn=turn,
                content=expressed.text,
                final=True,
            )
        delayed: list[tuple[AgentEvent, float]] = []
        follow_up = self.conversation.proactive_event_after_reply(
            state=state,
            turn=turn,
            causation_id=event.event_id,
        )
        if follow_up is not None:
            delayed.append((follow_up, self.settings.conversation.proactive_interval_seconds))
        return delayed, {"outward_reply": expressed.text, "region": "expression"}

    async def _handle_reflection(
        self,
        state: AgentState,
        event: AgentEvent,
    ) -> tuple[list[AgentEvent], JsonDict]:
        message = NaturalMessage(
            sender="runtime",
            recipient="reflection",
            purpose=MessagePurpose.REFLECTION,
            text=event_to_natural_text(event),
            event_id=event.event_id,
        )
        reflected = await self._run_region(
            state=state,
            event=event,
            message=message,
            history=[],
            specs=[],
        )
        text = reflected.text.strip()
        if not text or "无需跟进" in text:
            return [], {"reflection": text, "emitted": False}
        state.workspace.note(f"空闲回顾：{text}")
        thought = AgentEvent.make(
            agent_id=state.agent_id,
            type="agent.thought",
            source="reflection",
            payload={"content": text},
            causation_id=event.event_id,
            priority=self.settings.dmn.thought_priority,
        )
        return [thought], {"reflection": text, "emitted": True}

    async def _handle_memory_reflection(
        self,
        state: AgentState,
        event: AgentEvent,
    ) -> JsonDict:
        message = NaturalMessage(
            sender="runtime",
            recipient="memory",
            purpose=MessagePurpose.MEMORY,
            text=event_to_natural_text(event),
            event_id=event.event_id,
        )
        reflected = await self._run_region(
            state=state,
            event=event,
            message=message,
            history=[],
            specs=[],
        )
        content = reflected.text.strip()
        memory_ids = [str(value) for value in event.payload.get("memory_ids", [])]
        memory_state = state.workspace.variables.setdefault("memory_system", {})
        if not isinstance(memory_state, dict):
            memory_state = {}
            state.workspace.variables["memory_system"] = memory_state
        if content:
            digest = hashlib.sha256(("|".join(memory_ids) + content).encode()).hexdigest()[:20]
            record = self.memory_store.save(
                MemoryRecord(
                    memory_id=f"mem_ref_{digest}",
                    agent_id=state.agent_id,
                    kind="semantic",
                    title=_short_title(content, fallback="经验反思"),
                    content=content,
                    tags=["reflection", "reusable_experience"],
                    source_refs=[{"type": "memory", "id": value} for value in memory_ids],
                    confidence=0.8,
                )
            )
            memory_state["last_reflection_id"] = record.memory_id
        memory_state["pending_reflection_ids"] = []
        memory_state["last_reflection_at"] = utc_now()
        return {"memory_reflection": content, "source_memory_ids": memory_ids}

    async def _run_region(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        message: NaturalMessage,
        history: Sequence[JsonDict],
        specs: Sequence[ActionSpec],
    ) -> ModelTurn:
        context = self.context.compose(
            state=state,
            event=event,
            message=message,
            memory_system=self.memory_system,
            conversation_history=history,
        )
        result = await self.gateway.respond(
            profile=state.profile,
            message=message,
            context=context,
            action_specs=specs,
        )
        self.store.append_generator_log(
            agent_id=state.agent_id,
            event=event.to_dict(),
            context={"handoff": message.audit_record(), "natural_language": context},
            decision={
                "natural_language": result.text,
                "called_capabilities": [call.name for call in result.tool_calls],
            },
            state_version=state.version,
            model_tools=[action_spec_to_tool(spec) for spec in specs],
            model_trace=result.trace,
            comment=f"natural-language {message.purpose.value} region",
        )
        return result

    async def _execute_tool_calls(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        turn: ConversationTurn | None,
        calls: Iterable[ToolCall],
    ) -> list[AgentEvent]:
        calls = tuple(
            call
            for call in calls
            if call.name in {spec.name for spec in self.registry.list_specs()}
        )
        calls = self._ordered_tool_calls(calls)
        external = [
            call for call in calls if self.registry.get(call.name).source != "internal_runtime"
        ]
        task_id = self._resolve_action_task(state, event, turn, external)
        emitted: list[AgentEvent] = []
        for call in calls:
            spec = self.registry.get(call.name)
            selected_task_id = task_id
            if spec.source == "internal_runtime" and call.name == "runtime_create_task":
                selected_task_id = None
            emitted.extend(
                await self.action_executor.start_action(
                    state=state,
                    task_id=selected_task_id,
                    action_name=call.name,
                    args=call.arguments,
                    causation_id=event.event_id,
                    causation_event=event,
                )
            )
        return emitted

    def _ordered_tool_calls(self, calls: Sequence[ToolCall]) -> tuple[ToolCall, ...]:
        """Run waits/completion only after this turn's real work has started."""

        def phase(call: ToolCall) -> int:
            if call.name == "runtime_create_task":
                return 0
            if call.name == "runtime_update_task":
                patch = call.arguments.get("patch")
                status = patch.get("status") if isinstance(patch, dict) else None
                return 2 if status in {"waiting", "blocked"} else 0
            if call.name in {
                "runtime_wait",
                "runtime_complete_task",
                "runtime_cancel_task",
            }:
                return 2
            return 1

        return tuple(sorted(calls, key=phase))

    def _resolve_action_task(
        self,
        state: AgentState,
        event: AgentEvent,
        turn: ConversationTurn | None,
        external_calls: Sequence[ToolCall],
    ) -> str | None:
        current = state.tasks.get(event.task_id or "")
        if current is None or current.status in TERMINAL_TASK_STATES:
            current = self.task_system.active_or_current_task(state)
        if current is not None and current.status not in TERMINAL_TASK_STATES:
            return current.task_id
        if not external_calls:
            return None
        goal = turn.utterance if turn is not None else event_to_natural_text(event)
        continuation: JsonDict = {
            "kind": "refactored_model_tool_call",
            "natural_language_goal": goal,
        }
        if turn is not None:
            continuation.update(
                conversation_id=turn.conversation_id,
                turn_id=turn.turn_id,
                speaker_id=turn.speaker_id,
                recipient_id=turn.speaker_id,
            )
        task = self.task_system.create_task(
            state,
            title=_short_title(goal),
            goal=goal,
            purpose=MULTI_STEP_OBJECTIVE_PURPOSE,
            continuation=continuation,
        )
        return task.task_id

    async def _send_reply(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        turn: ConversationTurn | None,
        content: str,
        final: bool,
    ) -> None:
        content = content.strip()
        if not content:
            return
        recipient = self._recipient(state, event, turn)
        response_event_id = new_id("reply")
        if self.protocol is not None and recipient:
            response_event_id = await self.protocol.send_event(
                agent_id=state.agent_id,
                recipient=recipient,
                event_name="assistant.message",
                data={"content": content},
                causation_id=event.event_id,
            )
        state.workspace.add_transcript("assistant", content, event_id=response_event_id)
        if turn is not None:
            self.conversation.record_sent(
                turn,
                content=content,
                response_event_id=response_event_id,
                final=final,
            )
        await self._reply_queue.put(content)
        if self.on_reply is not None:
            result = self.on_reply(content)
            if inspect.isawaitable(result):
                await result
        if self.trace:
            trace_text("agent.reply", "outbound", content)

    def _close_task(self, state: AgentState, event: AgentEvent, response: str) -> None:
        task = state.tasks.get(event.task_id or "")
        if task is None or task.status in TERMINAL_TASK_STATES:
            return
        if task.active_action_runs or task.waiting_on:
            return
        if event.type in {"action.failed", "action.cancelled"}:
            self.task_system.update_task(
                state,
                task.task_id,
                {
                    "status": "blocked",
                    "error": event.payload.get("error") or event.payload,
                    "progress": {"message": response},
                },
            )
        elif event.type in {"action.completed", "action.internal.completed"}:
            self.task_system.complete_task(
                state,
                task.task_id,
                result={"summary": response, "evidence": event.payload},
            )

    def _is_decision_event(self, event: AgentEvent) -> bool:
        return event.type in self.DECISION_EVENT_TYPES

    def _clear_background_inflight(
        self,
        state: AgentState,
        event: AgentEvent,
    ) -> None:
        if event.type == "reflection.requested":
            value = state.workspace.variables.get("reflection_runtime")
            if isinstance(value, dict) and value.get("inflight_event_id") == event.event_id:
                value.pop("inflight_event_id", None)
        elif event.type == "memory.reflection.requested":
            value = state.workspace.variables.get("memory_system")
            if isinstance(value, dict) and value.get("reflection_inflight_event_id") == event.event_id:
                value.pop("reflection_inflight_event_id", None)

    def _select_action_specs(self, message: NaturalMessage) -> list[ActionSpec]:
        specs = list(self.registry.list_specs())
        limit = self.settings.runtime.max_model_tools
        if len(specs) < limit:
            return [REQUEST_EXPRESSION_SPEC, *specs]
        query = message.text.casefold()
        always = {
            "search_actions",
            "search_memory",
            "read_memory",
            "runtime_create_task",
            "runtime_wait",
            "runtime_update_task",
            "runtime_complete_task",
            "runtime_cancel_task",
        }
        ranked = sorted(
            specs,
            key=lambda spec: (
                spec.name in always,
                spec.source == "star_protocol",
                _tool_relevance(spec, query),
            ),
            reverse=True,
        )
        return [REQUEST_EXPRESSION_SPEC, *ranked[: max(0, limit - 1)]]

    def _turn_from_event(self, state: AgentState, event: AgentEvent) -> ConversationTurn | None:
        turn_id = event.payload.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            return self.conversation.context_from_task(state, event)
        try:
            return self.conversation.store.get_turn(state.agent_id, turn_id)
        except KeyError:
            return None

    def _recipient(
        self,
        state: AgentState,
        event: AgentEvent,
        turn: ConversationTurn | None,
    ) -> str:
        if turn is not None:
            return turn.speaker_id
        for key in ("recipient_id", "sender"):
            if value := event.payload.get(key):
                return str(value)
        task = state.tasks.get(event.task_id or "")
        if task is not None:
            value = task.continuation.get("recipient_id") or task.continuation.get("speaker_id")
            if value:
                return str(value)
        return ""

    def _apply_profile(self, state: AgentState) -> None:
        data = self.settings.profile.to_dict()
        data["agent_id"] = state.agent_id
        state.profile = type(state.profile).from_dict(data)

    def _finish_event(
        self,
        state: AgentState,
        event: AgentEvent,
        audit: JsonDict | None,
        comment: str,
    ) -> None:
        state.mark_processed(event.event_id)
        self.store.save_state(state)
        self.store.append_checkpoint(
            agent_id=state.agent_id,
            event=event.to_dict(),
            decision=audit,
            state_version=state.version,
            comment=comment,
        )

    async def _event_loop(self) -> None:
        while not self._stopping.is_set():
            event = await self.event_bus.next_event()
            try:
                await self.handle_event(event)
            except Exception as exc:
                if self.trace:
                    trace_line("refactor.runtime", f"event failed: {type(exc).__name__}: {exc}")
            finally:
                self.event_bus.task_done()

    async def _protocol_loop(self) -> None:
        assert self.protocol is not None
        while not self._stopping.is_set():
            try:
                event = await self.protocol.next_event(timeout=1)
            except asyncio.TimeoutError:
                self.registry.register_many(self.protocol.list_action_specs())
                continue
            await self.event_bus.publish(event)
            self.registry.register_many(self.protocol.list_action_specs())

    async def _control_loop(self) -> None:
        while not self._stopping.is_set():
            items = self.control_inbox.claim(limit=10)
            if not items:
                await asyncio.sleep(self.settings.control.poll_interval_seconds)
                continue
            for item in items:
                event = item.directive.to_event()
                await self.event_bus.publish(event)
                while not self._stopping.is_set():
                    state = self.store.load_state(self.agent_id)
                    if event.event_id in state.processed_event_ids:
                        self.control_inbox.acknowledge(item)
                        break
                    await asyncio.sleep(self.settings.control.poll_interval_seconds)

    async def _dmn_loop(self) -> None:
        while not self._stopping.is_set():
            await asyncio.sleep(self.settings.dmn.interval_seconds)
            if monotonic() - self._last_activity < self.settings.dmn.idle_after_seconds:
                continue
            async with self._lock:
                state = self.store.load_state(self.agent_id)
                if self.activation.in_backoff(state):
                    continue
                if any(
                    task.status not in TERMINAL_TASK_STATES
                    for task in state.tasks.values()
                ):
                    continue
                if any(run.status in {"created", "running"} for run in state.action_runs.values()):
                    continue
                reflection_state = state.workspace.variables.setdefault(
                    "reflection_runtime", {}
                )
                if not isinstance(reflection_state, dict):
                    reflection_state = {}
                    state.workspace.variables["reflection_runtime"] = reflection_state
                if reflection_state.get("inflight_event_id"):
                    continue
                recent = "；".join(state.workspace.notes[-6:]) or "近期没有新的工作记录。"
                reflection_event = AgentEvent.make(
                    agent_id=self.agent_id,
                    type="reflection.requested",
                    source="runtime",
                    payload={"content": f"请回顾这些近期记录：{recent}"},
                    priority=140,
                )
                reflection_state["inflight_event_id"] = reflection_event.event_id
                self.store.save_state(state)
            await self.event_bus.publish(reflection_event)
            self._last_activity = monotonic()

    async def _memory_loop(self) -> None:
        while not self._stopping.is_set():
            await asyncio.sleep(self.settings.memory.reflection_interval_seconds)
            if monotonic() - self._last_activity < self.settings.dmn.idle_after_seconds:
                continue
            async with self._lock:
                state = self.store.load_state(self.agent_id)
                if self.activation.in_backoff(state):
                    continue
                if any(run.status in {"created", "running"} for run in state.action_runs.values()):
                    continue
                memory_state = state.workspace.variables.get("memory_system")
                if not isinstance(memory_state, dict):
                    continue
                if memory_state.get("reflection_inflight_event_id"):
                    continue
                pending = [str(value) for value in memory_state.get("pending_reflection_ids", [])]
                if len(pending) < self.settings.memory.reflection_min_events:
                    continue
                records = [self.memory_store.get(self.agent_id, value) for value in pending]
                episodes = [record for record in records if record is not None]
                if not episodes:
                    continue
                content = "\n".join(
                    f"- {record.title}：{record.content[:1200]}" for record in episodes
                )
                reflection_event = AgentEvent.make(
                    agent_id=self.agent_id,
                    type="memory.reflection.requested",
                    source="runtime",
                    payload={
                        "content": f"请把这些经历归纳成可复用经验：\n{content}",
                        "memory_ids": pending,
                    },
                    priority=150,
                )
                memory_state["reflection_inflight_event_id"] = reflection_event.event_id
                self.store.save_state(state)
            await self.event_bus.publish(reflection_event)

    def _publish_later(self, event: AgentEvent, seconds: float) -> None:
        async def publish() -> None:
            await asyncio.sleep(seconds)
            await self.event_bus.publish(event)

        task = asyncio.create_task(publish())
        self._delayed.add(task)
        task.add_done_callback(self._delayed.discard)


def _short_title(text: str, *, fallback: str = "推进当前目标") -> str:
    normalized = " ".join(text.split()).strip("#-* ")
    return normalized[:80] if normalized else fallback


def _tool_relevance(spec: ActionSpec, query: str) -> int:
    searchable = f"{spec.name} {spec.description}".casefold()
    score = 20 if spec.name.casefold() in query else 0
    for term in (value for value in query.replace("，", " ").split() if len(value) >= 2):
        if term in searchable:
            score += 2
    return score
