from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Set

from .event_bus import EventBus
from .generator_runtime import GeneratorRuntime
from ...config import AgentConfig, load_agent_config
from ...protocols import AgentEvent, AgentState, AgentTask, GeneratorDecision, JsonDict
from ..action_systems.actions import ActionExecutor, ActionRegistry
from ..action_systems.task_system import TaskSystem
from ..cognition_system import (
    ConversationSystem,
    DecisionSystem,
    DMNSystem,
    EmotionSystem,
)
from ..console import trace_event, trace_json, trace_line, trace_text
from ..interfaces.model import ModelInterface
from ..interfaces.protocol import ProtocolInterface
from ..perception_systems import PerceptionSystem
from ..persistence_system import ConversationStore, MemoryStore
from ..persistence_system.store import JsonStateStore
from ..state_systems import MemorySystem
from ..state_systems.workspace import ContextBuilder


TERMINAL_TASK_STATES = {"completed", "failed", "cancelled"}
MULTI_STEP_OBJECTIVE_PURPOSE = "Runtime task created for a multi-step model-request objective."
RETRYABLE_GENERATOR_ERROR_MARKERS = (
    "429",
    "rate limit",
    "ratelimit",
    "too many requests",
    "timeout",
    "temporarily",
)
UNFINISHED_EXTERNAL_STATUS_VALUES = {
    "in_progress",
    "in progress",
    "running",
    "pending",
    "open",
    "unfinished",
    "not_done",
    "not done",
}
CONVERSATION_SYSTEM_EVENT_TYPES = {
    "user.message",
    "conversation.understanding.requested",
    "conversation.understanding.ready",
    "conversation.speech.requested",
    "conversation.utterance.ready",
    "conversation.utterance.sent",
}


class RuntimePolicy:
    """Thin gate that decides whether an event should activate Generator."""

    BROADCAST_EVENT_WAKE_NAMES = {
        "agent_requested",
        "help_requested",
        "task_assigned",
        "user.message",
        "user_message",
    }

    def __init__(self, *, conversation_enabled: bool = True) -> None:
        self.conversation_enabled = conversation_enabled

    def should_activate_generator(self, state: AgentState, event: AgentEvent) -> bool:
        if event.event_id in state.processed_event_ids:
            return False
        if event.type == "user.message":
            return not self.conversation_enabled
        if event.type == "conversation.decision.requested":
            return True
        if event.type == "action.completed":
            return True
        if event.type == "action.failed":
            return True
        if event.type == "action.cancelled":
            return not bool(event.payload.get("silent"))
        if event.type == "timer.fired":
            return True
        if event.type == "runtime.continue":
            return True
        if event.type == "agent.thought":
            return True
        if event.type == "protocol.tool_specification":
            return any(
                task.status not in TERMINAL_TASK_STATES
                for task in state.tasks.values()
            )
        if event.type == "protocol.event":
            return self._should_activate_protocol_event(event)
        if event.type == "protocol.action":
            return True
        # progress and started update state only; they should not create model storms.
        return False

    def _should_activate_protocol_event(self, event: AgentEvent) -> bool:
        if not event.payload.get("broadcast"):
            return True
        content = event.payload.get("content")
        if not isinstance(content, dict):
            return False
        name = str(content.get("name") or content.get("type") or "").strip().lower()
        return name in self.BROADCAST_EVENT_WAKE_NAMES


class AgentRuntime:
    """Event-driven agent actor runtime.

    LLM/generator is activated by events. Tools/actions are invoked through
    ActionExecutor. Long actions emit future events that resume the agent.
    """

    def __init__(
        self,
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
        dmn_system: Optional[DMNSystem] = None,
        conversation_system: Optional[ConversationSystem] = None,
        memory_system: Optional[MemorySystem] = None,
        emotion_system: Optional[EmotionSystem] = None,
        decision_system: Optional[DecisionSystem] = None,
        trace: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.store = store
        self.event_bus = event_bus or EventBus(trace=trace)
        self.trace = trace
        self.agent_config = (
            agent_config
            or (load_agent_config(agent_config_path) if agent_config_path else AgentConfig.empty())
        )
        self.perception_system = perception_system or PerceptionSystem()
        self.perception_system.bind_event_bus(self.event_bus)
        self.registry = ActionRegistry()
        self.protocol_interface = protocol_interface
        self.registry.register_many(
            self.protocol_interface.list_action_specs() if self.protocol_interface else []
        )
        self.task_system = TaskSystem()
        self.generator_runtime = generator_runtime or GeneratorRuntime(
            model_interface=model_interface,
            default_model_id=model_id,
            config_path=model_config_path,
            agent_config=self.agent_config,
            trace=trace,
        )
        self.memory_system = memory_system or MemorySystem(
            self.agent_config.memory,
            store=MemoryStore(self.store.root),
            context_policy=self.agent_config.generator.prompt_for(
                "memory_reflection"
            ).context_policy,
        )
        self.emotion_system = emotion_system or EmotionSystem(
            self.agent_config.emotion
        )
        self.context_builder = ContextBuilder(
            self.agent_config.generator.prompt_for("decision").context_policy
        )
        self.decision_system = decision_system or DecisionSystem(
            self.agent_config.decision,
            context_builder=self.context_builder,
            memory_system=self.memory_system,
            emotion_system=self.emotion_system,
        )
        self.action_executor = ActionExecutor(
            self.event_bus,
            self.registry,
            protocol_interface=self.protocol_interface,
            task_system=self.task_system,
            memory_system=self.memory_system,
            trace=trace,
        )
        self.conversation_system = conversation_system or ConversationSystem(
            generator_runtime=self.generator_runtime,
            store=ConversationStore(self.store.root),
            config=self.agent_config.conversation,
            memory_system=self.memory_system,
        )
        self.dmn_system = dmn_system or DMNSystem(
            self.agent_config.dmn,
            context_policy=self.agent_config.generator.prompt_for("dmn").context_policy,
            memory_system=self.memory_system,
            emotion_system=self.emotion_system,
        )
        self.policy = RuntimePolicy(
            conversation_enabled=self.agent_config.conversation.enabled,
        )
        self._lock = asyncio.Lock()
        self._worker: Optional[asyncio.Task] = None
        self._protocol_worker: Optional[asyncio.Task] = None
        self._dmn_worker: Optional[asyncio.Task] = None
        self._memory_worker: Optional[asyncio.Task] = None
        self._delayed_event_tasks: Set[asyncio.Task] = set()
        self._max_generator_retries = 3
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._worker and not self._worker.done():
            return
        self._stopping.clear()
        await self.generator_runtime.start()
        if self.protocol_interface:
            await self.protocol_interface.start(agent_id=self.agent_id)
            self.registry.register_many(self.protocol_interface.list_action_specs())
            self._protocol_worker = asyncio.create_task(self._run_protocol_loop())
        self._worker = asyncio.create_task(self._run_loop())
        for event in self._startup_recovery_events():
            await self.event_bus.publish(event)
        if self.dmn_system.enabled:
            self._dmn_worker = asyncio.create_task(self._run_dmn_loop())
        if self.memory_system.enabled and self.agent_config.memory.reflection_enabled:
            self._memory_worker = asyncio.create_task(self._run_memory_loop())

    async def stop(self) -> None:
        self._stopping.set()
        if self._memory_worker:
            self._memory_worker.cancel()
            try:
                await self._memory_worker
            except asyncio.CancelledError:
                pass
        if self._dmn_worker:
            self._dmn_worker.cancel()
            try:
                await self._dmn_worker
            except asyncio.CancelledError:
                pass
        if self._protocol_worker:
            self._protocol_worker.cancel()
            try:
                await self._protocol_worker
            except asyncio.CancelledError:
                pass
        if self.protocol_interface:
            await self.protocol_interface.stop()
        await self.generator_runtime.stop()
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        for task in list(self._delayed_event_tasks):
            task.cancel()
        if self._delayed_event_tasks:
            await asyncio.gather(*self._delayed_event_tasks, return_exceptions=True)
            self._delayed_event_tasks.clear()

    async def submit_user_message(self, content: str) -> None:
        await self.perception_system.perceive_local_user_message(
            agent_id=self.agent_id,
            content=content,
        )

    def _startup_recovery_events(self) -> List[AgentEvent]:
        state = self.store.load_state(self.agent_id)
        events: List[AgentEvent] = []
        for run in state.action_runs.values():
            if run.status not in {"created", "running"}:
                continue
            events.append(
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
                                "The previous process stopped before this action reached a terminal "
                                "outcome. Re-evaluate current environment state before retrying."
                            ),
                        },
                    },
                )
            )
        if events and self.trace:
            trace_line(
                "runtime.recovery",
                f"queued stale action recovery count=[bold]{len(events)}[/bold]",
            )
        return events

    async def _run_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                event = await self.event_bus.next_event(timeout=0.2)
            except asyncio.TimeoutError:
                continue
            try:
                await self.handle_event(event)
            except Exception as exc:
                self._record_runtime_handler_error(event, exc)
                if self.trace:
                    trace_text(
                        "error",
                        "event handler error",
                        f"{type(exc).__name__}: {exc}",
                    )
            finally:
                self.event_bus.task_done()

    def _record_runtime_handler_error(self, event: AgentEvent, exc: Exception) -> None:
        try:
            state = self.store.load_state(self.agent_id)
            state.workspace.note(
                f"runtime handler error event={event.event_id} "
                f"type={type(exc).__name__} message={exc}"
            )
            self.store.save_state(state)
            self.store.append_checkpoint(
                agent_id=self.agent_id,
                event=event.to_dict(),
                decision={
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                },
                state_version=state.version,
                comment="runtime handler error",
            )
        except Exception as log_exc:
            if self.trace:
                trace_text(
                    "error",
                    "runtime error logging failed",
                    f"{type(log_exc).__name__}: {log_exc}",
                )

    async def _run_protocol_loop(self) -> None:
        if self.protocol_interface is None:
            return
        while not self._stopping.is_set():
            try:
                event = await self.protocol_interface.next_event(timeout=0.2)
            except asyncio.TimeoutError:
                continue
            if event.type == "protocol.tool_specification":
                self.registry.register_many(self.protocol_interface.list_action_specs())
            await self.perception_system.publish(event)

    async def _run_memory_loop(self) -> None:
        while not self._stopping.is_set():
            await asyncio.sleep(self.memory_system.poll_interval_seconds())
            async with self._lock:
                state = self.store.load_state(self.agent_id)
                try:
                    reflection = await self.memory_system.maybe_reflect(
                        state=state,
                        generator_runtime=self.generator_runtime,
                    )
                except Exception as exc:
                    self.memory_system.record_error(state, exc)
                    self.store.save_state(state)
                    if self.trace:
                        trace_text(
                            "memory",
                            "reflection error",
                            f"{type(exc).__name__}: {exc}",
                        )
                    continue
                if reflection is None:
                    continue
                if self.trace:
                    self._trace_model_raw_request(
                        reflection.model_trace,
                        title="memory reflection request",
                    )
                    self._trace_model_raw_response(
                        reflection.model_trace,
                        title="memory reflection response",
                    )
                    trace_line(
                        "memory",
                        "stored reflection "
                        f"memory=[cyan]{reflection.record.memory_id}[/cyan] "
                        f"title=[bold]{reflection.record.title}[/bold]",
                    )
                self.store.append_generator_log(
                    agent_id=self.agent_id,
                    event=reflection.trigger_event.to_dict(),
                    context=reflection.public_context,
                    decision={"memory_id": reflection.record.memory_id},
                    state_version=state.version,
                    model_tools=[],
                    model_trace=reflection.model_trace,
                    comment="memory reflection",
                )
                self.store.save_state(state)
                self.store.append_checkpoint(
                    agent_id=self.agent_id,
                    event=reflection.trigger_event.to_dict(),
                    decision={"memory_id": reflection.record.memory_id},
                    state_version=state.version,
                    comment="memory reflection",
                )

    async def _run_dmn_loop(self) -> None:
        while not self._stopping.is_set():
            await asyncio.sleep(self.dmn_system.poll_interval_seconds())
            emitted_events: List[AgentEvent] = []
            async with self._lock:
                state = self.store.load_state(self.agent_id)
                self.task_system.reconcile(state)
                try:
                    reflection = await self.dmn_system.maybe_reflect(
                        state=state,
                        generator_runtime=self.generator_runtime,
                        action_specs=self.registry.list_specs(),
                    )
                except Exception as exc:
                    self.dmn_system.record_error(state, exc)
                    self.store.save_state(state)
                    if self.trace:
                        trace_text(
                            "dmn",
                            "reflection error",
                            f"{type(exc).__name__}: {exc}",
                        )
                    continue

                if reflection is None:
                    continue

                emitted_events = reflection.emitted_events
                if self.trace:
                    self._trace_context_selection(reflection.context)
                    self._trace_model_raw_request(
                        reflection.model_trace,
                        title="dmn raw request",
                    )
                    self._trace_model_raw_response(
                        reflection.model_trace,
                        title="dmn raw response",
                    )
                    self._trace_context_usage(
                        reflection.model_trace,
                        title="dmn context usage",
                    )
                    self._trace_dmn_reflection(reflection.decision, emitted_events)
                self.store.append_generator_log(
                    agent_id=self.agent_id,
                    event=reflection.trigger_event.to_dict(),
                    context=reflection.public_context,
                    decision=reflection.decision,
                    state_version=state.version,
                    model_tools=[],
                    model_trace=reflection.model_trace,
                    comment="dmn reflection",
                )
                self.store.save_state(state)
                self.store.append_checkpoint(
                    agent_id=self.agent_id,
                    event=reflection.trigger_event.to_dict(),
                    decision=reflection.decision,
                    state_version=state.version,
                    comment="dmn reflection",
                )

            for emitted in emitted_events:
                await self.event_bus.publish(emitted)

    async def handle_event(self, event: AgentEvent) -> None:
        async with self._lock:
            state = self.store.load_state(self.agent_id)
            if event.event_id in state.processed_event_ids:
                if self.trace:
                    trace_line(
                        "runtime.policy",
                        f"skip duplicate event type=[cyan]{event.type}[/cyan] id=[cyan]{event.event_id}[/cyan]",
                    )
                return

            if self.trace:
                trace_event("agent.event", event)

            self.task_system.apply_event(state, event)
            self._apply_configured_profile(state)
            self.emotion_system.observe_event(state, event)
            self.memory_system.observe_event(state, event)

            if (
                self.agent_config.conversation.enabled
                and event.type in CONVERSATION_SYSTEM_EVENT_TYPES
            ):
                try:
                    emitted_events, audit = await self._handle_conversation_event(
                        state,
                        event,
                    )
                except Exception as exc:
                    emitted_events, audit = self._handle_conversation_error(
                        state=state,
                        event=event,
                        exc=exc,
                    )
                state.mark_processed(event.event_id)
                self.store.save_state(state)
                self.store.append_checkpoint(
                    agent_id=self.agent_id,
                    event=event.to_dict(),
                    decision=audit,
                    state_version=state.version,
                    comment="conversation system event",
                )
                for emitted in emitted_events:
                    await self.event_bus.publish(emitted)
                return

            context: Optional[JsonDict] = None
            decision: Optional[GeneratorDecision] = None
            generator_trace: JsonDict = {}
            emitted_events: List[AgentEvent] = []

            if self.policy.should_activate_generator(state, event):
                context = self.decision_system.build_context(
                    state=state,
                    event=event,
                    action_specs=self.registry.list_specs(),
                )
                context = await self._maybe_refresh_context_summary(
                    state=state,
                    event=event,
                    context=context,
                )
                public_context = self.generator_runtime.public_context(context)
                model_tools = self.generator_runtime.model_tools(context)
                self._trace_context_selection(context)
                self._trace_internal_decision_context(public_context, model_tools=model_tools)
                try:
                    evaluation = await self.decision_system.evaluate(
                        context=context,
                        generator_runtime=self.generator_runtime,
                    )
                    decision = evaluation.decision
                    generator_trace = evaluation.model_trace
                except Exception as exc:
                    self.decision_system.record_error(state, event=event, exc=exc)
                    decision = {
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    }
                    self.store.append_generator_log(
                        agent_id=self.agent_id,
                        event=event.to_dict(),
                        context=public_context,
                        decision=decision,
                        state_version=state.version,
                        model_tools=model_tools,
                        model_trace=generator_trace,
                        comment="generator error before decision",
                    )
                    emitted_events = self._handle_generator_error(
                        state=state,
                        event=event,
                        exc=exc,
                        public_context=public_context,
                    )
                else:
                    self._clear_generator_retry(state, event, public_context)
                    self._trace_model_raw_request(generator_trace)
                    self._trace_model_raw_response(generator_trace)
                    self._trace_context_usage(generator_trace)
                    self._trace_generator_decision(decision, generator_trace)
                    self.store.append_generator_log(
                        agent_id=self.agent_id,
                        event=event.to_dict(),
                        context=public_context,
                        decision=decision,
                        state_version=state.version,
                        model_tools=model_tools,
                        model_trace=generator_trace,
                        comment="generator decision before apply",
                    )
                    self.conversation_system.record_decision(
                        state=state,
                        event=event,
                        decision=decision,
                    )
                    self.decision_system.record(
                        state,
                        event=event,
                        decision=decision,
                    )
                    emitted_events = await self._apply_decision(state, event, decision)
            else:
                if self.trace:
                    trace_line(
                        "runtime.policy",
                        "state-only event; generator not activated",
                    )

            state.mark_processed(event.event_id)
            self.store.save_state(state)
            self.store.append_checkpoint(
                agent_id=self.agent_id,
                event=event.to_dict(),
                decision=decision,
                state_version=state.version,
                comment="handled event",
            )

        # Publish after state checkpoint to avoid events seeing uncommitted state.
            for emitted in emitted_events:
                await self.event_bus.publish(emitted)

    async def _handle_conversation_event(
        self,
        state: AgentState,
        event: AgentEvent,
    ) -> tuple[List[AgentEvent], JsonDict]:
        if event.type == "user.message":
            requested = self.conversation_system.receive_utterance(state, event)
            if self.trace:
                trace_line(
                    "conversation.manager",
                    "created turn "
                    f"turn=[cyan]{requested.payload.get('turn_id')}[/cyan] "
                    f"speaker=[cyan]{requested.payload.get('speaker_id')}[/cyan]",
                )
            return [requested], {
                "conversation": "utterance_received",
                "turn_id": requested.payload.get("turn_id"),
            }

        if event.type == "conversation.understanding.requested":
            superseded = self.conversation_system.supersede_stale_understanding(
                state=state,
                event=event,
            )
            if superseded is not None:
                if self.trace:
                    trace_line(
                        "conversation.manager",
                        "skipped stale understanding "
                        f"turn=[cyan]{event.payload.get('turn_id')}[/cyan] "
                        f"newer_turn=[cyan]{superseded.get('superseded_by_turn_id')}[/cyan]",
                    )
                return [], {
                    "conversation": "understanding_superseded",
                    **superseded,
                }
            understood = await self.conversation_system.understand(state, event)
            trace = understood.result.trace
            if self.trace:
                self._trace_model_raw_request(trace)
                self._trace_model_raw_response(trace)
                self._trace_context_usage(trace, title="wernicke context usage")
                trace_json(
                    "conversation.wernicke",
                    "understanding",
                    understood.result.understanding.to_dict(),
                )
            self.store.append_generator_log(
                agent_id=self.agent_id,
                event=event.to_dict(),
                context={
                    "conversation": event.payload,
                    "stage": "understanding",
                },
                decision={"understanding": understood.result.understanding.to_dict()},
                state_version=state.version,
                model_tools=understood.result.model_tools,
                model_trace=trace,
                comment="wernicke understanding",
            )
            return [understood.event], {
                "conversation": "understanding_ready",
                "understanding": understood.result.understanding.to_dict(),
            }

        if event.type == "conversation.understanding.ready":
            emitted = self.conversation_system.route_understanding(state, event)
            return emitted, {
                "conversation": "understanding_routed",
                "emitted_event_types": [item.type for item in emitted],
            }

        if event.type == "conversation.speech.requested":
            suppressed = self.conversation_system.suppress_if_newer_turn_is_waiting(
                state=state,
                event=event,
            )
            if suppressed is not None:
                if self.trace:
                    trace_line(
                        "conversation.manager",
                        "suppressed stale speech "
                        f"turn=[cyan]{event.payload.get('turn_id')}[/cyan] "
                        f"newer_turn=[cyan]{suppressed.get('superseded_by_turn_id')}[/cyan]",
                    )
                return [], {
                    "conversation": "speech_suppressed",
                    **suppressed,
                }
            speech = await self.conversation_system.speak(state, event)
            trace = speech.result.trace
            if self.trace:
                self._trace_model_raw_request(trace)
                self._trace_model_raw_response(trace)
                self._trace_context_usage(trace, title="broca context usage")
                trace_text(
                    "conversation.broca",
                    "utterance",
                    speech.result.text,
                )
            self.store.append_generator_log(
                agent_id=self.agent_id,
                event=event.to_dict(),
                context={
                    "conversation": event.payload,
                    "stage": "speech",
                },
                decision={"utterance": speech.result.text},
                state_version=state.version,
                model_tools=[],
                model_trace=trace,
                comment="broca utterance",
            )
            return [speech.event], {
                "conversation": "utterance_ready",
                "content": speech.result.text,
            }

        if event.type == "conversation.utterance.ready":
            suppressed = self.conversation_system.suppress_if_newer_turn_is_waiting(
                state=state,
                event=event,
            )
            if suppressed is not None:
                if self.trace:
                    trace_line(
                        "conversation.manager",
                        "suppressed stale utterance before send "
                        f"turn=[cyan]{event.payload.get('turn_id')}[/cyan] "
                        f"reason=[yellow]{suppressed.get('reason')}[/yellow]",
                    )
                return [], {
                    "conversation": "utterance_suppressed",
                    **suppressed,
                }
            content = str(event.payload.get("content") or "")
            await self._send_reply(state, content, event=event)
            sent = self.conversation_system.mark_sent(state, event)
            return [sent], {
                "conversation": "utterance_sent",
                "content": content,
            }

        if event.type == "conversation.utterance.sent":
            follow_up, delay = self.conversation_system.proactive_follow_up_after_sent(
                state=state,
                event=event,
            )
            if follow_up is None:
                return [], {"conversation": "utterance_settled"}
            self._publish_later(follow_up, delay=delay)
            if self.trace:
                trace_line(
                    "conversation.manager",
                    "scheduled proactive continuation "
                    f"turn=[cyan]{follow_up.payload.get('turn_id')}[/cyan] "
                    f"in={delay:.1f}s",
                )
            return [], {
                "conversation": "proactive_continuation_scheduled",
                "delay_seconds": delay,
                "turn_id": follow_up.payload.get("turn_id"),
            }

        return [], {"conversation": event.type}

    def _handle_conversation_error(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        exc: Exception,
    ) -> tuple[List[AgentEvent], JsonDict]:
        retry_count = int(event.payload.get("conversation_retry_count") or 0) + 1
        retryable = self._is_retryable_generator_error(exc)
        error = {"type": type(exc).__name__, "message": str(exc)}
        state.workspace.note(
            "conversation system error "
            f"event={event.type} retry={retry_count} "
            f"type={error['type']} message={error['message']}"
        )
        if retryable and retry_count <= self._max_generator_retries:
            delay = self._generator_retry_delay(retry_count)
            retry_event = AgentEvent.make(
                agent_id=event.agent_id,
                type=event.type,
                source="runtime.recovery",
                task_id=event.task_id,
                action_run_id=event.action_run_id,
                correlation_id=event.correlation_id,
                causation_id=event.event_id,
                payload={
                    **event.payload,
                    "conversation_retry_count": retry_count,
                    "conversation_retry_error": error,
                },
                priority=event.priority,
            )
            self._publish_later(retry_event, delay=delay)
            if self.trace:
                trace_line(
                    "conversation.manager",
                    "scheduled retry "
                    f"event=[cyan]{event.type}[/cyan] in={delay:.1f}s",
                )
        else:
            self.conversation_system.mark_failed(
                state=state,
                event=event,
                error=error,
            )
            if self.trace:
                trace_text(
                    "error",
                    "conversation system error",
                    f"{error['type']}: {error['message']}",
                )
        return [], {
            "conversation": "generator_error",
            "event_type": event.type,
            "retry_count": retry_count,
            "retryable": retryable,
            "error": error,
        }

    async def _maybe_refresh_context_summary(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        context: JsonDict,
    ) -> JsonDict:
        request = context.get("_summary_request")
        if not isinstance(request, dict):
            return context
        if "context_builder" not in self.agent_config.generator.sessions:
            return context
        try:
            result, summary_trace = await self.generator_runtime.generate_text_with_trace(
                request,
                session_id="context_builder",
            )
            content = str(result.text or "").strip()
            if not content:
                raise ValueError("ContextBuilderSystem returned an empty summary.")
        except Exception as exc:
            self.context_builder.record_summary_error(state, request, exc)
            if self.trace:
                trace_text(
                    "agent.context",
                    "context summary failed",
                    f"{type(exc).__name__}: {exc}",
                )
            return context

        summary = self.context_builder.store_summary(state, request, content)
        self.store.append_generator_log(
            agent_id=self.agent_id,
            event=event.to_dict(),
            context=self.generator_runtime.public_context(request),
            decision={"context_summary": summary},
            state_version=state.version,
            model_tools=[],
            model_trace=summary_trace,
            comment="context builder summary refresh",
        )
        if self.trace:
            self._trace_model_raw_request(
                summary_trace,
                title="context builder raw request",
            )
            self._trace_model_raw_response(
                summary_trace,
                title="context builder raw response",
            )
            self._trace_context_usage(summary_trace, title="context builder usage")
            trace_text(
                "agent.context",
                "context summary stored",
                content,
                subtitle=str(summary.get("summary_id") or ""),
            )
        return self.decision_system.build_context(
            state=state,
            event=event,
            action_specs=self.registry.list_specs(),
        )

    def _apply_configured_profile(self, state: AgentState) -> None:
        configured = self.agent_config.profile.to_agent_profile(state.agent_id)
        profile = state.profile
        for field_name in (
            "name",
            "system_profile",
            "persona_profile",
            "behavior_profile",
            "identity_profile",
            "background_profile",
            "values_profile",
            "voice_profile",
            "speech_profile",
            "relationship_profile",
            "self_boundaries",
        ):
            configured_value = getattr(configured, field_name)
            if configured_value:
                setattr(profile, field_name, configured_value)

    def _handle_generator_error(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        exc: Exception,
        public_context: JsonDict,
    ) -> List[AgentEvent]:
        error = {"type": type(exc).__name__, "message": str(exc)}
        retry_key = self._generator_retry_key(state, event, public_context)
        retry_count = self._increment_generator_retry(state, retry_key)
        retryable = self._is_retryable_generator_error(exc)
        task = state.tasks.get(retry_key) if retry_key in state.tasks else None

        state.workspace.note(
            "generator error "
            f"key={retry_key} retry={retry_count} type={error['type']} message={error['message']}"
        )

        if task and task.status not in TERMINAL_TASK_STATES:
            task.error = {
                "generator_error": error,
                "retry_count": retry_count,
                "retryable": retryable,
            }
            if retryable and retry_count <= self._max_generator_retries:
                delay = self._generator_retry_delay(retry_count)
                task.progress["generator_retry"] = {
                    "message": f"generator retry scheduled in {delay:.1f}s",
                    "retry_count": retry_count,
                }
                task.status = (
                    "waiting"
                    if task.active_action_runs or task.waiting_on
                    else "runnable"
                )
                task.touch()
                if not task.active_action_runs and not task.waiting_on:
                    retry_event = AgentEvent.make(
                        agent_id=state.agent_id,
                        type="runtime.continue",
                        source="runtime",
                        task_id=task.task_id,
                        causation_id=event.event_id,
                        payload={
                            "reason": "generator_error_retry",
                            "retry_count": retry_count,
                            "retry_after_seconds": delay,
                            "error": error,
                        },
                    )
                    self._publish_later(retry_event, delay=delay)
                    if self.trace:
                        trace_line(
                            "runtime.policy",
                            "scheduled generator retry "
                            f"task=[magenta]{task.task_id}[/magenta] "
                            f"in={delay:.1f}s",
                        )
            else:
                task.status = "blocked"
                task.progress["generator_retry"] = {
                    "message": "generator failed; task blocked",
                    "retry_count": retry_count,
                }
                task.touch()
        elif retryable and retry_count <= self._max_generator_retries:
            delay = self._generator_retry_delay(retry_count)
            retry_event = AgentEvent.make(
                agent_id=state.agent_id,
                type="runtime.continue",
                source="runtime",
                task_id=task.task_id if task else None,
                causation_id=event.event_id,
                payload={
                    "reason": "generator_error_retry",
                    "retry_count": retry_count,
                    "retry_after_seconds": delay,
                    "error": error,
                },
            )
            self._publish_later(retry_event, delay=delay)

        return []

    def _generator_retry_key(
        self,
        state: AgentState,
        event: AgentEvent,
        public_context: JsonDict,
    ) -> str:
        if event.task_id:
            return event.task_id
        decision = public_context.get("decision")
        if isinstance(decision, dict):
            focus_task_id = decision.get("focus_task_id")
            if isinstance(focus_task_id, str) and focus_task_id:
                return focus_task_id
        if state.workspace.current_task_id:
            return state.workspace.current_task_id
        return "__agent__"

    def _increment_generator_retry(self, state: AgentState, retry_key: str) -> int:
        counts = state.workspace.variables.setdefault("generator_retry_counts", {})
        if not isinstance(counts, dict):
            counts = {}
            state.workspace.variables["generator_retry_counts"] = counts
        retry_count = int(counts.get(retry_key, 0)) + 1
        counts[retry_key] = retry_count
        return retry_count

    def _clear_generator_retry(
        self,
        state: AgentState,
        event: AgentEvent,
        public_context: JsonDict,
    ) -> None:
        retry_key = self._generator_retry_key(state, event, public_context)
        counts = state.workspace.variables.get("generator_retry_counts")
        if isinstance(counts, dict) and retry_key in counts:
            counts.pop(retry_key, None)
        task = state.tasks.get(retry_key)
        if task is not None:
            task.progress.pop("generator_retry", None)
            if str(task.progress.get("message") or "").startswith("generator retry"):
                task.progress.pop("message", None)
                task.progress.pop("retry_count", None)
            if isinstance(task.error, dict) and "generator_error" in task.error:
                task.error = None
            task.touch()

    def _is_retryable_generator_error(self, exc: Exception) -> bool:
        text = f"{type(exc).__name__} {exc}".lower()
        return any(marker in text for marker in RETRYABLE_GENERATOR_ERROR_MARKERS)

    def _generator_retry_delay(self, retry_count: int) -> float:
        return min(30.0, 2.0 ** retry_count)

    def _publish_later(self, event: AgentEvent, *, delay: float) -> None:
        async def publish_after_delay() -> None:
            try:
                await asyncio.sleep(delay)
                if not self._stopping.is_set():
                    await self.event_bus.publish(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self.trace:
                    trace_text(
                        "error",
                        "delayed event publish error",
                        f"{type(exc).__name__}: {exc}",
                    )

        task = asyncio.create_task(publish_after_delay())
        self._delayed_event_tasks.add(task)
        task.add_done_callback(self._delayed_event_tasks.discard)

    async def _apply_decision(
        self,
        state: AgentState,
        event: AgentEvent,
        decision: GeneratorDecision,
    ) -> List[AgentEvent]:
        ref_map: Dict[str, str] = {}
        emitted_events: List[AgentEvent] = []
        sent_reply = False
        completed_task_ids: List[str] = []
        deferred_completion_task_ids: List[str] = []
        task_graph_changed = False
        has_action_command = any(
            isinstance(command, dict) and command.get("type") == "start_action"
            for command in decision.get("commands", [])
        )

        for command in decision.get("commands", []):
            ctype = command.get("type")

            if ctype == "reply":
                content = str(command.get("content", ""))
                suppress_reply = has_action_command or event.type == "protocol.tool_specification"
                if content and not suppress_reply:
                    emitted_events.append(
                        self.conversation_system.request_speech(
                            state=state,
                            event=event,
                            content=content,
                        )
                    )
                    sent_reply = True
                elif content and self.trace:
                    trace_line(
                        "conversation.manager",
                        "kept decision text internal while actions continue "
                        f"event=[cyan]{event.type}[/cyan]",
                    )
                continue

            if ctype == "create_task":
                continuation = dict(command.get("continuation") or {})
                reply_recipient = self._reply_recipient_for_event(state, event)
                if reply_recipient:
                    continuation.setdefault("reply_recipient", reply_recipient)
                self.conversation_system.attach_context_to_continuation(
                    state,
                    event,
                    continuation,
                )
                task = self.task_system.create_task(
                    state,
                    title=str(command.get("title", "Untitled task")),
                    goal=str(command.get("goal", "")),
                    purpose=str(command.get("purpose", "")),
                    parent_task_id=self._parent_task_id_for_create(state, command),
                    dependencies=command.get("dependencies") or [],
                    continuation=continuation,
                )
                task_ref = command.get("task_ref")
                if task_ref:
                    ref_map[str(task_ref)] = task.task_id
                if self.trace:
                    trace_line(
                        "task.system",
                        f"create task=[magenta]{task.task_id}[/magenta] "
                        f"title=[bold]{task.title}[/bold]",
                    )
                task_graph_changed = True
                continue

            if ctype == "start_action":
                action_name = str(command["action_name"])
                if self._is_internal_runtime_action(action_name):
                    task_id = self._resolve_optional_action_task_id(state, command, ref_map)
                else:
                    task_id = self._resolve_or_create_action_task_id(
                        state,
                        event,
                        command,
                        ref_map,
                    )
                if task_id and task_id in state.tasks:
                    self.conversation_system.attach_context_to_continuation(
                        state,
                        event,
                        state.tasks[task_id].continuation,
                    )
                try:
                    action_events = await self.action_executor.start_action(
                        state=state,
                        task_id=task_id,
                        action_name=action_name,
                        args=command.get("args") or {},
                        mode_hint=command.get("mode_hint"),
                        causation_id=event.event_id,
                        causation_event=event,
                    )
                except KeyError as exc:
                    state.workspace.note(
                        "ignored generator start_action for unknown action "
                        f"{action_name}: {exc}"
                    )
                    if self.trace:
                        trace_line(
                            "runtime.policy",
                            "ignored unknown action "
                            f"action=[bold]{action_name}[/bold]",
                        )
                    continue
                internal_events = [
                    action_event
                    for action_event in action_events
                    if action_event.type.startswith("action.internal.")
                ]
                for internal_event in internal_events:
                    self._apply_internal_action_event(
                        internal_event,
                        ref_map=ref_map,
                        completed_task_ids=completed_task_ids,
                        deferred_completion_task_ids=deferred_completion_task_ids,
                    )
                if internal_events:
                    task_graph_changed = True
                action_events = [
                    action_event
                    for action_event in action_events
                    if not action_event.type.startswith("action.internal.")
                ]
                # Resolve wait condition generated later in the same decision.
                if action_events:
                    command["_last_action_run_id"] = action_events[0].action_run_id
                    if command.get("task_ref"):
                        ref_map[f"last_run:{command['task_ref']}"] = action_events[0].action_run_id or ""
                emitted_events.extend(action_events)
                continue

            if ctype == "wait":
                task_id = self._resolve_task_id(state, command, ref_map)
                condition = dict(command.get("condition") or {})
                # If the generator used action_name-level waiting, bind it to the latest run.
                task_ref = command.get("task_ref")
                if task_ref and not condition.get("action_run_id"):
                    run_id = ref_map.get(f"last_run:{task_ref}")
                    if run_id:
                        condition["action_run_id"] = run_id
                self.task_system.add_wait(state, task_id, condition)
                task_graph_changed = True
                if self.trace:
                    trace_line(
                        "task.system",
                        f"wait task=[magenta]{task_id}[/magenta] "
                        f"condition={json.dumps(condition, ensure_ascii=False)}"
                    )
                continue

            if ctype == "update_task":
                task_id = self._resolve_task_id(state, command, ref_map)
                self.task_system.update_task(state, task_id, command.get("patch") or {})
                task_graph_changed = True
                continue

            if ctype == "complete_task":
                task_id = self._resolve_task_id(state, command, ref_map)
                if self._should_defer_task_completion(state, event, task_id, command):
                    task = state.tasks[task_id]
                    self.task_system.defer_completion(
                        state,
                        task_id,
                        reason="external evidence still indicates unfinished work",
                        blockers=[{"kind": "unfinished_external_evidence"}],
                    )
                    deferred_completion_task_ids.append(task_id)
                    if self.trace:
                        trace_line(
                            "task.system",
                            "defer completion "
                            f"[magenta]{task_id}[/magenta]"
                        )
                    continue
                completion = self.task_system.complete_task(
                    state,
                    task_id,
                    command.get("result"),
                )
                task_graph_changed = True
                if completion.get("completed"):
                    completed_task_ids.append(task_id)
                    if self.trace:
                        trace_line("task.system", f"complete task=[magenta]{task_id}[/magenta]")
                else:
                    deferred_completion_task_ids.append(task_id)
                    if self.trace:
                        trace_line(
                            "task.system",
                            "defer completion "
                            f"task=[magenta]{task_id}[/magenta] "
                            f"blockers={json.dumps(completion.get('blockers', []), ensure_ascii=False)}",
                        )
                continue

            if ctype == "cancel_task":
                task_id = self._resolve_task_id(state, command, ref_map)
                reason = str(command.get("reason", "cancelled"))
                cancel_events = await self.action_executor.cancel_action_runs(
                    state=state,
                    task_id=task_id,
                    reason=reason,
                    silent=True,
                    include_descendants=True,
                )
                self.task_system.cancel_task(state, task_id, reason)
                task_graph_changed = True
                emitted_events.extend(cancel_events)
                if self.trace:
                    trace_line(
                        "task.system",
                        f"cancel task=[magenta]{task_id}[/magenta] "
                        f"reason={reason}"
                    )
                continue

            raise ValueError(f"Unknown command type: {ctype}")

        if (
            not sent_reply
            and completed_task_ids
            and event.type == "action.completed"
            and self._should_auto_reply_on_completion(state, completed_task_ids)
        ):
            emitted_events.append(
                self.conversation_system.request_speech(
                    state=state,
                    event=event,
                    content=self._completion_reply_content(event),
                    kind="task_completion_response",
                )
            )

        if not emitted_events:
            task_id = self._next_runnable_task_id(state)
            should_continue = (
                event.type != "runtime.continue"
                or bool(completed_task_ids)
                or self._deferred_completion_should_continue(
                    state,
                    event,
                    deferred_completion_task_ids,
                    next_task_id=task_id,
                )
                or (task_graph_changed and task_id != event.task_id)
            )
            if task_id and should_continue:
                emitted_events.append(
                    AgentEvent.make(
                        agent_id=state.agent_id,
                        type="runtime.continue",
                        source="runtime",
                        task_id=task_id,
                        causation_id=event.event_id,
                        payload={
                            "reason": "decision left runnable work without emitted events",
                        },
                    )
                )
                if self.trace:
                    trace_line(
                        "runtime.policy",
                        "emit runtime.continue "
                        f"task=[magenta]{task_id}[/magenta]",
                    )
            elif (
                event.type == "runtime.continue"
                and task_id
                and state.tasks[task_id].status == "runnable"
            ):
                stall_reason = (
                    "completion was deferred repeatedly without a different runnable task"
                    if deferred_completion_task_ids
                    else "decision produced no action, wait, completion, cancellation, or task transition"
                )
                self.task_system.mark_stalled(
                    state,
                    task_id,
                    reason=stall_reason,
                    decision_summary=str(decision.get("decision_summary") or ""),
                )
                if self.trace:
                    trace_line(
                        "runtime.policy",
                        "mark stalled "
                        f"task=[magenta]{task_id}[/magenta] "
                        f"reason={stall_reason}",
                    )

        return emitted_events

    def _should_auto_reply_on_completion(
        self,
        state: AgentState,
        completed_task_ids: List[str],
    ) -> bool:
        replyable_task_ids = [
            task_id
            for task_id in completed_task_ids
            if not self._has_nonterminal_parent_task(state, task_id)
        ]
        if not replyable_task_ids:
            return False
        if state.workspace.transcript:
            return True
        return any(
            bool(state.tasks[task_id].continuation.get("reply_recipient"))
            for task_id in replyable_task_ids
            if task_id in state.tasks
        )

    def _is_internal_runtime_action(self, action_name: str) -> bool:
        try:
            return self.registry.get(action_name).source == "internal_runtime"
        except KeyError:
            return False

    def _resolve_optional_action_task_id(
        self,
        state: AgentState,
        command: JsonDict,
        ref_map: Dict[str, str],
    ) -> Optional[str]:
        if command.get("task_id") or command.get("task_ref") or state.workspace.current_task_id:
            return self._resolve_task_id(state, command, ref_map)
        return None

    def _apply_internal_action_event(
        self,
        event: AgentEvent,
        *,
        ref_map: Dict[str, str],
        completed_task_ids: List[str],
        deferred_completion_task_ids: List[str],
    ) -> None:
        payload = event.payload or {}
        result = payload.get("result")
        if not isinstance(result, dict):
            result = {}

        result_task_id = result.get("task_id") or event.task_id
        task_ref = result.get("task_ref")
        if isinstance(task_ref, str) and isinstance(result_task_id, str):
            ref_map[task_ref] = result_task_id

        if payload.get("internal_command_type") != "complete_task":
            return
        if not isinstance(result_task_id, str):
            return
        if payload.get("deferred") or result.get("deferred"):
            deferred_completion_task_ids.append(result_task_id)
            return
        completed_task_ids.append(result_task_id)

    def _has_nonterminal_parent_task(self, state: AgentState, task_id: str) -> bool:
        task = state.tasks.get(task_id)
        if task is None or not task.parent_task_id:
            return False
        parent = state.tasks.get(task.parent_task_id)
        return bool(parent and parent.status not in TERMINAL_TASK_STATES)

    def _completion_reply_content(self, event: AgentEvent) -> str:
        result = event.payload.get("result")
        if isinstance(result, dict):
            summary = result.get("summary") or result.get("message")
            if summary:
                return str(summary)
        return "任务已完成。"

    def _next_runnable_task_id(self, state: AgentState) -> Optional[str]:
        return self.task_system.next_runnable_task_id(
            state,
            preferred_task_id=state.workspace.current_task_id,
        )

    def _deferred_completion_should_continue(
        self,
        state: AgentState,
        event: AgentEvent,
        task_ids: List[str],
        *,
        next_task_id: Optional[str],
    ) -> bool:
        if not task_ids:
            return False
        if next_task_id and next_task_id != event.task_id:
            return True
        for task_id in task_ids:
            task = state.tasks.get(task_id)
            deferred = task.progress.get("completion_deferred") if task else None
            if isinstance(deferred, dict) and int(deferred.get("attempt_count", 0)) <= 1:
                return True
        return False

    def _should_defer_task_completion(
        self,
        state: AgentState,
        event: AgentEvent,
        task_id: str,
        command: JsonDict,
    ) -> bool:
        task = state.tasks.get(task_id)
        if task is None or not self._is_multi_step_objective_task(task):
            return False
        evidence_values = [
            event.payload,
            event.payload.get("result") if isinstance(event.payload, dict) else None,
            command.get("result"),
        ]
        return any(
            self._contains_unfinished_external_status(value)
            for value in evidence_values
        )

    def _contains_unfinished_external_status(self, value: Any, *, depth: int = 0) -> bool:
        if value is None or depth > 4:
            return False
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key).lower()
                if (
                    key_text in {"task_status", "status", "state", "phase"}
                    and isinstance(item, str)
                    and item.strip().lower() in UNFINISHED_EXTERNAL_STATUS_VALUES
                ):
                    return True
                if (
                    key_text in {"done", "completed", "finished"}
                    and item is False
                ):
                    return True
                if self._contains_unfinished_external_status(item, depth=depth + 1):
                    return True
            return False
        if isinstance(value, list):
            return any(
                self._contains_unfinished_external_status(item, depth=depth + 1)
                for item in value
            )
        return False

    def _parent_task_id_for_create(self, state: AgentState, command: JsonDict) -> Optional[str]:
        explicit_parent_id = command.get("parent_task_id")
        if isinstance(explicit_parent_id, str) and explicit_parent_id in state.tasks:
            return explicit_parent_id

        current_task_id = state.workspace.current_task_id
        current = state.tasks.get(current_task_id) if current_task_id else None
        if current is None or current.status in TERMINAL_TASK_STATES:
            return None
        if self._is_multi_step_objective_task(current):
            return current.task_id
        parent = state.tasks.get(current.parent_task_id) if current.parent_task_id else None
        if (
            parent
            and parent.status not in TERMINAL_TASK_STATES
            and self._is_multi_step_objective_task(parent)
        ):
            return parent.task_id
        return None

    def _is_multi_step_objective_task(self, task: AgentTask) -> bool:
        return (
            task.purpose == MULTI_STEP_OBJECTIVE_PURPOSE
            or task.continuation.get("kind") == "multi_step_objective"
        )

    def _resolve_or_create_action_task_id(
        self,
        state: AgentState,
        event: AgentEvent,
        command: JsonDict,
        ref_map: Dict[str, str],
    ) -> str:
        try:
            return self._resolve_task_id(state, command, ref_map)
        except KeyError as exc:
            if command.get("task_ref") and state.workspace.current_task_id:
                current = state.tasks.get(state.workspace.current_task_id)
                if current and current.status not in TERMINAL_TASK_STATES:
                    state.workspace.note(
                        "Recovered unresolved task_ref by using current task "
                        f"{current.task_id}: {command.get('task_ref')}"
                    )
                    return current.task_id

            action_name = str(command.get("action_name", "unknown_action"))
            as_objective = self._should_recover_action_as_objective_task(
                state=state,
                event=event,
                action_name=action_name,
            )
            objective = self._latest_user_objective(state)
            continuation: JsonDict = {
                "recovered_from": str(exc),
                "source_event_id": event.event_id,
            }
            if as_objective:
                continuation.update(
                    {
                        "kind": "multi_step_objective",
                        "source": "recovered_start_action",
                    }
                )
            reply_recipient = self._reply_recipient_for_event(state, event)
            if reply_recipient:
                continuation["reply_recipient"] = reply_recipient
            self.conversation_system.attach_context_to_continuation(
                state,
                event,
                continuation,
            )
            task = self.task_system.create_task(
                state,
                title=(
                    self._objective_title(objective)
                    if as_objective
                    else f"Run {action_name}"
                ),
                goal=(
                    objective
                    if as_objective and objective
                    else f"Execute action {action_name} requested by event {event.type}."
                ),
                purpose=(
                    MULTI_STEP_OBJECTIVE_PURPOSE
                    if as_objective
                    else (
                        "Runtime-created task because the generator started an action "
                        "without a resolvable task."
                    )
                ),
                continuation=continuation,
            )
            task_ref = command.get("task_ref")
            if task_ref:
                ref_map[str(task_ref)] = task.task_id
            state.workspace.note(
                "Recovered generator start_action without task by creating "
                f"{task.task_id} for {action_name}."
            )
            if self.trace:
                trace_line(
                    "runtime.policy",
                    "recovered start_action without task; "
                    f"created task=[magenta]{task.task_id}[/magenta] "
                    f"action=[bold]{action_name}[/bold]",
                )
            return task.task_id

    def _should_recover_action_as_objective_task(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        action_name: str,
    ) -> bool:
        try:
            spec = self.registry.get(action_name)
        except KeyError:
            spec = None
        if spec and spec.source == "star_protocol":
            return True
        if event.source == "star_protocol" or event.type.startswith("protocol."):
            return True
        objective = self._latest_user_objective(state).lower()
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

    def _latest_user_objective(self, state: AgentState) -> str:
        for item in reversed(state.workspace.transcript):
            if item.get("role") != "user":
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return ""

    def _objective_title(self, objective: str) -> str:
        normalized = " ".join(objective.split())
        if not normalized:
            return "Continue multi-step objective"
        return normalized

    def _resolve_task_id(self, state: AgentState, command: JsonDict, ref_map: Dict[str, str]) -> str:
        if command.get("task_id"):
            task_id = str(command["task_id"])
        elif command.get("task_ref"):
            task_ref = str(command["task_ref"])
            if task_ref not in ref_map:
                raise KeyError(f"Unknown task_ref: {task_ref}")
            task_id = ref_map[task_ref]
        elif state.workspace.current_task_id:
            task_id = state.workspace.current_task_id
        else:
            raise KeyError("Command requires task_id or task_ref, and no current task exists.")
        if task_id not in state.tasks:
            raise KeyError(f"Unknown task_id: {task_id}")
        return task_id

    async def _send_reply(self, state: AgentState, content: str, *, event: AgentEvent) -> None:
        state.workspace.add_transcript("assistant", content, event_id=event.event_id)
        if self.trace:
            trace_text("agent.reply", "assistant", content)
        if self.protocol_interface is None:
            return
        channel = event.payload.get("channel")
        if isinstance(channel, str) and channel and channel != "star_protocol":
            return
        recipient = self._reply_recipient_for_event(state, event)
        if recipient is None:
            return
        await self.protocol_interface.send_event(
            agent_id=state.agent_id,
            recipient=recipient,
            event_name="assistant.message",
            data={
                "content": content,
                "event_id": event.event_id,
                "task_id": event.task_id,
                "conversation_id": event.payload.get("conversation_id"),
                "turn_id": event.payload.get("turn_id"),
            },
            causation_id=event.event_id,
        )
        if self.trace:
            trace_line(
                "protocol.star",
                f"sent assistant.message recipient=[cyan]{recipient}[/cyan] "
                f"causation=[cyan]{event.event_id}[/cyan]",
            )

    def _reply_recipient_for_event(self, state: AgentState, event: AgentEvent) -> Optional[str]:
        explicit_recipient = event.payload.get("recipient_id")
        if isinstance(explicit_recipient, str) and explicit_recipient:
            if explicit_recipient != state.agent_id:
                state.workspace.variables["last_star_reply_recipient"] = explicit_recipient
                return explicit_recipient

        sender = event.payload.get("sender")
        if (
            event.source == "star_protocol"
            and event.type in {"user.message", "protocol.event", "protocol.action"}
            and isinstance(sender, str)
            and sender
        ):
            state.workspace.variables["last_star_reply_recipient"] = sender
            return sender

        if event.task_id and event.task_id in state.tasks:
            recipient = state.tasks[event.task_id].continuation.get("reply_recipient")
            if isinstance(recipient, str) and recipient:
                return recipient

        recipient = state.workspace.variables.get("last_star_reply_recipient")
        if isinstance(recipient, str) and recipient:
            return recipient
        return None

    def _trace_dmn_reflection(
        self,
        decision: GeneratorDecision,
        emitted_events: List[AgentEvent],
    ) -> None:
        if not self.trace:
            return
        payload = {
            "decision": decision,
            "emitted_events": [
                {
                    "event_id": event.event_id,
                    "type": event.type,
                    "source": event.source,
                    "payload": event.payload,
                }
                for event in emitted_events
            ],
        }
        trace_json("dmn", "reflection parsed", payload)

    def _trace_internal_decision_context(
        self,
        context: JsonDict,
        *,
        model_tools: Optional[List[JsonDict]] = None,
    ) -> None:
        if not self.trace:
            return
        trigger = context.get("decision", {}).get("trigger", {})
        focus = context.get("focus", {})
        workspace = context.get("workspace", {})
        compact = {
            "context_kind": context.get("context_kind"),
            "event": {
                "type": trigger.get("type"),
                "payload": trigger.get("payload", {}),
                "task_id": trigger.get("task_id"),
                "action_run_id": trigger.get("action_run_id"),
            },
            "workspace": {
                "current_task_id": workspace.get("current_task_id"),
                "last_decision_summary": workspace.get("last_decision_summary"),
                "stored_counts": workspace.get("stored_counts"),
            },
            "runtime": context.get("runtime", {}),
            "focus": focus,
            "evidence_types": [item.get("type") for item in context.get("evidence", [])],
            "tooling": context.get("tooling", {}),
            "model_tools": [
                tool.get("function", {}).get("name")
                for tool in (model_tools or [])
            ],
        }
        trace_json("agent.context", "decision context sent to generator", compact)

    def _trace_context_selection(self, context: JsonDict) -> None:
        if not self.trace:
            return
        manifest = context.get("_selection_manifest")
        if isinstance(manifest, dict):
            trace_json("agent.context", "context selection manifest", manifest)

    def _trace_context_usage(
        self,
        generator_trace: JsonDict,
        *,
        title: str = "model context usage",
    ) -> None:
        if not self.trace:
            return
        usage = generator_trace.get("context_usage")
        if isinstance(usage, dict):
            trace_json("agent.context", title, usage)

    def _trace_model_raw_request(
        self,
        generator_trace: JsonDict,
        *,
        title: str = "raw request",
    ) -> None:
        if not self.trace:
            return
        model_request = generator_trace.get("model_request")
        if not model_request:
            return
        trace_json("model.request", title, model_request)

    def _trace_model_raw_response(
        self,
        generator_trace: JsonDict,
        *,
        title: str = "raw response",
    ) -> None:
        if not self.trace:
            return
        model_response = generator_trace.get("model_response")
        if not model_response:
            return
        trace_json("model.response", title, model_response)

    def _trace_generator_decision(
        self,
        decision: GeneratorDecision,
        generator_trace: JsonDict,
    ) -> None:
        if not self.trace:
            return
        payload = {
            "parse": generator_trace.get("parse", {}),
            "decision": decision,
        }
        trace_json("agent.decision", "generator decision parsed", payload)
