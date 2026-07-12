from __future__ import annotations

import asyncio
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from agent.protocols import ActionRun, AgentEvent
from agent.runtime.interfaces.model import ModelInterface, ModelResult
from agent.runtime.persistence_system import JsonStateStore

from agent_ling_refactor.app import create_refactored_runtime
from agent_ling_refactor.context import ContextComposer
from agent_ling_refactor.messages import MessagePurpose, NaturalMessage
from agent_ling_refactor.model_gateway import ToolCall
from agent_ling_refactor.prompts import PromptCompiler
from agent_ling_refactor.settings import load_refactor_settings


class ScriptedModel(ModelInterface):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def chat(
        self,
        messages: object,
        *,
        model: str | None = None,
        **kwargs: object,
    ) -> ModelResult:
        items = list(messages)  # type: ignore[arg-type]
        system = str(getattr(items[0], "content", ""))
        self.calls.append({"system": system, "messages": items, "kwargs": kwargs})
        if "当前职责：理解" in system:
            return ModelResult(text="对方在礼貌问候，没有提出行动请求。")
        if "当前职责：把收到" in system:
            return ModelResult(text="你好，我在。")
        raise AssertionError(f"Unexpected region prompt: {system}")


class ConcurrentDecisionModel(ModelInterface):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.decision_started = asyncio.Event()
        self.reply_sent = asyncio.Event()
        self.decision_finished_after_reply = False

    async def chat(
        self,
        messages: object,
        *,
        model: str | None = None,
        **kwargs: object,
    ) -> ModelResult:
        items = list(messages)  # type: ignore[arg-type]
        system = str(getattr(items[0], "content", ""))
        if "当前职责：理解" in system:
            self.calls.append("understanding")
            return ModelResult(
                text="对方要求查看当前任务状态，这是一个需要行动的请求。",
                raw=SimpleNamespace(
                    tool_calls=[
                        {
                            "id": "signal-1",
                            "name": "request_decision",
                            "arguments": {
                                "reason": "需要读取任务状态",
                                "objective": "查看当前任务状态并反馈",
                            },
                        }
                    ]
                ),
            )
        if "当前职责：根据收到" in system:
            self.calls.append("decision")
            self.decision_started.set()
            await asyncio.wait_for(self.reply_sent.wait(), timeout=1)
            self.decision_finished_after_reply = self.reply_sent.is_set()
            return ModelResult(
                text="读取当前任务状态。",
                raw=SimpleNamespace(
                    tool_calls=[
                        {
                            "id": "tool-1",
                            "name": "query_task_status",
                            "arguments": {},
                        }
                    ]
                ),
            )
        if "当前职责：把收到" in system:
            self.calls.append("expression")
            await asyncio.wait_for(self.decision_started.wait(), timeout=1)
            return ModelResult(text="我先看一下当前状态，很快告诉你。")
        raise AssertionError(f"Unexpected region prompt: {system}")


class EndToEndActionModel(ModelInterface):
    def __init__(self) -> None:
        self.decision_calls = 0

    async def chat(
        self,
        messages: object,
        *,
        model: str | None = None,
        **kwargs: object,
    ) -> ModelResult:
        items = list(messages)  # type: ignore[arg-type]
        system = str(getattr(items[0], "content", ""))
        user = str(getattr(items[-1], "content", ""))
        if "当前职责：理解" in system:
            return ModelResult(
                text="对方要求读取当前任务状态，需要决策区域调用能力。",
                raw=SimpleNamespace(
                    tool_calls=[
                        {
                            "id": "signal-e2e",
                            "name": "request_decision",
                            "arguments": {"reason": "需要读取状态"},
                        }
                    ]
                ),
            )
        if "当前职责：根据收到" in system:
            self.decision_calls += 1
            if self.decision_calls == 1:
                return ModelResult(
                    text="读取内部任务状态。",
                    raw=SimpleNamespace(
                        tool_calls=[
                            {
                                "id": "tool-e2e",
                                "name": "query_task_status",
                                "arguments": {},
                            }
                        ]
                    ),
                )
            self.assert_action_result_in_context(user)
            return ModelResult(text="状态已经读取完成，可以把结果告诉对方。")
        if "当前职责：把收到" in system:
            if "状态已经读取完成" in user:
                return ModelResult(text="我看过了，当前状态已经读取完成。")
            return ModelResult(text="我先看一下当前状态。")
        raise AssertionError(f"Unexpected region prompt: {system}")

    def assert_action_result_in_context(self, context: str) -> None:
        if "query_task_status 已成功完成" not in context:
            raise AssertionError(f"Action result missing from decision context: {context}")


class RateLimitedModel(ModelInterface):
    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self,
        messages: object,
        *,
        model: str | None = None,
        **kwargs: object,
    ) -> ModelResult:
        self.calls += 1
        raise RuntimeError("429 Too many requests")


class LoopStoppingModel(ModelInterface):
    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self,
        messages: object,
        *,
        model: str | None = None,
        **kwargs: object,
    ) -> ModelResult:
        self.calls += 1
        if self.calls == 1:
            tool = {"id": "read", "name": "query_task_status", "arguments": {}}
        elif self.calls == 2:
            tool = {
                "id": "wait",
                "name": "runtime_update_task",
                "arguments": {
                    "patch": {
                        "status": "waiting",
                        "progress": {"message": "等待新的外部输入"},
                    }
                },
            }
        else:
            raise AssertionError("internal completion incorrectly reactivated the model")
        return ModelResult(
            text="",
            raw=SimpleNamespace(tool_calls=[tool]),
        )


class NeverCalledModel(ModelInterface):
    async def chat(
        self,
        messages: object,
        *,
        model: str | None = None,
        **kwargs: object,
    ) -> ModelResult:
        raise AssertionError("state-only recovery must not call the model")


class RefactoredConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordinary_reply_is_understanding_then_expression(self) -> None:
        with TemporaryDirectory() as directory:
            model = ScriptedModel()
            application = create_refactored_runtime(
                agent_id="ling-refactor-test",
                store=JsonStateStore(directory),
                model=model,
                trace=False,
            )
            event = AgentEvent.make(
                agent_id=application.agent_id,
                type="user.message",
                source="test",
                payload={
                    "content": "你好",
                    "sender": "human-1",
                    "conversation_id": "conversation-1",
                },
            )

            await application.runtime.handle_event(event)

            self.assertEqual(await application.runtime.next_reply(timeout=0.1), "你好，我在。")
            self.assertEqual(len(model.calls), 2)
            self.assertIn("当前职责：理解", str(model.calls[0]["system"]))
            self.assertIn("当前职责：把收到", str(model.calls[1]["system"]))
            state = application.runtime.store.load_state(application.agent_id)
            self.assertIn(event.event_id, state.processed_event_ids)
            pending_memory = state.workspace.variables["memory_system"][
                "pending_reflection_ids"
            ]
            self.assertEqual(len(pending_memory), 2)
            self.assertEqual(
                state.workspace.variables["emotion_state"]["last_trigger"]["event_type"],
                "conversation.understanding.ready",
            )
            conversation = application.runtime.conversation.store.load_state(application.agent_id)
            turn = next(iter(conversation.turns.values()))
            self.assertEqual(
                turn.understanding["semantic_summary"],
                "对方在礼貌问候，没有提出行动请求。",
            )
            self.assertEqual(turn.response_text, "你好，我在。")

    async def test_decision_runs_concurrently_without_blocking_expression(self) -> None:
        with TemporaryDirectory() as directory:
            model = ConcurrentDecisionModel()

            async def on_reply(_: str) -> None:
                model.reply_sent.set()

            application = create_refactored_runtime(
                agent_id="ling-parallel-test",
                store=JsonStateStore(directory),
                model=model,
                on_reply=on_reply,
                trace=False,
            )
            event = AgentEvent.make(
                agent_id=application.agent_id,
                type="user.message",
                source="test",
                payload={
                    "content": "帮我看一下当前任务状态",
                    "sender": "human-1",
                    "conversation_id": "conversation-1",
                },
            )

            await application.runtime.handle_event(event)

            self.assertEqual(
                await application.runtime.next_reply(timeout=0.1),
                "我先看一下当前状态，很快告诉你。",
            )
            self.assertEqual(model.calls[0], "understanding")
            self.assertCountEqual(model.calls[1:], ["decision", "expression"])
            self.assertTrue(model.decision_finished_after_reply)
            state = application.runtime.store.load_state(application.agent_id)
            self.assertEqual(len(state.tasks), 1)
            self.assertEqual(len(state.action_runs), 1)
            run = next(iter(state.action_runs.values()))
            self.assertEqual(run.action_name, "query_task_status")

    async def test_sync_action_runs_through_to_broca_final_reply(self) -> None:
        with TemporaryDirectory() as directory:
            model = EndToEndActionModel()
            application = create_refactored_runtime(
                agent_id="ling-e2e-test",
                store=JsonStateStore(directory),
                model=model,
                trace=False,
            )
            await application.runtime.start()
            try:
                await application.runtime.submit_user_message(
                    "查看当前任务状态",
                    sender="human-1",
                    conversation_id="conversation-1",
                )
                first = await application.runtime.next_reply(timeout=2)
                second = await application.runtime.next_reply(timeout=2)
                await application.runtime.event_bus.join()
            finally:
                await application.runtime.stop()

            self.assertEqual(first, "我先看一下当前状态。")
            self.assertEqual(second, "我看过了，当前状态已经读取完成。")
            state = application.runtime.store.load_state(application.agent_id)
            task = next(iter(state.tasks.values()))
            self.assertEqual(task.status, "completed")
            run = next(iter(state.action_runs.values()))
            self.assertEqual(run.status, "succeeded")

    async def test_rate_limit_stops_followup_model_requests(self) -> None:
        with TemporaryDirectory() as directory:
            model = RateLimitedModel()
            application = create_refactored_runtime(
                agent_id="ling-backoff-test",
                store=JsonStateStore(directory),
                model=model,
                trace=False,
            )
            first_event = AgentEvent.make(
                agent_id=application.agent_id,
                type="user.message",
                source="test",
                payload={"content": "你好", "sender": "human-1"},
            )
            second_event = AgentEvent.make(
                agent_id=application.agent_id,
                type="user.message",
                source="test",
                payload={"content": "还在吗", "sender": "human-1"},
            )

            await application.runtime.handle_event(first_event)
            first_reply = await application.runtime.next_reply(timeout=0.1)
            await application.runtime.handle_event(second_event)
            second_reply = await application.runtime.next_reply(timeout=0.1)

            self.assertEqual(model.calls, 1)
            self.assertIn("暂停后续模型唤醒", first_reply)
            self.assertIn("正在退避", second_reply)
            self.assertNotIn("429", first_reply)
            state = application.runtime.store.load_state(application.agent_id)
            activation = state.workspace.variables["model_activation"]
            self.assertIn("backoff_until", activation)

    async def test_startup_objective_enters_decision_without_fake_recipient_reply(self) -> None:
        with TemporaryDirectory() as directory:
            model = EndToEndActionModel()
            application = create_refactored_runtime(
                agent_id="ling-objective-test",
                store=JsonStateStore(directory),
                model=model,
                trace=False,
            )
            event = AgentEvent.make(
                agent_id=application.agent_id,
                type="runtime.objective",
                source="runtime",
                payload={"content": "观察环境并推进可执行目标"},
            )

            await application.runtime.handle_event(event)

            self.assertEqual(model.decision_calls, 1)
            with self.assertRaises(asyncio.TimeoutError):
                await application.runtime.next_reply(timeout=0.01)
            state = application.runtime.store.load_state(application.agent_id)
            self.assertEqual(len(state.tasks), 1)

    async def test_internal_completion_terminates_decision_feedback_loop(self) -> None:
        with TemporaryDirectory() as directory:
            model = LoopStoppingModel()
            application = create_refactored_runtime(
                agent_id="ling-loop-test",
                store=JsonStateStore(directory),
                model=model,
                trace=False,
            )
            await application.runtime.start()
            try:
                await application.runtime.submit_objective("读取状态后等待外部输入")
                await asyncio.wait_for(application.runtime.event_bus.join(), timeout=2)
            finally:
                await application.runtime.stop()

            self.assertEqual(model.calls, 2)
            state = application.runtime.store.load_state(application.agent_id)
            task = next(iter(state.tasks.values()))
            self.assertEqual(task.status, "waiting")
            checkpoints = (
                application.runtime.store.root
                / "checkpoints"
                / f"{application.agent_id}.jsonl"
            ).read_text()
            self.assertIn('"model_activation": "suppressed"', checkpoints)

    async def test_restart_recovers_orphan_action_without_model_storm(self) -> None:
        with TemporaryDirectory() as directory:
            store = JsonStateStore(directory)
            state = store.load_state("ling-recovery-test")
            from agent.runtime.action_systems.task_system import (
                MULTI_STEP_OBJECTIVE_PURPOSE,
            )
            from agent_ling_refactor.scheduling import RefactoredTaskSystem

            tasks = RefactoredTaskSystem()
            task = tasks.create_task(
                state,
                title="waiting",
                goal="wait safely",
                purpose=MULTI_STEP_OBJECTIVE_PURPOSE,
            )
            tasks.add_wait(state, task.task_id, {"awaiting": "human_response"})
            run = ActionRun(
                action_run_id="run-orphan",
                agent_id=state.agent_id,
                task_id=task.task_id,
                action_name="observe_scene",
                args={},
                mode="async",
                source="star_protocol",
                status="created",
            )
            state.action_runs[run.action_run_id] = run
            task.active_action_runs.append(run.action_run_id)
            store.save_state(state)
            application = create_refactored_runtime(
                agent_id=state.agent_id,
                store=store,
                model=NeverCalledModel(),
                trace=False,
            )

            await application.runtime.start()
            try:
                await asyncio.wait_for(application.runtime.event_bus.join(), timeout=1)
            finally:
                await application.runtime.stop()

            restored = store.load_state(state.agent_id)
            self.assertEqual(restored.action_runs[run.action_run_id].status, "failed")
            self.assertEqual(restored.tasks[task.task_id].status, "waiting")


class RefactoredPromptTests(unittest.TestCase):
    def test_prompts_are_small_and_do_not_embed_runtime_protocols(self) -> None:
        settings = load_refactor_settings()
        compiler = PromptCompiler(settings.prompts)
        forbidden = (
            "GeneratorDecision",
            "start_action JSON",
            "task_ref",
            "completion_blockers",
            "context_selection",
        )
        for purpose in MessagePurpose:
            message = NaturalMessage(
                sender="test",
                recipient="test",
                purpose=purpose,
                text="测试交接",
            )
            prompt = compiler.system_prompt(
                profile=settings.profile,
                message=message,
                tools_available=True,
            )
            self.assertLess(len(prompt), 800)
            for marker in forbidden:
                self.assertNotIn(marker, prompt)

    def test_expression_context_excludes_task_and_action_protocol_details(self) -> None:
        settings = load_refactor_settings()
        from agent.protocols import AgentState
        from agent.runtime.persistence_system import MemoryStore
        from agent.runtime.state_systems import MemorySystem

        state = AgentState.new("context-test")
        message = NaturalMessage(
            sender="understanding",
            recipient="expression",
            purpose=MessagePurpose.EXPRESSION,
            text="对方在问候，请自然回应。",
        )
        event = AgentEvent.make(
            agent_id=state.agent_id,
            type="user.message",
            source="test",
            payload={"content": "你好"},
        )
        with TemporaryDirectory() as directory:
            context = ContextComposer(settings.runtime).compose(
                state=state,
                event=event,
                message=message,
                memory_system=MemorySystem(settings.memory, store=MemoryStore(directory)),
            )
        self.assertIn("表达交接", context)
        self.assertNotIn("最近行动", context)
        self.assertNotIn("当前任务", context)
        self.assertNotIn("相关记忆", context)

    def test_generator_logs_preserve_natural_language_handoffs(self) -> None:
        with TemporaryDirectory() as directory:
            # This fixture documents the durable shape without coupling the test
            # to private runtime methods.
            message = NaturalMessage(
                sender="understanding",
                recipient="expression",
                purpose=MessagePurpose.EXPRESSION,
                text="对方只是问候，请自然回应。",
            )
            record = message.audit_record()
            serialized = json.dumps(record, ensure_ascii=False)
            self.assertEqual(record["text"], "对方只是问候，请自然回应。")
            self.assertNotIn("commands", serialized)
            self.assertNotIn("decision_summary", serialized)

    def test_wait_tool_is_ordered_after_external_work(self) -> None:
        with TemporaryDirectory() as directory:
            application = create_refactored_runtime(
                agent_id="ordering-test",
                store=JsonStateStore(directory),
                model=ScriptedModel(),
                trace=False,
            )
            ordered = application.runtime._ordered_tool_calls(  # noqa: SLF001
                (
                    ToolCall(
                        call_id="wait",
                        name="runtime_wait",
                        arguments={"condition": {"awaiting": "human_response"}},
                    ),
                    ToolCall(
                        call_id="work",
                        name="query_task_status",
                        arguments={},
                    ),
                )
            )
        self.assertEqual([call.name for call in ordered], ["query_task_status", "runtime_wait"])


if __name__ == "__main__":
    unittest.main()
