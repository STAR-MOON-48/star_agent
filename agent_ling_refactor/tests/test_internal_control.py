from __future__ import annotations

import asyncio
from tempfile import TemporaryDirectory
import unittest

from agent.runtime.interfaces.model import ModelInterface, ModelResult
from agent.runtime.persistence_system import JsonStateStore

from agent_ling_refactor.app import create_refactored_runtime
from agent_ling_refactor.control import ControlInbox


class InternalControlModel(ModelInterface):
    def __init__(self) -> None:
        self.regions: list[str] = []

    async def chat(
        self,
        messages: object,
        *,
        model: str | None = None,
        **kwargs: object,
    ) -> ModelResult:
        items = list(messages)  # type: ignore[arg-type]
        system = str(getattr(items[0], "content", ""))
        if "当前职责：根据收到" in system:
            self.regions.append("decision")
            return ModelResult(text="先重新检查失败是否来自前置条件，而不是重复原动作。")
        if "当前职责：空闲时回顾" in system:
            self.regions.append("dmn")
            return ModelResult(text="这次失败可能暴露了一个被忽略的前置条件，值得重新判断。")
        raise AssertionError(f"Unexpected internal control region: {system}")


async def wait_until_processed(
    inbox: ControlInbox,
    directive_id: str,
    *,
    timeout: float = 2,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if inbox.is_processed(directive_id):
            return
        await asyncio.sleep(0.02)
    raise TimeoutError(f"Directive was not processed: {directive_id}")


class ControlInboxTests(unittest.TestCase):
    def test_directive_is_atomically_claimed_and_acknowledged(self) -> None:
        with TemporaryDirectory() as directory:
            inbox = ControlInbox(directory, "agent-1")
            directive = inbox.enqueue(
                target="decision",
                text="  重新检查前置条件，不要重复动作。  ",
            )
            items = inbox.claim()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].directive.text, "重新检查前置条件，不要重复动作。")
            event = items[0].directive.to_event()
            self.assertEqual(event.type, "operator.directive")
            self.assertEqual(event.payload["content"], directive.text)
            inbox.acknowledge(items[0])
            self.assertTrue(inbox.is_processed(directive.directive_id))


class InternalControlRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_decision_directive_is_internal_and_has_no_broca_reply(self) -> None:
        with TemporaryDirectory() as directory:
            model = InternalControlModel()
            application = create_refactored_runtime(
                agent_id="control-decision-test",
                store=JsonStateStore(directory),
                model=model,
                trace=False,
            )
            await application.runtime.start()
            try:
                directive = application.runtime.control_inbox.enqueue(
                    target="decision",
                    text="把注意力转向失败动作的前置条件，不要立即重试。",
                )
                await wait_until_processed(
                    application.runtime.control_inbox,
                    directive.directive_id,
                )
            finally:
                await application.runtime.stop()

            self.assertEqual(model.regions, ["decision"])
            with self.assertRaises(asyncio.TimeoutError):
                await application.runtime.next_reply(timeout=0.01)
            state = application.runtime.store.load_state(application.agent_id)
            self.assertTrue(
                any("内部决策结果" in note for note in state.workspace.notes)
            )

    async def test_note_target_updates_workspace_without_model(self) -> None:
        with TemporaryDirectory() as directory:
            model = InternalControlModel()
            application = create_refactored_runtime(
                agent_id="control-note-test",
                store=JsonStateStore(directory),
                model=model,
                trace=False,
            )
            await application.runtime.start()
            try:
                directive = application.runtime.control_inbox.enqueue(
                    target="note",
                    text="后续判断优先保留证据来源。",
                )
                await wait_until_processed(
                    application.runtime.control_inbox,
                    directive.directive_id,
                )
            finally:
                await application.runtime.stop()

            self.assertEqual(model.regions, [])
            state = application.runtime.store.load_state(application.agent_id)
            self.assertTrue(
                any("后续判断优先保留证据来源" in note for note in state.workspace.notes)
            )

    async def test_dmn_directive_reflects_then_may_reach_decision(self) -> None:
        with TemporaryDirectory() as directory:
            model = InternalControlModel()
            application = create_refactored_runtime(
                agent_id="control-dmn-test",
                store=JsonStateStore(directory),
                model=model,
                trace=False,
            )
            await application.runtime.start()
            try:
                directive = application.runtime.control_inbox.enqueue(
                    target="dmn",
                    text="想一想最近失败里是否存在共同的前置条件。",
                )
                await wait_until_processed(
                    application.runtime.control_inbox,
                    directive.directive_id,
                )
                await asyncio.wait_for(application.runtime.event_bus.join(), timeout=2)
            finally:
                await application.runtime.stop()

            self.assertEqual(model.regions, ["dmn", "decision"])
            state = application.runtime.store.load_state(application.agent_id)
            self.assertTrue(any("空闲回顾" in note for note in state.workspace.notes))


if __name__ == "__main__":
    unittest.main()
