from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from agent.protocols import ActionRun, AgentEvent, AgentState, AgentTask
from agent.runtime.state_systems import MemorySystem

from .messages import MessagePurpose, NaturalMessage
from .settings import RuntimeSettings


class ContextComposer:
    """Compose a bounded natural-language handoff for the model.

    Durable state remains structured at rest.  At the cognitive boundary it is
    rendered as ordinary language, so subsystems do not need to learn private
    JSON dialects or duplicate schema instructions in prompts.
    """

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings

    def compose(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        message: NaturalMessage,
        memory_system: MemorySystem,
        conversation_history: Sequence[Mapping[str, Any]] = (),
    ) -> str:
        sections = self._sections_for(
            state=state,
            event=event,
            message=message,
            memory_system=memory_system,
            conversation_history=conversation_history,
        )
        rendered = "\n\n".join(
            f"## {title}\n{content}" for title, content in sections if content
        )
        if len(rendered) <= self.settings.max_context_chars:
            return rendered
        # Keep the immediate handoff and newest evidence when trimming.  This
        # is deterministic and never relies on provider-side silent truncation.
        head = f"## 当前交接\n{message.text}\n\n## 当前事件\n{event_to_natural_text(event)}"
        remaining = max(0, self.settings.max_context_chars - len(head) - 24)
        return f"{head}\n\n## 其余相关背景\n{rendered[-remaining:]}" if remaining else head

    def _sections_for(
        self,
        *,
        state: AgentState,
        event: AgentEvent,
        message: NaturalMessage,
        memory_system: MemorySystem,
        conversation_history: Sequence[Mapping[str, Any]],
    ) -> list[tuple[str, str]]:
        conversation = self._conversation(state, conversation_history)
        current_state = self._state_notes(state)
        if message.purpose == MessagePurpose.EXPRESSION:
            return [
                ("表达交接", message.text),
                ("最近交流", conversation),
                ("此刻状态", current_state),
            ]
        if message.purpose == MessagePurpose.UNDERSTANDING:
            return [
                ("理解交接", message.text),
                ("最近交流", conversation),
                ("相关记忆", self._memory(state, message.text, memory_system)),
                ("此刻状态", current_state),
            ]
        if message.purpose == MessagePurpose.DECISION:
            return [
                ("决策交接", message.text),
                ("当前事件", event_to_natural_text(event)),
                ("当前任务", self._focus_task(state, message.task_id or event.task_id)),
                ("最近行动", self._action_history(state)),
                ("相关记忆", self._memory(state, message.text, memory_system)),
                ("最近交流", conversation),
                ("此刻状态", current_state),
            ]
        return [
            ("当前交接", message.text),
            ("近期工作记录", "\n".join(state.workspace.notes[-8:])),
            ("此刻状态", current_state),
        ]

    def _focus_task(self, state: AgentState, preferred_id: str | None) -> str:
        task = state.tasks.get(preferred_id or "")
        if task is None:
            task = state.tasks.get(state.workspace.current_task_id or "")
        if task is None:
            return "目前没有未完成的焦点任务。"
        return task_to_natural_text(task)

    def _conversation(
        self,
        state: AgentState,
        turns: Sequence[Mapping[str, Any]],
    ) -> str:
        lines: list[str] = []
        for turn in turns[-self.settings.recent_transcript_items :]:
            speaker = str(turn.get("speaker_id") or "对方")
            utterance = turn.get("utterance") or turn.get("utterance_summary")
            response = turn.get("response_text") or turn.get("agent_response_intent")
            if utterance:
                lines.append(f"- {speaker}：{utterance}")
            if response:
                lines.append(f"- {state.profile.name}：{response}")
        if not lines:
            for item in state.workspace.transcript[-self.settings.recent_transcript_items :]:
                role = "对方" if item.get("role") == "user" else state.profile.name
                content = str(item.get("content") or "").strip()
                if content:
                    lines.append(f"- {role}：{content}")
        return "\n".join(lines)

    def _action_history(self, state: AgentState) -> str:
        runs = list(state.action_runs.values())[-self.settings.recent_action_runs :]
        if not runs:
            return ""
        return "\n".join(f"- {action_run_to_natural_text(run)}" for run in runs)

    def _memory(
        self,
        state: AgentState,
        query: str,
        memory_system: MemorySystem,
    ) -> str:
        if not memory_system.enabled:
            return ""
        records = memory_system.search(
            agent_id=state.agent_id,
            query=query,
            limit=self.settings.memory_retrieval_limit,
        )
        lines: list[str] = []
        for record in records:
            title = str(record.get("title") or "未命名记忆")
            content = str(record.get("content") or record.get("summary") or "").strip()
            confidence = record.get("confidence")
            suffix = f"，置信度 {confidence}" if confidence is not None else ""
            lines.append(f"- {title}{suffix}：{content[:600]}")
        return "\n".join(lines)

    def _state_notes(self, state: AgentState) -> str:
        lines: list[str] = []
        emotion = state.workspace.variables.get("emotion_state")
        if isinstance(emotion, dict):
            mood = emotion.get("mood") or emotion.get("primary")
            intensity = emotion.get("intensity")
            if mood:
                lines.append(f"当前情绪倾向为 {mood}，强度约为 {intensity}。")
        recent_notes = state.workspace.notes[-3:]
        if recent_notes:
            lines.append("近期工作记录：" + "；".join(str(note) for note in recent_notes))
        return "\n".join(lines)


def event_to_natural_text(event: AgentEvent) -> str:
    content = str(event.payload.get("content") or "").strip()
    action_name = str(event.payload.get("action_name") or "该能力")
    if event.type == "user.message":
        return f"对方说：{content}"
    if event.type == "action.completed":
        return f"{action_name} 已成功完成。结果是：{natural_value(event.payload.get('result'))}"
    if event.type == "action.failed":
        return f"{action_name} 执行失败。原因是：{natural_value(event.payload.get('error'))}"
    if event.type == "action.cancelled":
        return f"{action_name} 已取消。相关信息：{natural_value(event.payload)}"
    if event.type == "action.progress":
        return f"{action_name} 有新进展：{natural_value(event.payload.get('progress'))}"
    if event.type == "agent.thought":
        return f"空闲回顾产生了一个待判断的想法：{content}"
    if event.type == "conversation.proactive":
        return content or "对方允许继续当前话题，请自然补充一个有价值的相关想法。"
    if event.type == "runtime.continue":
        return content or "运行时发现还有可以继续推进的工作。"
    if event.type == "runtime.objective":
        return f"当前自主目标是：{content}"
    if content:
        return f"收到 {event.type} 事件：{content}"
    return f"收到 {event.type} 事件。详情是：{natural_value(event.payload)}"


def task_to_natural_text(task: AgentTask) -> str:
    parts = [
        f"任务“{task.title}”当前为 {task.status}",
        f"目标是：{task.goal}",
    ]
    if task.progress:
        parts.append(f"进展：{natural_value(task.progress)}")
    if task.waiting_on:
        parts.append(f"正在等待：{natural_value(task.waiting_on)}")
    if task.error:
        parts.append(f"错误：{natural_value(task.error)}")
    return "。".join(parts) + "。"


def action_run_to_natural_text(run: ActionRun) -> str:
    text = f"能力 {run.action_name} 的状态是 {run.status}"
    if run.result:
        text += f"，结果为 {natural_value(run.result)}"
    if run.error:
        text += f"，错误为 {natural_value(run.error)}"
    return text + "。"


def natural_value(value: Any, *, depth: int = 0) -> str:
    if value is None:
        return "无"
    if depth >= 3:
        return str(value)[:300]
    if isinstance(value, Mapping):
        if not value:
            return "无"
        return "；".join(
            f"{key} 是 {natural_value(item, depth=depth + 1)}"
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return "无"
        return "、".join(natural_value(item, depth=depth + 1) for item in value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return "、".join(natural_value(item, depth=depth + 1) for item in value)
    return str(value)
