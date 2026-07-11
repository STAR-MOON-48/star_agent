from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from menglong import Assistant, Context, System, Tool, User

from ...config import AgentConfig, load_agent_config
from ...config.loader import SafeFormatDict
from ...protocols import GeneratorDecision, JsonDict
from .generator_session import (
    GeneratorRequest,
    GeneratorResult,
    GeneratorSession,
    LLMGeneratorSession,
    model_request_trace,
    model_response_trace,
)
from ..interfaces.model import ModelInterface, ModelResult
from ..interfaces.star_model import DEFAULT_MODEL_ID, StarModel
from ..state_systems.context_policy import model_request_usage


class GeneratorRuntime:
    """Kernel-level manager for generator sessions and model requests."""

    def __init__(
        self,
        *,
        session: Optional[GeneratorSession] = None,
        model_interface: Optional[ModelInterface] = None,
        default_model_id: Optional[str] = None,
        config_path: Optional[str] = None,
        agent_config: Optional[AgentConfig] = None,
        agent_config_path: Optional[str] = None,
        trace: bool = True,
    ) -> None:
        self.trace = trace
        self.agent_config = (
            agent_config
            or (load_agent_config(agent_config_path) if agent_config_path else AgentConfig.empty())
        )
        self.default_session_id = self.agent_config.generator.default_session
        self.model_interface = model_interface
        self.sessions: dict[str, GeneratorSession] = {}
        self._started = False
        if session is None:
            self.model_interface = self.model_interface or StarModel(
                default_model_id=default_model_id or DEFAULT_MODEL_ID,
                config_path=config_path,
            )
        else:
            self.sessions[self.default_session_id] = session

    async def start(self) -> None:
        self._started = True
        session = await self._session_for(self.default_session_id)
        await session.start()

    async def stop(self) -> None:
        self._started = False
        for session in list(self.sessions.values()):
            await session.stop()

    async def generate(self, runtime_context: JsonDict) -> GeneratorDecision:
        return (await self.generate_with_trace(runtime_context)).decision

    async def generate_with_trace(self, runtime_context: JsonDict) -> GeneratorResult:
        session_id = self.session_id_for(runtime_context)
        model_tools = self.model_tools(runtime_context)
        prompt = self.agent_config.generator.prompt_for(session_id)
        (
            menglong_context,
            effective_runtime_context,
            context_usage,
            compaction,
        ) = self._prepare_model_context(
            runtime_context,
            session_id=session_id,
            tools=model_tools,
        )
        if not context_usage["within_budget"]:
            raise ValueError(
                "Model request exceeds configured context budget: "
                f"estimated_input_tokens={context_usage['estimated_input_tokens']} "
                f"available_input_tokens={context_usage['available_input_tokens']}"
            )
        model_kwargs: JsonDict = {
            "max_tokens": prompt.context_policy.reserve_output_tokens,
        }
        if model_tools:
            model_kwargs["tools"] = model_tools
        session = await self._session_for(session_id)
        result = await session.generate_with_trace(
            context=menglong_context,
            runtime_context=effective_runtime_context,
            model_kwargs=model_kwargs,
        )
        result.trace["generator_session"] = {
            "session_id": session_id,
            "system_name": self.agent_config.generator.prompt_for(session_id).system_name,
        }
        model_response = result.trace.get("model_response")
        if isinstance(model_response, dict) and isinstance(model_response.get("usage"), dict):
            context_usage["model_reported_usage"] = model_response["usage"]
        result.trace["context_usage"] = context_usage
        if compaction is not None:
            result.trace["context_compaction"] = compaction
        selection_manifest = effective_runtime_context.get("_selection_manifest")
        if isinstance(selection_manifest, dict):
            result.trace["context_selection"] = selection_manifest
        return result

    async def generate_text_with_trace(
        self,
        runtime_context: JsonDict,
        *,
        session_id: str,
        tools: Optional[list[JsonDict]] = None,
    ) -> tuple[ModelResult, JsonDict]:
        prompt = self.agent_config.generator.prompt_for(session_id)
        model_tools = list(tools or [])
        (
            menglong_context,
            effective_runtime_context,
            context_usage,
            compaction,
        ) = self._prepare_model_context(
            runtime_context,
            session_id=session_id,
            tools=model_tools,
        )
        if not context_usage["within_budget"]:
            raise ValueError(
                "Text generator request exceeds configured context budget: "
                f"estimated_input_tokens={context_usage['estimated_input_tokens']} "
                f"available_input_tokens={context_usage['available_input_tokens']}"
            )
        model_kwargs: JsonDict = {
            "max_tokens": prompt.context_policy.reserve_output_tokens,
        }
        if model_tools:
            model_kwargs["tools"] = model_tools
        request = GeneratorRequest(
            request_id=f"gt_{uuid4().hex[:12]}",
            context=menglong_context,
            runtime_context=effective_runtime_context,
            model_kwargs=model_kwargs,
        )
        session = await self._session_for(session_id)
        result = await session.chat(
            context=menglong_context,
            **model_kwargs,
        )
        trace: JsonDict = {
            "request_id": request.request_id,
            "model_request": model_request_trace(request),
            "model_response": model_response_trace(result),
            "generator_session": {
                "session_id": session_id,
                "system_name": prompt.system_name,
            },
            "context_usage": context_usage,
        }
        if compaction is not None:
            trace["context_compaction"] = compaction
        if result.usage:
            context_usage["model_reported_usage"] = result.usage
        return result, trace

    async def chat(self, context: Any, **kwargs: Any) -> ModelResult:
        session = await self._session_for(self.default_session_id)
        return await session.chat(context=context, **kwargs)

    def build_context(self, runtime_context: JsonDict, *, session_id: str | None = None) -> Context:
        context = Context()
        public_context = self.public_context(runtime_context)
        prompt = self.agent_config.generator.prompt_for(session_id)
        context.add(System(self._build_system_prompt(public_context, session_id=prompt.session_id)))
        if prompt.session_id == "decision":
            self._add_decision_messages(context, public_context)
        elif prompt.session_id == "wernicke":
            self._add_wernicke_messages(context, public_context)
        elif prompt.session_id == "broca":
            self._add_broca_messages(context, public_context)
        else:
            context.add(User(self._build_user_payload(public_context, session_id=prompt.session_id)))
        return context

    def _prepare_model_context(
        self,
        runtime_context: JsonDict,
        *,
        session_id: str,
        tools: list[JsonDict],
    ) -> tuple[Context, JsonDict, JsonDict, JsonDict | None]:
        prompt = self.agent_config.generator.prompt_for(session_id)
        menglong_context = self.build_context(runtime_context, session_id=session_id)
        usage = model_request_usage(
            messages=getattr(menglong_context, "messages", menglong_context),
            tools=tools,
            policy=prompt.context_policy,
        )
        if not usage["compaction_recommended"]:
            return menglong_context, runtime_context, usage, None

        compacted_context, compaction = self._compact_conversation_history(
            runtime_context,
            session_id=session_id,
            tools=tools,
            before_usage=usage,
        )
        if compacted_context is runtime_context:
            return menglong_context, runtime_context, usage, compaction

        menglong_context = self.build_context(compacted_context, session_id=session_id)
        usage = model_request_usage(
            messages=getattr(menglong_context, "messages", menglong_context),
            tools=tools,
            policy=prompt.context_policy,
        )
        compaction["after_estimated_input_tokens"] = usage["estimated_input_tokens"]
        compaction["within_budget_after_compaction"] = usage["within_budget"]
        return menglong_context, compacted_context, usage, compaction

    def _compact_conversation_history(
        self,
        runtime_context: JsonDict,
        *,
        session_id: str,
        tools: list[JsonDict],
        before_usage: JsonDict,
    ) -> tuple[JsonDict, JsonDict]:
        prompt = self.agent_config.generator.prompt_for(session_id)
        conversation = runtime_context.get("conversation")
        turns = conversation.get("recent_turns") if isinstance(conversation, dict) else None
        compaction: JsonDict = {
            "reason": "model_request_near_context_limit",
            "source_of_truth": "conversation_store",
            "trigger_tokens": prompt.context_policy.compaction_trigger_tokens,
            "target_tokens": prompt.context_policy.compaction_target_tokens,
            "available_input_tokens": prompt.context_policy.available_input_tokens,
            "before_estimated_input_tokens": before_usage["estimated_input_tokens"],
            "applied": False,
        }
        if not isinstance(conversation, dict) or not isinstance(turns, list) or not turns:
            compaction["reason"] = "no_optional_conversation_history_to_compact"
            return runtime_context, compaction

        selected: list[Any] = []
        released_refs: list[JsonDict] = []
        target_tokens = prompt.context_policy.compaction_target_tokens
        available_tokens = prompt.context_policy.available_input_tokens
        for turn in reversed(turns):
            trial_turns = [turn, *selected]
            trial_runtime_context = self._with_recent_turns(
                runtime_context,
                conversation,
                trial_turns,
            )
            trial_model_context = self.build_context(
                trial_runtime_context,
                session_id=session_id,
            )
            trial_usage = model_request_usage(
                messages=getattr(trial_model_context, "messages", trial_model_context),
                tools=tools,
                policy=prompt.context_policy,
            )
            keep_newest_if_possible = not selected and (
                trial_usage["estimated_input_tokens"] <= available_tokens
            )
            if (
                trial_usage["estimated_input_tokens"] <= target_tokens
                or keep_newest_if_possible
            ):
                selected = trial_turns
            else:
                released_refs.append(self._conversation_turn_ref(turn))

        compacted_context = self._with_recent_turns(
            runtime_context,
            conversation,
            selected,
        )
        compaction.update(
            {
                "applied": len(selected) < len(turns),
                "stored_turn_count": len(turns),
                "retained_turn_count": len(selected),
                "retained_turn_refs": [
                    self._conversation_turn_ref(turn) for turn in selected
                ],
                "released_turn_refs": list(reversed(released_refs)),
            }
        )
        if not compaction["applied"]:
            return runtime_context, compaction
        return compacted_context, compaction

    def _with_recent_turns(
        self,
        runtime_context: JsonDict,
        conversation: JsonDict,
        turns: list[Any],
    ) -> JsonDict:
        compacted_context = dict(runtime_context)
        compacted_context["conversation"] = {
            **conversation,
            "recent_turns": turns,
        }
        return compacted_context

    def _conversation_turn_ref(self, turn: Any) -> JsonDict:
        if not isinstance(turn, dict):
            return {"type": "conversation_turn", "id": ""}
        return {
            "type": "conversation_turn",
            "id": str(turn.get("turn_id") or turn.get("source_event_id") or ""),
        }

    def public_context(self, runtime_context: JsonDict) -> JsonDict:
        return {
            key: value
            for key, value in runtime_context.items()
            if not key.startswith("_")
        }

    def model_tools(self, runtime_context: JsonDict) -> list[JsonDict]:
        tools = runtime_context.get("_model_tools")
        return list(tools) if isinstance(tools, list) else []

    def session_id_for(self, runtime_context: JsonDict) -> str:
        session_id = runtime_context.get("_generator_session")
        if isinstance(session_id, str) and session_id:
            return session_id
        decision = runtime_context.get("decision")
        if isinstance(decision, dict):
            session_id = decision.get("generator_session")
            if isinstance(session_id, str) and session_id:
                return session_id
        return self.default_session_id

    async def _session_for(self, session_id: str) -> GeneratorSession:
        if session_id not in self.sessions:
            if self.model_interface is None:
                raise RuntimeError(f"No model interface available for generator session: {session_id}")
            self.sessions[session_id] = LLMGeneratorSession(self.model_interface)
        session = self.sessions[session_id]
        if self._started:
            await session.start()
        return session

    def _build_system_prompt(self, runtime_context: JsonDict, *, session_id: str) -> str:
        agent = runtime_context.get("agent", {})
        prompt = self.agent_config.generator.prompt_for(session_id)
        profile_context = prompt.profile_context_template.format_map(
            SafeFormatDict(agent)
        )
        return (
            prompt.system_prompt.strip()
            + "\n\n"
            + profile_context.strip()
        )

    def _build_user_payload(self, runtime_context: JsonDict, *, session_id: str) -> str:
        prompt = self.agent_config.generator.prompt_for(session_id)
        event_context = {
            key: value for key, value in runtime_context.items() if key != "agent"
        }
        return (
            prompt.user_payload_prefix.strip()
            + "\n\n"
            + self._runtime_context_markdown(event_context)
        )

    def _add_decision_messages(self, context: Context, runtime_context: JsonDict) -> None:
        context.add(User(self._decision_context_markdown(runtime_context)))
        for run in self._tool_loop_runs(runtime_context):
            run_id = str(run.get("action_run_id") or "")
            action_name = str(run.get("action_name") or "")
            if not run_id or not action_name:
                continue
            context.add(
                Assistant(
                    content=self._assistant_tool_call_text(run),
                    actions=[
                        {
                            "id": run_id,
                            "name": action_name,
                            "arguments": run.get("args") or {},
                        }
                    ],
                )
            )
            context.add(
                Tool(
                    tool_id=run_id,
                    name=action_name,
                    content=self._tool_result_markdown(run),
                )
            )
        context.add(User(self._current_decision_request_markdown(runtime_context)))

    def _add_wernicke_messages(self, context: Context, runtime_context: JsonDict) -> None:
        conversation = runtime_context.get("conversation")
        if not isinstance(conversation, dict):
            conversation = {}
        self._add_prior_conversation_messages(
            context,
            conversation.get("recent_turns"),
        )
        initial = {
            key: value
            for key, value in runtime_context.items()
            if key not in {"agent", "conversation"}
        }
        current = {
            key: value
            for key, value in conversation.items()
            if key not in {"recent_turns", "tool_history"}
        }
        initial["conversation"] = current
        prompt = self.agent_config.generator.prompt_for("wernicke")
        context.add(
            User(
                prompt.user_payload_prefix.strip()
                + "\n\n"
                + self._runtime_context_markdown(initial)
            )
        )
        tool_history = conversation.get("tool_history")
        if isinstance(tool_history, list):
            for round_data in tool_history:
                if not isinstance(round_data, dict):
                    continue
                calls = round_data.get("tool_calls")
                if not isinstance(calls, list) or not calls:
                    continue
                context.add(
                    Assistant(
                        content=str(round_data.get("assistant_text") or ""),
                        actions=[
                            {
                                "id": str(call.get("id") or ""),
                                "name": str(call.get("name") or ""),
                                "arguments": call.get("arguments") or {},
                            }
                            for call in calls
                            if isinstance(call, dict)
                        ],
                    )
                )
                results = round_data.get("tool_results")
                if not isinstance(results, list):
                    continue
                for result in results:
                    if not isinstance(result, dict):
                        continue
                    context.add(
                        Tool(
                            tool_id=str(result.get("tool_call_id") or ""),
                            name=str(result.get("name") or ""),
                            content=self._markdown_value(result.get("content")),
                        )
                    )
        if tool_history:
            context.add(
                User(
                    "请根据刚才的内部工具结果继续理解。需要更多信息时继续调用工具；"
                    "理解完成后调用 commit_understanding。"
                )
            )

    def _add_broca_messages(self, context: Context, runtime_context: JsonDict) -> None:
        conversation = runtime_context.get("conversation")
        if not isinstance(conversation, dict):
            conversation = {}
        self._add_prior_conversation_messages(
            context,
            conversation.get("recent_turns"),
        )
        current_context = {
            key: value for key, value in runtime_context.items() if key != "agent"
        }
        current_context["conversation"] = {
            key: value
            for key, value in conversation.items()
            if key != "recent_turns"
        }
        prompt = self.agent_config.generator.prompt_for("broca")
        context.add(
            User(
                prompt.user_payload_prefix.strip()
                + "\n\n"
                + self._runtime_context_markdown(current_context)
            )
        )

    def _add_prior_conversation_messages(self, context: Context, turns: Any) -> None:
        if not isinstance(turns, list):
            return
        summarized_history: list[str] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            history_tier = str(turn.get("history_tier") or "verbatim")
            if history_tier in {"compact", "summary"}:
                summarized_history.append(
                    self._conversation_history_summary_line(turn)
                )
                continue
            if summarized_history:
                context.add(
                    User(
                        "# Earlier Conversation Summary\n"
                        + "\n".join(summarized_history)
                    )
                )
                summarized_history = []
            utterance = str(turn.get("utterance") or "").strip()
            if utterance:
                speaker = str(turn.get("speaker_id") or "对方")
                context.add(User(f"[{speaker}] {utterance}"))
            response = str(turn.get("response_text") or "").strip()
            if response:
                context.add(Assistant(content=response))
        if summarized_history:
            context.add(
                User(
                    "# Earlier Conversation Summary\n"
                    + "\n".join(summarized_history)
                )
            )

    def _conversation_history_summary_line(self, turn: JsonDict) -> str:
        turn_id = str(turn.get("turn_id") or "unknown")
        history_tier = str(turn.get("history_tier") or "summary")
        if history_tier == "compact":
            meaning = turn.get("utterance_summary")
            speech_act = turn.get("speech_act")
            intents = turn.get("intents")
            response_intent = turn.get("agent_response_intent")
            status = turn.get("status")
        else:
            exchange = turn.get("exchange_summary")
            if not isinstance(exchange, dict):
                exchange = {}
            meaning = exchange.get("meaning")
            speech_act = exchange.get("speech_act")
            intents = None
            response_intent = exchange.get("agent_response_intent")
            status = exchange.get("status")
        parts = [
            f"含义={self._inline_value(meaning)}",
            f"言语行为={self._inline_value(speech_act)}",
        ]
        if intents:
            parts.append(f"意图={self._inline_value(intents)}")
        if response_intent:
            parts.append(f"Agent表达意图={self._inline_value(response_intent)}")
        if status:
            parts.append(f"状态={self._inline_value(status)}")
        return f"- [{history_tier}] {turn_id}: " + "; ".join(parts)

    def _decision_context_markdown(self, runtime_context: JsonDict) -> str:
        decision = runtime_context.get("decision") if isinstance(runtime_context.get("decision"), dict) else {}
        runtime = runtime_context.get("runtime") if isinstance(runtime_context.get("runtime"), dict) else {}
        workspace = runtime_context.get("workspace") if isinstance(runtime_context.get("workspace"), dict) else {}
        tooling = runtime_context.get("tooling") if isinstance(runtime_context.get("tooling"), dict) else {}
        focus = runtime_context.get("focus") if isinstance(runtime_context.get("focus"), dict) else {}
        cognition = runtime_context.get("cognition") if isinstance(runtime_context.get("cognition"), dict) else {}
        long_term_memory = runtime_context.get("long_term_memory") if isinstance(runtime_context.get("long_term_memory"), dict) else {}
        context_selection = runtime_context.get("context_selection") if isinstance(runtime_context.get("context_selection"), dict) else {}
        action_guidance = runtime.get("action_guidance") if isinstance(runtime.get("action_guidance"), dict) else {}

        lines = [
            "# Agent Decision Context",
            "",
            "这是一段事件驱动 Agent 的工作记忆摘要。可调用工具的完整 schema 在本次模型请求的 tools 字段中，不在正文重复展开。",
            "",
            "## Runtime",
            f"- 模式: {runtime.get('mode', '')}",
            f"- 下一步提示: {decision.get('next_step_hint', '')}",
            f"- 当前任务: {workspace.get('current_task_id') or '无'}",
            f"- 上次决策摘要: {workspace.get('last_decision_summary') or '无'}",
            "",
            "## Focus Task",
            self._task_markdown(focus.get("task")),
            "",
            "## Available Tool Names",
            self._tool_names_markdown(action_guidance, tooling),
            "",
            "## Selected Workspace References",
            self._markdown_value(workspace.get("selected_refs") or []),
            "",
            "## Context Selection",
            self._markdown_value(context_selection),
            "",
            "## Active Action Runs",
            self._active_action_runs_markdown(focus.get("action_runs") or []),
            "",
            "## Recent Evidence",
            self._evidence_markdown(runtime_context.get("evidence") or []),
            "",
            "## Cognition State",
            self._markdown_value(cognition),
            "",
            "## Recalled Long-term Memory",
            self._markdown_value(long_term_memory),
            "",
            "## Visible Tasks",
            self._tasks_markdown(runtime_context.get("tasks") or []),
        ]
        return "\n".join(lines).strip()

    def _current_decision_request_markdown(self, runtime_context: JsonDict) -> str:
        decision = runtime_context.get("decision") if isinstance(runtime_context.get("decision"), dict) else {}
        trigger = decision.get("trigger") if isinstance(decision.get("trigger"), dict) else {}
        payload = trigger.get("payload") if isinstance(trigger.get("payload"), dict) else {}
        content = payload.get("content")
        lines = [
            "# Current Event",
            f"- Event type: {trigger.get('type', '')}",
            f"- Source: {trigger.get('source', '')}",
            f"- Task: {trigger.get('task_id') or '无'}",
        ]
        if content:
            lines.extend(["", "## User / Event Content", str(content)])
            if trigger.get("type") == "conversation.decision.requested":
                lines.extend(
                    [
                        "",
                        "## Wernicke Understanding",
                        self._markdown_value(payload.get("understanding")),
                        "",
                        "## Decision Request",
                        str(payload.get("decision_request") or "请判断是否需要行动或形成表达意图。"),
                        "",
                        "Wernicke 的理解是带说话者来源和置信度的内部解释，不是已验证的世界事实。",
                    ]
                )
        elif (
            trigger.get("type") in {"action.completed", "action.failed", "action.cancelled"}
            and trigger.get("action_run_id")
        ):
            lines.extend(
                [
                    "",
                    "## Event Payload",
                    "The complete triggering tool result is represented by the assistant/tool message pair above.",
                ]
            )
        else:
            lines.extend(["", "## Event Payload", self._markdown_value(payload)])
        lines.extend(
            [
                "",
                "请基于以上上下文决定下一步：",
                "- 需要外部或内部能力时，在本次响应中直接发起 tool_call。",
                "- 只需要对外表达时，用自然语言写清表达意图；Runtime 会交给 BrocaSystem 组织最终话语。",
                "- 如果任务仍未完成，不要只描述计划；继续调用必要工具或明确等待。",
            ]
        )
        return "\n".join(lines).strip()

    def _tool_loop_runs(self, runtime_context: JsonDict) -> list[JsonDict]:
        focus = runtime_context.get("focus")
        if not isinstance(focus, dict):
            return []
        runs = focus.get("action_runs") or []
        selected: list[JsonDict] = []
        for run in runs:
            if not isinstance(run, dict):
                continue
            if run.get("status") in {"succeeded", "failed", "cancelled"}:
                selected.append(run)
        return selected

    def _active_action_runs_markdown(self, runs: Any) -> str:
        if not isinstance(runs, list):
            return "无。"
        active = [
            run
            for run in runs
            if isinstance(run, dict) and run.get("status") in {"created", "running"}
        ]
        if not active:
            return "无。"
        return self._markdown_value(active)

    def _assistant_tool_call_text(self, run: JsonDict) -> str:
        action_name = run.get("action_name") or "unknown_action"
        status = run.get("status") or "unknown"
        return f"我调用工具 `{action_name}`，等待工具返回结果。当前记录状态：{status}。"

    def _tool_result_markdown(self, run: JsonDict) -> str:
        lines = [
            f"# Tool Result: {run.get('action_name', '')}",
            f"- Action run: {run.get('action_run_id', '')}",
            f"- Status: {run.get('status', '')}",
        ]
        if run.get("result") is not None:
            lines.extend(["", "## Result", self._markdown_value(run.get("result"))])
        if run.get("error"):
            lines.extend(["", "## Error", self._markdown_value(run.get("error"))])
        if run.get("progress"):
            lines.extend(["", "## Progress", self._markdown_value(run.get("progress"))])
        return "\n".join(lines).strip()

    def _runtime_context_markdown(self, runtime_context: JsonDict) -> str:
        lines = ["# Runtime Context"]
        for key, value in runtime_context.items():
            lines.extend(["", f"## {key}", self._markdown_value(value)])
        return "\n".join(lines).strip()

    def _task_markdown(self, task: Any) -> str:
        if not isinstance(task, dict) or not task:
            return "无任务焦点。"
        lines = [
            f"- Task ID: {task.get('task_id', '')}",
            f"- 标题: {task.get('title', '')}",
            f"- 目标: {task.get('goal', '')}",
            f"- 状态: {task.get('status', '')}",
            f"- 父任务: {task.get('parent_task_id') or '无'}",
            f"- 子任务: {self._inline_value(task.get('child_task_ids') or [])}",
            f"- 依赖任务: {self._inline_value(task.get('dependencies') or [])}",
            f"- 等待: {self._inline_value(task.get('waiting_on') or [])}",
            f"- 调度判断: {self._inline_or_block(task.get('scheduling') or {}, indent=2)}",
            f"- 进度: {self._inline_value(task.get('progress') or {})}",
        ]
        if task.get("result"):
            lines.extend(["- 结果:", self._indent(self._markdown_value(task.get("result")))])
        if task.get("error"):
            lines.extend(["- 错误:", self._indent(self._markdown_value(task.get("error")))])
        return "\n".join(lines)

    def _tool_names_markdown(self, action_guidance: JsonDict, tooling: JsonDict) -> str:
        lines = []
        groups = [
            ("外部环境工具", action_guidance.get("candidate_external_action_names") or []),
            ("内部 runtime 工具", action_guidance.get("candidate_internal_runtime_action_names") or []),
            ("本地工具", action_guidance.get("candidate_local_action_names") or []),
        ]
        for label, names in groups:
            if names:
                lines.append(f"- {label}: {', '.join(str(name) for name in names)}")
        if not lines:
            names = tooling.get("candidate_action_names") or []
            lines.append(f"- 候选工具: {', '.join(str(name) for name in names) if names else '无'}")
        return "\n".join(lines)

    def _evidence_markdown(self, evidence: Any) -> str:
        if not isinstance(evidence, list) or not evidence:
            return "无。"
        sections: list[str] = []
        for index, item in enumerate(evidence, start=1):
            if not isinstance(item, dict):
                sections.append(f"{index}. {self._inline_value(item)}")
                continue
            lines = [f"{index}. {item.get('type', 'evidence')}"]
            summary = item.get("summary")
            if summary:
                lines.append(f"   - 摘要: {summary}")
            for key in ("role", "content", "task_id", "data"):
                if key in item and item.get(key) not in (None, ""):
                    lines.append(f"   - {key}: {self._inline_or_block(item.get(key), indent=5)}")
            sections.append("\n".join(lines))
        return "\n".join(sections)

    def _tasks_markdown(self, tasks: Any) -> str:
        if not isinstance(tasks, list) or not tasks:
            return "无。"
        return "\n\n".join(self._task_markdown(task) for task in tasks if isinstance(task, dict))

    def _markdown_value(self, value: Any, *, depth: int = 0) -> str:
        if value is None:
            return "无"
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        if depth >= 3:
            return self._inline_value(value)
        if isinstance(value, dict):
            if not value:
                return "空"
            lines = []
            for key, item in value.items():
                if isinstance(item, (dict, list)) and item:
                    lines.append(f"- {key}:")
                    lines.append(self._indent(self._markdown_value(item, depth=depth + 1)))
                else:
                    lines.append(f"- {key}: {self._inline_value(item)}")
            return "\n".join(lines)
        if isinstance(value, list):
            if not value:
                return "空"
            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append("-")
                    lines.append(self._indent(self._markdown_value(item, depth=depth + 1)))
                else:
                    lines.append(f"- {self._inline_value(item)}")
            return "\n".join(lines)
        return str(value)

    def _inline_or_block(self, value: Any, *, indent: int = 0) -> str:
        if isinstance(value, (dict, list)):
            return "\n" + self._indent(self._markdown_value(value), spaces=indent)
        return self._inline_value(value)

    def _inline_value(self, value: Any) -> str:
        if value is None:
            return "无"
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            if not value:
                return "空"
            if all(not isinstance(item, (dict, list)) for item in value):
                return ", ".join(str(item) for item in value)
        if isinstance(value, dict):
            if not value:
                return "空"
            return "; ".join(f"{key}={item}" for key, item in value.items())
        return str(value)

    def _indent(self, text: str, *, spaces: int = 2) -> str:
        prefix = " " * spaces
        return "\n".join(prefix + line if line else line for line in text.splitlines())
