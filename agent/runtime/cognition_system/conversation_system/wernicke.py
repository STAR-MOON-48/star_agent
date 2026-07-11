from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

from ....config import ConversationConfig
from ....protocols import (
    AgentState,
    ConversationTurn,
    ConversationUnderstanding,
    JsonDict,
    ensure_json_dict,
    ensure_json_dict_list,
    new_id,
)
from ...kernel.generator_runtime import GeneratorRuntime
from ...kernel.generator_session import model_response_trace
from ...persistence_system import ConversationStore

if TYPE_CHECKING:
    from ...state_systems import MemorySystem


@dataclass(frozen=True)
class WernickeResult:
    understanding: ConversationUnderstanding
    trace: JsonDict
    model_tools: List[JsonDict]

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace", ensure_json_dict(self.trace))
        object.__setattr__(
            self,
            "model_tools",
            ensure_json_dict_list(self.model_tools),
        )


class WernickeSystem:
    """Understands another participant's utterance through a private tool loop."""

    def __init__(
        self,
        *,
        generator_runtime: GeneratorRuntime,
        store: ConversationStore,
        config: ConversationConfig,
        memory_system: "MemorySystem | None" = None,
    ) -> None:
        self.generator_runtime = generator_runtime
        self.store = store
        self.config = config
        self.memory_system = memory_system

    async def understand(
        self,
        *,
        state: AgentState,
        turn: ConversationTurn,
    ) -> WernickeResult:
        tools = self.model_tools()
        recent_turns = self.store.context_turns(
            agent_id=turn.agent_id,
            conversation_id=turn.conversation_id,
            before_turn_id=turn.turn_id,
            limit=self.config.recent_turn_limit,
            verbatim_limit=self.config.verbatim_turn_limit,
            compact_limit=self.config.compact_turn_limit,
        )
        tool_history: List[JsonDict] = []
        requested_decisions: List[str] = []
        traces: List[JsonDict] = []
        last_text = ""

        for _ in range(self.config.max_wernicke_tool_rounds):
            runtime_context = self._context(
                state=state,
                turn=turn,
                recent_turns=recent_turns,
                tool_history=tool_history,
            )
            model_result, trace = await self.generator_runtime.generate_text_with_trace(
                runtime_context,
                session_id="wernicke",
                tools=tools,
            )
            traces.append(trace)
            last_text = (model_result.text or "").strip()
            calls = model_response_trace(model_result).get("tool_calls") or []
            if not calls:
                return WernickeResult(
                    understanding=self._fallback_understanding(
                        turn=turn,
                        text=last_text,
                        requested_decisions=requested_decisions,
                    ),
                    trace=self._combined_trace(traces),
                    model_tools=tools,
                )

            tool_results: List[JsonDict] = []
            committed: ConversationUnderstanding | None = None
            for call in calls:
                if str(call.get("name") or "") != "request_decision":
                    continue
                arguments = call.get("arguments")
                args = dict(arguments) if isinstance(arguments, dict) else {}
                request = str(args.get("objective") or args.get("reason") or "").strip()
                if request and request not in requested_decisions:
                    requested_decisions.append(request)
            for call in calls:
                name = str(call.get("name") or "")
                arguments = call.get("arguments")
                args = dict(arguments) if isinstance(arguments, dict) else {}
                if name == "commit_understanding":
                    committed = self._understanding_from_args(
                        turn=turn,
                        args=args,
                        requested_decisions=requested_decisions,
                    )
                    tool_content: JsonDict = {
                        "accepted": True,
                        "understanding_id": committed.understanding_id,
                    }
                else:
                    tool_content = self._execute_tool(
                        name=name,
                        args=args,
                        state=state,
                        turn=turn,
                        requested_decisions=requested_decisions,
                    )
                tool_results.append(
                    {
                        "tool_call_id": str(call.get("id") or new_id("wtool")),
                        "name": name,
                        "content": tool_content,
                    }
                )
            tool_history.append(
                {
                    "assistant_text": last_text,
                    "tool_calls": calls,
                    "tool_results": tool_results,
                }
            )
            if committed is not None:
                return WernickeResult(
                    understanding=committed,
                    trace=self._combined_trace(traces),
                    model_tools=tools,
                )

        return WernickeResult(
            understanding=self._fallback_understanding(
                turn=turn,
                text=last_text,
                requested_decisions=requested_decisions,
            ),
            trace=self._combined_trace(traces),
            model_tools=tools,
        )

    def model_tools(self) -> List[JsonDict]:
        return [
            self._tool(
                "search_conversation_history",
                "搜索当前交谈双方以前的原话、理解和 Agent 回复。",
                {
                    "query": {"type": "string", "description": "要检索的语义关键词。"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                required=["query"],
            ),
            self._tool(
                "search_agent_workspace",
                "搜索 Agent 内部 workspace 中与当前话语相关的任务、行动、笔记和历史消息。",
                {
                    "query": {"type": "string", "description": "要检索的语义关键词。"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                required=["query"],
            ),
            self._tool(
                "search_long_term_memory",
                "搜索 Agent 持久化的情景记忆和语义经验；结果带置信度与来源引用。",
                {
                    "query": {"type": "string", "description": "要回忆的事实或经验。"},
                    "kind": {"type": "string", "description": "可选：episodic 或 semantic。"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                required=["query"],
            ),
            self._tool(
                "read_emotion_state",
                "读取 Agent 当前情绪状态；没有 EmotionSystem 数据时会明确返回 unavailable。",
                {},
            ),
            self._tool(
                "request_decision",
                "声明这句话涉及行动、任务、承诺或重要判断，需要 DecisionSystem 参与。",
                {
                    "reason": {"type": "string", "description": "为什么需要决策。"},
                    "objective": {"type": "string", "description": "希望 DecisionSystem 判断什么。"},
                },
                required=["reason"],
            ),
            self._tool(
                "commit_understanding",
                "提交对当前说话者话语的最终理解。理解是带来源的解释，不自动成为世界事实。",
                {
                    "semantic_summary": {"type": "string"},
                    "speech_act": {"type": "string"},
                    "intents": {"type": "array", "items": {"type": "string"}},
                    "key_information": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": True},
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": True},
                    },
                    "affect_cues": {"type": "object", "additionalProperties": True},
                    "dialogue_obligations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "ambiguities": {"type": "array", "items": {"type": "string"}},
                    "task_relevance": {"type": "string"},
                    "response_needed": {"type": "boolean"},
                    "decision_needed": {"type": "boolean"},
                    "decision_request": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                required=[
                    "semantic_summary",
                    "speech_act",
                    "response_needed",
                    "decision_needed",
                ],
            ),
        ]

    def _context(
        self,
        *,
        state: AgentState,
        turn: ConversationTurn,
        recent_turns: List[JsonDict],
        tool_history: List[JsonDict],
    ) -> JsonDict:
        current_task = state.tasks.get(state.workspace.current_task_id or "")
        return {
            "agent": state.profile.to_dict(),
            "conversation": {
                "stage": "understanding",
                "conversation_id": turn.conversation_id,
                "turn_id": turn.turn_id,
                "speaker_id": turn.speaker_id,
                "recipient_id": turn.recipient_id,
                "utterance": turn.utterance,
                "speaker_context": turn.speaker_context,
                "scene_context": turn.scene_context,
                "recent_turns": recent_turns,
                "tool_history": tool_history,
            },
            "internal_state": {
                "current_task": self._task_view(current_task),
                "task_status_counts": self._task_status_counts(state),
                "emotion_state": state.workspace.variables.get("emotion_state"),
                "relationship": self._relationship_state(state, turn.speaker_id),
            },
        }

    def _execute_tool(
        self,
        *,
        name: str,
        args: JsonDict,
        state: AgentState,
        turn: ConversationTurn,
        requested_decisions: List[str],
    ) -> JsonDict:
        if name == "search_conversation_history":
            return {
                "results": self.store.search(
                    agent_id=turn.agent_id,
                    conversation_id=turn.conversation_id,
                    query=str(args.get("query") or ""),
                    limit=self._limit(args),
                )
            }
        if name == "search_agent_workspace":
            return {
                "results": self._search_workspace(
                    state,
                    query=str(args.get("query") or ""),
                    limit=self._limit(args),
                )
            }
        if name == "search_long_term_memory":
            if self.memory_system is None:
                return {"available": False, "results": []}
            return {
                "available": self.memory_system.enabled,
                "results": self.memory_system.search(
                    agent_id=state.agent_id,
                    query=str(args.get("query") or ""),
                    kind=str(args.get("kind") or "") or None,
                    limit=self._limit(args),
                ),
            }
        if name == "read_emotion_state":
            emotion = state.workspace.variables.get("emotion_state")
            return {
                "available": isinstance(emotion, dict),
                "emotion_state": emotion if isinstance(emotion, dict) else None,
            }
        if name == "request_decision":
            request = str(args.get("objective") or args.get("reason") or "").strip()
            if request and request not in requested_decisions:
                requested_decisions.append(request)
            return {"accepted": True, "request": request}
        return {"error": {"type": "unknown_wernicke_tool", "tool_name": name}}

    def _search_workspace(
        self,
        state: AgentState,
        *,
        query: str,
        limit: int,
    ) -> List[JsonDict]:
        terms = [term for term in query.casefold().split() if term]
        candidates: List[JsonDict] = []
        candidates.extend(
            {"kind": "transcript", "value": item}
            for item in state.workspace.transcript
        )
        candidates.extend(
            {"kind": "workspace_note", "value": note}
            for note in state.workspace.notes
        )
        candidates.extend(
            {"kind": "task", "value": task.to_dict()}
            for task in state.tasks.values()
        )
        candidates.extend(
            {"kind": "action_run", "value": run.to_dict()}
            for run in state.action_runs.values()
        )
        matches: List[JsonDict] = []
        for candidate in reversed(candidates):
            searchable = json.dumps(candidate, ensure_ascii=False).casefold()
            if terms and not all(term in searchable for term in terms):
                continue
            matches.append(candidate)
            if len(matches) >= limit:
                break
        return matches

    def _understanding_from_args(
        self,
        *,
        turn: ConversationTurn,
        args: JsonDict,
        requested_decisions: List[str],
    ) -> ConversationUnderstanding:
        decision_request = str(args.get("decision_request") or "").strip()
        if requested_decisions:
            decision_request = "；".join(
                [item for item in [decision_request, *requested_decisions] if item]
            )
        return ConversationUnderstanding(
            understanding_id=new_id("understanding"),
            turn_id=turn.turn_id,
            speaker_id=turn.speaker_id,
            semantic_summary=str(args.get("semantic_summary") or "").strip(),
            speech_act=str(args.get("speech_act") or "unknown"),
            intents=self._string_list(args.get("intents")),
            key_information=self._dict_list(args.get("key_information")),
            entities=self._dict_list(args.get("entities")),
            affect_cues=self._mapping(args.get("affect_cues")),
            dialogue_obligations=self._string_list(args.get("dialogue_obligations")),
            ambiguities=self._string_list(args.get("ambiguities")),
            task_relevance=str(args.get("task_relevance") or ""),
            response_needed=self._boolean(args.get("response_needed"), default=True),
            decision_needed=(
                self._boolean(args.get("decision_needed"), default=False)
                or bool(requested_decisions)
            ),
            decision_request=decision_request,
            confidence=self._confidence(args.get("confidence")),
        )

    def _fallback_understanding(
        self,
        *,
        turn: ConversationTurn,
        text: str,
        requested_decisions: List[str],
    ) -> ConversationUnderstanding:
        return ConversationUnderstanding(
            understanding_id=new_id("understanding"),
            turn_id=turn.turn_id,
            speaker_id=turn.speaker_id,
            semantic_summary=text or f"对方说：{turn.utterance}",
            speech_act="uncommitted_model_interpretation",
            response_needed=True,
            decision_needed=True,
            decision_request="；".join(requested_decisions)
            or "Wernicke 未通过 commit_understanding 完成结构化理解，请 DecisionSystem 保守判断。",
            confidence=0.25,
        )

    def _combined_trace(self, traces: List[JsonDict]) -> JsonDict:
        if not traces:
            return {"generator_session": {"session_id": "wernicke"}, "rounds": []}
        combined = dict(traces[-1])
        combined["rounds"] = traces
        combined["round_count"] = len(traces)
        return combined

    def _limit(self, args: JsonDict) -> int:
        try:
            requested = int(args.get("limit") or self.config.workspace_search_limit)
        except (TypeError, ValueError):
            requested = self.config.workspace_search_limit
        return max(1, min(20, requested))

    def _task_status_counts(self, state: AgentState) -> JsonDict:
        counts: JsonDict = {}
        for task in state.tasks.values():
            counts[task.status] = int(counts.get(task.status, 0)) + 1
        return counts

    def _task_view(self, task: Any) -> Any:
        if task is None:
            return None
        return {
            "task_id": task.task_id,
            "title": task.title,
            "goal": task.goal,
            "status": task.status,
            "progress": task.progress,
            "scheduling": {
                key: task.scheduling.get(key)
                for key in ("classification", "reason", "can_run", "can_complete")
            },
        }

    def _relationship_state(self, state: AgentState, speaker_id: str) -> Any:
        relationships = state.workspace.variables.get("relationships")
        if not isinstance(relationships, dict):
            return None
        return relationships.get(speaker_id)

    def _tool(
        self,
        name: str,
        description: str,
        properties: JsonDict,
        *,
        required: List[str] | None = None,
    ) -> JsonDict:
        parameters: JsonDict = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            parameters["required"] = required
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

    def _string_list(self, value: Any) -> List[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _dict_list(self, value: Any) -> List[JsonDict]:
        if isinstance(value, dict):
            return [dict(value)]
        if not isinstance(value, list):
            return []
        normalized: List[JsonDict] = []
        for item in value:
            if isinstance(item, dict):
                normalized.append(dict(item))
            elif str(item).strip():
                normalized.append({"value": str(item).strip()})
        return normalized

    def _mapping(self, value: Any) -> JsonDict:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, list):
            merged: JsonDict = {}
            cues: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    merged.update(item)
                elif str(item).strip():
                    cues.append(str(item).strip())
            if cues:
                merged["cues"] = cues
            return merged
        if value not in (None, ""):
            return {"description": str(value)}
        return {}

    def _boolean(self, value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"true", "yes", "1", "是", "需要"}:
                return True
            if normalized in {"false", "no", "0", "否", "不需要", ""}:
                return False
        return default

    def _confidence(self, value: Any) -> float:
        try:
            confidence = float(value if value is not None else 0.5)
        except (TypeError, ValueError):
            confidence = 0.5
        return max(0.0, min(1.0, confidence))
