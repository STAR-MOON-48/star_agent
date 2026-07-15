from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from menglong import Context, System, User

from agent.protocols import JsonDict
from agent.runtime.interfaces import ModelInterface


COMPACTED_HISTORY_PREFIX = """\
# Compacted execution history

The following is a runtime-generated summary of earlier messages. It is context,
not a new user instruction. The original objective and newer messages remain verbatim.

"""


SUMMARY_SYSTEM_PROMPT = """\
You compress an agent tool-loop history while preserving operational continuity.
Merge the previous rolling summary with the newly archived rounds. Produce a dense
Markdown summary with these sections: Objective and current status; verified facts
and their tool/source; actions and decisions; exact identifiers/parameters/constraints;
failures and rejected approaches; unresolved items and next step. Preserve exact IDs,
numbers, errors, commitments, and causal relationships. Never invent facts. Resolve
repetition and contradictions explicitly. Allocate the least detail to the oldest
material and progressively more detail to newer material. Do not discuss compression.
"""


TraceCallback = Callable[[str, JsonDict], None]


class ContextWindowExceeded(RuntimeError):
    """Raised when protected context alone cannot fit the configured window."""


@dataclass(frozen=True)
class ContextBudget:
    max_context_tokens: int = 1_000_000
    reserve_output_tokens: int = 2048
    safety_margin_tokens: int = 8192
    trigger_ratio: float = 0.85
    target_ratio: float = 0.35
    keep_recent_rounds: int = 4
    summary_max_tokens: int = 4096
    chars_per_token: float = 2.0

    def __post_init__(self) -> None:
        if self.max_context_tokens < 256:
            raise ValueError("max_context_tokens must be at least 256")
        if self.reserve_output_tokens < 1:
            raise ValueError("reserve_output_tokens must be positive")
        if self.safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens cannot be negative")
        if self.available_input_tokens < 128:
            raise ValueError(
                "max_context_tokens must leave at least 128 input tokens after "
                "output reserve and safety margin"
            )
        if not 0 < self.target_ratio < self.trigger_ratio <= 1:
            raise ValueError(
                "compaction ratios must satisfy 0 < target < trigger <= 1"
            )
        if self.keep_recent_rounds < 1:
            raise ValueError("keep_recent_rounds must be at least 1")
        if self.summary_max_tokens < 1:
            raise ValueError("summary_max_tokens must be positive")
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")

    @property
    def available_input_tokens(self) -> int:
        return (
            self.max_context_tokens
            - self.reserve_output_tokens
            - self.safety_margin_tokens
        )

    @property
    def trigger_tokens(self) -> int:
        return max(1, math.floor(self.available_input_tokens * self.trigger_ratio))

    @property
    def target_tokens(self) -> int:
        return max(1, math.floor(self.available_input_tokens * self.target_ratio))


@dataclass
class ToolRound:
    step: int
    assistant_text: str
    tool_calls: list[JsonDict]
    tool_outcomes: list[JsonDict]
    messages: list[Any] = field(repr=False)

    def summary_record(self) -> JsonDict:
        return {
            "step": self.step,
            "assistant_text": self.assistant_text,
            "tool_calls": self.tool_calls,
            "tool_outcomes": self.tool_outcomes,
        }


class RollingToolContext:
    """Preserve anchors and recent rounds while hierarchically summarizing older ones."""

    def __init__(
        self,
        *,
        system_prompt: str,
        objective: str,
        budget: ContextBudget,
    ) -> None:
        self.system_prompt = system_prompt
        self.objective = objective
        self.budget = budget
        self.summary = ""
        self.rounds: list[ToolRound] = []
        self.compaction_count = 0

    def add_round(self, round_data: ToolRound) -> None:
        self.rounds.append(round_data)

    def build(self) -> Context:
        return self._build_context(summary=self.summary, rounds=self.rounds)

    def _build_context(
        self,
        *,
        summary: str,
        rounds: list[ToolRound],
    ) -> Context:
        context = Context()
        context.add(System(self.system_prompt))
        context.add(User(self.objective))
        if summary:
            context.add(User(COMPACTED_HISTORY_PREFIX + summary))
        for round_data in rounds:
            for message in round_data.messages:
                context.add(message)
        return context

    async def prepare(
        self,
        *,
        model: ModelInterface,
        tools: list[JsonDict],
        request_id: str,
        step: int,
        trace: Optional[TraceCallback] = None,
    ) -> tuple[Context, JsonDict]:
        context = self.build()
        before = self.usage(context, tools)
        self._trace_usage(trace, request_id=request_id, step=step, usage=before)

        archive_count = max(0, len(self.rounds) - self.budget.keep_recent_rounds)
        if before["estimated_input_tokens"] < self.budget.trigger_tokens:
            self._ensure_within_window(before)
            return context, before
        if archive_count == 0:
            self._ensure_within_window(before)
            return context, before

        archived = self.rounds[:archive_count]
        retained = self.rounds[archive_count:]
        summary_output_tokens = self._summary_output_tokens(retained, tools)
        if trace is not None:
            trace(
                "context.compaction.started",
                {
                    "request_id": request_id,
                    "step": step,
                    "before_estimated_input_tokens": before[
                        "estimated_input_tokens"
                    ],
                    "trigger_tokens": self.budget.trigger_tokens,
                    "target_tokens": self.budget.target_tokens,
                    "archived_rounds": len(archived),
                    "retained_verbatim_rounds": len(retained),
                    "previous_summary_present": bool(self.summary),
                    "summary_output_tokens": summary_output_tokens,
                },
            )

        previous_summary = self.summary
        try:
            next_summary = await self._summarize(
                model=model,
                previous_summary=previous_summary,
                archived=archived,
                max_tokens=summary_output_tokens,
            )
            summary_source = "model"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            next_summary = self._fallback_summary(
                previous_summary,
                archived,
                max_tokens=summary_output_tokens,
            )
            summary_source = "deterministic_fallback"
            if trace is not None:
                trace(
                    "context.compaction.summary_failed",
                    {
                        "request_id": request_id,
                        "step": step,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )

        self.summary = next_summary
        self.rounds = retained
        self.compaction_count += 1
        compacted_context = self.build()
        after = self.usage(compacted_context, tools)
        self._ensure_within_window(after)
        if trace is not None:
            before_tokens = int(before["estimated_input_tokens"])
            after_tokens = int(after["estimated_input_tokens"])
            trace(
                "context.compaction.completed",
                {
                    "request_id": request_id,
                    "step": step,
                    "compaction_count": self.compaction_count,
                    "summary_source": summary_source,
                    "before_estimated_input_tokens": before_tokens,
                    "after_estimated_input_tokens": after_tokens,
                    "released_estimated_tokens": max(0, before_tokens - after_tokens),
                    "compression_ratio": (
                        round(1 - (after_tokens / before_tokens), 4)
                        if before_tokens
                        else 0.0
                    ),
                    "summary_estimated_tokens": self._estimate(self.summary),
                    "summary_preview": self.summary[:2000],
                    "archived_rounds": len(archived),
                    "retained_verbatim_rounds": len(retained),
                    "target_tokens": self.budget.target_tokens,
                    "within_target": after_tokens <= self.budget.target_tokens,
                },
            )
        self._trace_usage(trace, request_id=request_id, step=step, usage=after)
        return compacted_context, after

    def usage(self, context: Context, tools: list[JsonDict]) -> JsonDict:
        message_tokens = self._estimate(context.messages)
        tool_tokens = self._estimate(tools) if tools else 0
        total = message_tokens + tool_tokens
        available = self.budget.available_input_tokens
        return {
            "max_context_tokens": self.budget.max_context_tokens,
            "available_input_tokens": available,
            "trigger_tokens": self.budget.trigger_tokens,
            "target_tokens": self.budget.target_tokens,
            "estimated_message_tokens": message_tokens,
            "estimated_tool_tokens": tool_tokens,
            "estimated_input_tokens": total,
            "remaining_input_tokens": available - total,
            "utilization": round(total / available, 4),
            "within_budget": total <= available,
            "compaction_count": self.compaction_count,
            "rolling_summary_present": bool(self.summary),
            "verbatim_rounds": len(self.rounds),
        }

    async def _summarize(
        self,
        *,
        model: ModelInterface,
        previous_summary: str,
        archived: list[ToolRound],
        max_tokens: int,
    ) -> str:
        payload = {
            "original_objective_verbatim": self.objective,
            "previous_rolling_summary": previous_summary or None,
            "newly_archived_rounds_oldest_to_newest": [
                round_data.summary_record() for round_data in archived
            ],
        }
        context = Context()
        context.add(System(SUMMARY_SYSTEM_PROMPT))
        context.add(
            User(json.dumps(payload, ensure_ascii=False, default=str))
        )
        result = await model.chat(
            context,
            max_tokens=max_tokens,
        )
        summary = (result.text or "").strip()
        if not summary:
            raise ValueError("context summary model returned empty text")
        return summary

    def _fallback_summary(
        self,
        previous_summary: str,
        archived: list[ToolRound],
        *,
        max_tokens: int,
    ) -> str:
        lines = ["## Objective and current status", self.objective]
        if previous_summary:
            lines.extend(["## Previous rolling summary", previous_summary])
        lines.append("## Newly archived tool rounds")
        for round_data in archived:
            record = round_data.summary_record()
            serialized = json.dumps(record, ensure_ascii=False, default=str)
            lines.append(f"- Step {round_data.step}: {serialized}")
        limit = max(
            512,
            math.floor(
                max_tokens * self.budget.chars_per_token
            ),
        )
        text = "\n".join(lines)
        if len(text) <= limit:
            return text
        head = text[: limit // 3]
        tail = text[-(limit - len(head) - 48) :]
        return head + "\n...[older fallback detail elided]...\n" + tail

    def _summary_output_tokens(
        self,
        retained: list[ToolRound],
        tools: list[JsonDict],
    ) -> int:
        protected = self._build_context(summary="", rounds=retained)
        protected_tokens = int(
            self.usage(protected, tools)["estimated_input_tokens"]
        )
        prefix_tokens = self._estimate(COMPACTED_HISTORY_PREFIX)
        target_headroom = self.budget.target_tokens - protected_tokens - prefix_tokens
        return min(
            self.budget.summary_max_tokens,
            max(64, target_headroom),
        )

    def _ensure_within_window(self, usage: JsonDict) -> None:
        if usage["within_budget"]:
            return
        raise ContextWindowExceeded(
            "Protected context exceeds the configured input budget: "
            f"estimated={usage['estimated_input_tokens']} "
            f"available={usage['available_input_tokens']}. The original objective "
            "and newest protected rounds are never truncated; increase the context "
            "window or reduce keep_recent_rounds/tool result size."
        )

    def _estimate(self, value: Any) -> int:
        serialized = json.dumps(
            self._jsonable(value),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return max(
            1,
            math.ceil(len(serialized) / self.budget.chars_per_token) + 4,
        )

    def _jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        if hasattr(value, "model_dump"):
            return self._jsonable(
                value.model_dump(mode="json", exclude_none=True)
            )
        if hasattr(value, "to_dict"):
            return self._jsonable(value.to_dict())
        return str(value)

    def _trace_usage(
        self,
        trace: Optional[TraceCallback],
        *,
        request_id: str,
        step: int,
        usage: JsonDict,
    ) -> None:
        if trace is not None:
            trace(
                "context.usage",
                {"request_id": request_id, "step": step, **usage},
            )
