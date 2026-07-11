from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

from ...config import ContextPolicyConfig
from ...protocols import JsonDict, ensure_json_dict


CONTEXT_POLICY_VARIABLE_KEY = "context_policy"


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json", exclude_none=True))
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    return str(value)


def estimate_tokens(value: Any, policy: ContextPolicyConfig) -> int:
    serialized = json.dumps(
        jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return max(1, math.ceil(len(serialized) / policy.chars_per_token) + 4)


def stable_digest(value: Any) -> str:
    serialized = json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContextCandidate:
    ref: JsonDict
    value: Any
    priority: int
    order: int
    reason: str
    mandatory: bool = False
    token_multiplier: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", ensure_json_dict(self.ref))

    def estimated_tokens(self, policy: ContextPolicyConfig) -> int:
        return max(1, math.ceil(estimate_tokens(self.value, policy) * self.token_multiplier))


@dataclass(frozen=True)
class ContextSelection:
    selected: list[ContextCandidate]
    not_selected: list[ContextCandidate]
    used_tokens: int
    budget_tokens: int
    manifest: JsonDict

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", ensure_json_dict(self.manifest))


def select_candidates(
    candidates: Iterable[ContextCandidate],
    *,
    policy: ContextPolicyConfig,
    budget_tokens: int,
    initial_used_tokens: int = 0,
) -> ContextSelection:
    candidate_list = list(candidates)
    ranked = sorted(
        candidate_list,
        key=lambda candidate: (
            candidate.mandatory,
            candidate.priority,
            candidate.order,
        ),
        reverse=True,
    )
    selected: list[ContextCandidate] = []
    not_selected: list[ContextCandidate] = []
    used_tokens = max(0, initial_used_tokens)
    manifest_entries: list[JsonDict] = []

    for candidate in ranked:
        candidate_tokens = candidate.estimated_tokens(policy)
        fits = used_tokens + candidate_tokens <= budget_tokens
        is_selected = candidate.mandatory or fits
        if is_selected:
            selected.append(candidate)
            used_tokens += candidate_tokens
        else:
            not_selected.append(candidate)
        manifest_entries.append(
            {
                "ref": candidate.ref,
                "priority": candidate.priority,
                "reason": candidate.reason,
                "mandatory": candidate.mandatory,
                "estimated_tokens": candidate_tokens,
                "selected": is_selected,
                "selection_reason": (
                    "mandatory"
                    if candidate.mandatory
                    else "fits_budget"
                    if fits
                    else "available_by_reference"
                ),
            }
        )

    selected.sort(key=lambda candidate: candidate.order)
    not_selected.sort(key=lambda candidate: candidate.order)
    return ContextSelection(
        selected=selected,
        not_selected=not_selected,
        used_tokens=used_tokens,
        budget_tokens=budget_tokens,
        manifest={
            "budget_tokens": budget_tokens,
            "initial_used_tokens": initial_used_tokens,
            "used_tokens": used_tokens,
            "budget_exceeded_by_mandatory": max(0, used_tokens - budget_tokens),
            "selected_count": len(selected),
            "available_by_reference_count": len(not_selected),
            "entries": manifest_entries,
        },
    )


def model_request_usage(
    *,
    messages: Any,
    tools: list[JsonDict],
    policy: ContextPolicyConfig,
) -> JsonDict:
    message_tokens = estimate_tokens(messages, policy)
    tool_tokens = estimate_tokens(tools, policy) if tools else 0
    estimated_input_tokens = message_tokens + tool_tokens
    return {
        "estimator": "serialized_chars",
        "chars_per_token": policy.chars_per_token,
        "max_context_tokens": policy.max_context_tokens,
        "reserve_output_tokens": policy.reserve_output_tokens,
        "safety_margin_tokens": policy.safety_margin_tokens,
        "compaction_trigger_tokens": policy.compaction_trigger_tokens,
        "compaction_target_tokens": policy.compaction_target_tokens,
        "available_input_tokens": policy.available_input_tokens,
        "estimated_message_tokens": message_tokens,
        "estimated_tool_tokens": tool_tokens,
        "estimated_input_tokens": estimated_input_tokens,
        "remaining_input_tokens": policy.available_input_tokens - estimated_input_tokens,
        "compaction_recommended": (
            estimated_input_tokens >= policy.compaction_trigger_tokens
        ),
        "within_budget": estimated_input_tokens <= policy.available_input_tokens,
    }
