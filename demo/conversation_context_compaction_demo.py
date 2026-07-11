"""Conversation context compaction demo.

Run with: uv run python demo/conversation_context_compaction_demo.py
"""

from __future__ import annotations

from collections import Counter
from tempfile import TemporaryDirectory

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

from agent.runtime.kernel.generator_runtime import GeneratorRuntime
from agent.runtime.persistence_system import ConversationStore
from agent.runtime.state_systems.context_policy import model_request_usage
from agent_ling.config import load_agent_config


console = Console()


def tiered_history(config: object) -> tuple[list[dict], dict[str, int]]:
    with TemporaryDirectory() as state_dir:
        store = ConversationStore(state_dir)
        for index in range(24):
            turn = store.create_turn(
                agent_id="context-compaction-demo",
                conversation_id="conversation-tier-demo",
                speaker_id="human_demo",
                recipient_id="context-compaction-demo",
                channel="demo",
                source_event_id=f"event_{index:02d}",
                utterance=f"第 {index} 轮原始问题",
            )
            turn.understanding = {
                "semantic_summary": f"第 {index} 轮问题的语义摘要",
                "speech_act": "question",
                "intents": ["了解信息"],
            }
            turn.speech_intent = {
                "content": f"回答第 {index} 轮问题",
            }
            turn.response_text = f"第 {index} 轮原始回答"
            turn.status = "completed"
            store.save_turn(turn)
        history = store.context_turns(
            agent_id="context-compaction-demo",
            conversation_id="conversation-tier-demo",
            limit=getattr(config, "recent_turn_limit"),
            verbatim_limit=getattr(config, "verbatim_turn_limit"),
            compact_limit=getattr(config, "compact_turn_limit"),
        )
    counts = Counter(str(turn.get("history_tier")) for turn in history)
    return history, dict(counts)


def main() -> None:
    config = load_agent_config()
    runtime = GeneratorRuntime(agent_config=config, trace=False)
    policy = config.generator.prompt_for("broca").context_policy
    history, tier_counts = tiered_history(config.conversation)
    turns = [
        {
            "turn_id": f"turn_{index:02d}",
            "speaker_id": "human_demo",
            "utterance": f"历史问题 {index}: " + "问" * 90000,
            "response_text": f"历史回答 {index}: " + "答" * 90000,
        }
        for index in range(12)
    ]
    runtime_context = {
        "agent": {
            "agent_id": "context-compaction-demo",
            "name": "Ling",
            "system_profile": "事件驱动 Agent。",
            "persona_profile": "表达清晰。",
            "behavior_profile": "保留事实并按需检索。",
        },
        "conversation": {
            "stage": "speech",
            "conversation_id": "conversation-compaction-demo",
            "turn_id": "turn_current",
            "speaker_id": "human_demo",
            "incoming_utterance": "请根据当前内容回复。",
            "understanding": {"semantic_summary": "当前问题需要直接回答。"},
            "recent_turns": turns,
            "speech_intent": {
                "kind": "direct_conversation_response",
                "content": "回答当前问题。",
            },
        },
        "expression_state": {},
    }
    tiered_runtime_context = {
        **runtime_context,
        "conversation": {
            **runtime_context["conversation"],
            "recent_turns": history,
        },
    }
    tiered_model_context = runtime.build_context(
        tiered_runtime_context,
        session_id="broca",
    )

    original_model_context = runtime.build_context(runtime_context, session_id="broca")
    before_usage = model_request_usage(
        messages=original_model_context.messages,
        tools=[],
        policy=policy,
    )
    (
        prepared_model_context,
        prepared_runtime_context,
        after_usage,
        compaction,
    ) = runtime._prepare_model_context(
        runtime_context,
        session_id="broca",
        tools=[],
    )

    retained_turns = prepared_runtime_context["conversation"]["recent_turns"]
    retained_ids = [turn["turn_id"] for turn in retained_turns]
    expected_ids = [turn["turn_id"] for turn in turns[-len(retained_turns) :]]

    assert policy.max_context_tokens == 1_000_000
    assert policy.compaction_trigger_tokens == 900_000
    assert policy.compaction_target_tokens == 300_000
    assert before_usage["compaction_recommended"] is True
    assert compaction is not None and compaction["applied"] is True
    assert after_usage["within_budget"] is True
    assert after_usage["estimated_input_tokens"] <= policy.compaction_target_tokens
    assert retained_ids == expected_ids
    assert len(turns) == 12
    assert len(prepared_model_context.messages) == 2 + len(retained_turns) * 2
    assert tier_counts == {"summary": 6, "compact": 12, "verbatim": 6}
    assert history[0]["exchange_summary"]["meaning"] == "第 0 轮问题的语义摘要"
    assert history[-1]["utterance"] == "第 23 轮原始问题"
    assert len(tiered_model_context.messages) == 15

    console.rule("[bold blue]Conversation Context Compaction Demo[/bold blue]")
    console.print(
        Panel(
            JSON.from_data(
                {
                    "configured_window_tokens": policy.max_context_tokens,
                    "compaction_trigger_tokens": policy.compaction_trigger_tokens,
                    "compaction_target_tokens": policy.compaction_target_tokens,
                    "available_input_tokens": policy.available_input_tokens,
                    "before_estimated_input_tokens": before_usage[
                        "estimated_input_tokens"
                    ],
                    "after_estimated_input_tokens": after_usage[
                        "estimated_input_tokens"
                    ],
                }
            ),
            title="Model boundary",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            JSON.from_data(
                {
                    "active_history_turns": len(history),
                    "model_messages_after_layering": len(
                        tiered_model_context.messages
                    ),
                    "tier_counts": tier_counts,
                    "oldest_representation": history[0],
                    "newest_representation": history[-1],
                }
            ),
            title="Recency layers",
            border_style="magenta",
        )
    )
    console.print(
        Panel(
            JSON.from_data(compaction),
            title="Compaction manifest",
            border_style="yellow",
        )
    )
    console.print(
        Panel(
            (
                f"ConversationStore source turns: {len(turns)}\n"
                f"Model request retained turns: {len(retained_turns)}\n"
                "The current turn remains mandatory; persisted history was not modified."
            ),
            title="Result",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
