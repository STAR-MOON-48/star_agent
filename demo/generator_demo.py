from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import GeneratorRuntime
from agent.protocols import AgentEvent, AgentState
from agent.runtime.action_systems.actions import ActionRegistry
from agent.runtime.console import console, trace_json, trace_rule, trace_text
from agent.runtime.interfaces.star_model import DEFAULT_MODEL_ID
from agent.runtime.kernel.generator_session import parse_generator_decision
from agent.runtime.state_systems.workspace import ContextBuilder
from agent_ling.config import load_agent_config


def runtime_context(content: str, *, agent_config_path: str | None = None) -> dict[str, Any]:
    agent_config = load_agent_config(agent_config_path)
    profile = agent_config.profile.to_agent_profile("agent_demo")
    state = AgentState.new(profile.agent_id)
    state.profile = profile
    event = AgentEvent.make(
        agent_id=profile.agent_id,
        type="conversation.decision.requested",
        source="conversation_system",
        correlation_id="generator_demo_conversation",
        payload={
            "conversation_id": "generator_demo_conversation",
            "turn_id": "generator_demo_turn",
            "speaker_id": "human",
            "recipient_id": profile.agent_id,
            "content": content,
            "understanding": {
                "semantic_summary": content,
                "speech_act": "request",
                "decision_needed": True,
                "response_needed": True,
                "confidence": 1.0,
            },
            "decision_request": "判断需要调用哪些 Agent 能力，并形成必要的表达意图。",
        },
    )
    return ContextBuilder().build(
        state=state,
        event=event,
        action_specs=ActionRegistry().list_specs(),
    )


def print_json(title: str, value: Any, *, channel: str = "agent.context") -> None:
    trace_json(channel, title, value)


def print_context_summary(context: Any) -> None:
    table = Table(title="MengLong Context sent to real model")
    table.add_column("Index", justify="right")
    table.add_column("Role", style="cyan")
    table.add_column("Content Preview", style="white")
    for index, message in enumerate(context.messages):
        content = message.content if isinstance(message.content, str) else str(message.content)
        table.add_row(str(index), str(message.role), content.replace("\n", " "))
    console.print(table)


def print_commands(decision: dict[str, Any]) -> None:
    table = Table(title="Real Model GeneratorDecision Commands", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("Type", style="cyan")
    table.add_column("Payload", style="white")
    for index, command in enumerate(decision.get("commands", []), start=1):
        payload = {k: v for k, v in command.items() if k != "type"}
        table.add_row(str(index), str(command.get("type")), json.dumps(payload, ensure_ascii=False))
    console.print(table)


async def show_real_generator_flow(
    prompt: str,
    model_id: str,
    agent_config_path: str | None,
) -> None:
    agent_config = load_agent_config(agent_config_path)
    gr = GeneratorRuntime(default_model_id=model_id, agent_config=agent_config)
    context = runtime_context(prompt, agent_config_path=agent_config_path)
    public_context = gr.public_context(context)
    model_tools = gr.model_tools(context)
    menglong_context = gr.build_context(context)

    trace_rule("app", "1. Runtime context")
    print_json("Decision trigger", context["decision"]["trigger"], channel="agent.context")
    print_json("Agent internal decision context", public_context, channel="agent.context")
    print_json("Model request tools field", model_tools, channel="model.request")
    print_context_summary(menglong_context)

    trace_rule("app", "2. Real model call")
    trace_text(
        "generator",
        "call path",
        f"GeneratorRuntime -> LLMGeneratorSession -> StarModel -> MengLong Model\nmodel_id={model_id}",
    )
    await gr.start()
    try:
        result = await gr.generate_with_trace(context)
    finally:
        await gr.stop()

    trace_rule("app", "3. Raw model boundary")
    print_json("Model raw request", result.trace.get("model_request", {}), channel="model.request")
    print_json("Model raw response", result.trace.get("model_response", {}), channel="model.response")

    trace_rule("app", "4. Parsed generator decision")
    print_json("Parsed GeneratorDecision", result.decision, channel="agent.decision")
    print_commands(result.decision)


async def show_real_chat_flow(
    prompt: str,
    model_id: str,
    agent_config_path: str | None,
) -> None:
    agent_config = load_agent_config(agent_config_path)
    gr = GeneratorRuntime(default_model_id=model_id, agent_config=agent_config)
    context = gr.build_context(runtime_context(prompt, agent_config_path=agent_config_path))

    trace_rule("app", "5. Direct model request through GeneratorRuntime")
    await gr.start()
    try:
        result = await gr.chat(context, model=model_id)
    finally:
        await gr.stop()

    trace_text("model.response", f"raw text response ({result.model or model_id})", result.text or "")
    try:
        parsed = parse_generator_decision(result.text or "")
    except ValueError as exc:
        trace_text("generator", "raw response parse note", str(exc))
    else:
        print_json("Parsed raw response", parsed, channel="agent.decision")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real-model generator demo.")
    parser.add_argument(
        "--prompt",
        default="帮我分析这个大型项目，完成后给我一个报告。",
        help="User message to put into the runtime event.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"MengLong model id. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--agent-config",
        default=None,
        help="Optional agent prompt/profile config TOML path.",
    )
    parser.add_argument(
        "--raw-chat",
        action="store_true",
        help="Also show a direct GeneratorRuntime.chat call and raw model response.",
    )
    args = parser.parse_args()

    trace_rule("app", "Real Generator Runtime Demo")
    await show_real_generator_flow(args.prompt, args.model, args.agent_config)
    if args.raw_chat:
        await show_real_chat_flow(args.prompt, args.model, args.agent_config)
    trace_rule("app", "Demo finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
