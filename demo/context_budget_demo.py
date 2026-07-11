"""Context budget demo. Run with: uv run python demo/context_budget_demo.py"""

from __future__ import annotations

from dataclasses import replace

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

from agent.protocols import ActionRun, ActionSpec, AgentEvent, AgentState
from agent.runtime.action_systems.actions import ActionRegistry
from agent.runtime.action_systems.task_system import TaskSystem
from agent.runtime.kernel.generator_runtime import GeneratorRuntime
from agent.runtime.state_systems.context_policy import model_request_usage
from agent.runtime.state_systems.workspace import ContextBuilder
from agent_ling.config import load_agent_config


console = Console()


def main() -> None:
    config = load_agent_config()
    default_policy = config.generator.prompt_for("decision").context_policy
    policy = replace(
        default_policy,
        max_context_tokens=32768,
        reserve_output_tokens=4096,
        safety_margin_tokens=2048,
        compaction_trigger_tokens=24000,
        compaction_target_tokens=24000,
    )
    state = AgentState.new("context-budget-demo")
    task_system = TaskSystem()
    root = task_system.create_task(
        state,
        title="长期环境修复",
        goal="持续推进环境修复并保留完整执行证据",
        purpose="Context budget demonstration root task.",
    )

    for index in range(160):
        state.workspace.add_transcript(
            "user" if index % 2 == 0 else "assistant",
            f"历史对话 {index}: " + "环境状态与修复证据。" * 60,
            event_id=f"transcript_event_{index}",
        )
    for index in range(120):
        state.workspace.note(
            f"历史工作笔记 {index}: " + "中继器、材料、位置与依赖状态。" * 50
        )
    for index in range(45):
        run = ActionRun(
            action_run_id=f"historical_run_{index}",
            agent_id=state.agent_id,
            task_id=root.task_id,
            action_name="environment_observe",
            args={"index": index},
            mode="sync",
            source="star_protocol",
            status="succeeded",
            result={
                "index": index,
                "evidence": "完整工具结果。" * 80,
            },
        )
        state.action_runs[run.action_run_id] = run
    for index in range(30):
        child = task_system.create_task(
            state,
            title=f"历史子任务 {index}",
            goal="完成一个历史步骤",
            purpose="Context budget demonstration child task.",
            parent_task_id=root.task_id,
        )
        task_system.complete_task(
            state,
            child.task_id,
            {"evidence": "完整子任务结果。" * 50},
        )

    registry = ActionRegistry()
    for index in range(60):
        registry.register(
            ActionSpec(
                name=f"environment_capability_{index}",
                description=(
                    f"Star environment capability {index} for observation, repair, and status inspection."
                ),
                source="star_protocol",
                target="demo-environment",
                input_schema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                    "required": ["target"],
                },
            )
        )

    event = AgentEvent.make(
        agent_id=state.agent_id,
        type="runtime.continue",
        source="runtime",
        task_id=root.task_id,
        payload={"reason": "continue environment repair"},
    )
    builder = ContextBuilder(policy)
    runtime_context = builder.build(
        state=state,
        event=event,
        action_specs=registry.list_specs(),
    )
    generator_runtime = GeneratorRuntime(agent_config=config, trace=False)
    model_context = generator_runtime.build_context(
        runtime_context,
        session_id="decision",
    )
    model_tools = generator_runtime.model_tools(runtime_context)
    usage = model_request_usage(
        messages=model_context.messages,
        tools=model_tools,
        policy=policy,
    )

    stored = {
        "transcript_messages": len(state.workspace.transcript),
        "workspace_notes": len(state.workspace.notes),
        "tasks": len(state.tasks),
        "action_runs": len(state.action_runs),
        "registered_actions": len(list(registry.list_specs())),
    }
    sent = {
        "evidence_items": len(runtime_context["evidence"]),
        "visible_related_tasks": len(runtime_context["tasks"]),
        "selected_action_runs": len(runtime_context["focus"]["action_runs"]),
        "model_messages": len(model_context.messages),
        "model_tools": len(model_tools),
    }
    selected_terminal_runs = [
        run
        for run in runtime_context["focus"]["action_runs"]
        if run.get("status") in {"succeeded", "failed", "cancelled"}
    ]
    summary_digest = runtime_context["_summary_request"]["summary"]["source_digest"]

    assert usage["within_budget"] is True, usage
    assert runtime_context["context_selection"]["available_by_reference_count"] > 0
    assert len(model_context.messages) == 3 + len(selected_terminal_runs) * 2
    assert len(model_tools) < stored["registered_actions"]
    assert any(
        tool.get("function", {}).get("name") == "search_workspace"
        for tool in model_tools
    )
    assert runtime_context.get("_summary_request") is not None
    console.rule("[bold blue]Context Budget Demo[/bold blue]")
    console.print(
        Panel(
            JSON.from_data({"stored_source_of_truth": stored, "sent_this_turn": sent}),
            title="Full storage vs selected model request",
            border_style="blue",
        )
    )
    console.print(
        Panel(
            JSON.from_data(runtime_context["context_selection"]),
            title="Public context selection",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            JSON.from_data(usage),
            title="Final model-boundary estimate",
            border_style="green",
        )
    )
    console.print(
        Panel(
            JSON.from_data(runtime_context["workspace"]["selected_refs"]),
            title="Selected references",
            border_style="magenta",
        )
    )
    console.print(
        Panel(
            JSON.from_data(
                {
                    "summary_refresh_requested": True,
                    "source_digest_groups": [
                        summary_digest[index : index + 8]
                        for index in range(0, len(summary_digest), 8)
                    ],
                    "covered_ref_count": runtime_context["_summary_request"]["summary"][
                        "covered_ref_count"
                    ],
                    "estimated_source_tokens": runtime_context["_summary_request"][
                        "summary"
                    ]["estimated_source_tokens"],
                }
            ),
            title="Threshold-triggered ContextBuilderSystem request",
            border_style="yellow",
        )
    )
    console.print(
        Panel(
            (
                "完整 Workspace 未删除；本轮仅发送预算内相关单元。"
                "历史 ActionRun 以完整 assistant(tool_call)/tool(result) 对进入 messages；"
                "其余信息保留引用，可通过 search_workspace/read_task/read_action_run 回读。"
            ),
            title="Demo result",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
