"""Context continuity regression demo.

Run with: uv run python demo/context_continuity_demo.py
"""

from __future__ import annotations

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

from agent.protocols import ActionRun, ActionSpec, AgentEvent, AgentState
from agent.runtime.action_systems.actions import ActionRegistry
from agent.runtime.action_systems.task_system import (
    MULTI_STEP_OBJECTIVE_PURPOSE,
    TaskSystem,
)
from agent.runtime.state_systems.workspace import ContextBuilder
from agent_ling.config import load_agent_config


console = Console()


EXTERNAL_ACTIONS = [
    ("observe_environment", "观察环境整体状态和当前目标。"),
    ("inspect_location", "检查当前位置或指定地点的出口与资源。"),
    ("scan_area", "扫描当前位置并发现隐藏线索。"),
    ("list_tasks", "列出环境任务和完成条件。"),
    ("list_available_activities", "列出当前可执行活动。"),
    ("observe_agents", "观察其他 agent 的公开状态。"),
    ("coordinate_plan", "查看共享任务进度和协作建议。"),
    ("claim_task", "声明负责一个共享子任务。"),
    ("release_task", "释放已声明的共享子任务。"),
    ("share_observation", "共享观察和任务进展。"),
    ("send_signal", "发送协作信号。"),
    ("transfer_resource", "转交资源给另一个 agent。"),
    ("move_to", "移动到相邻地点。"),
    ("collect_resource", "收集当前位置可见资源。"),
    ("analyze_artifact", "分析当前位置的装置。"),
    ("craft_item", "合成工具和物品。"),
    ("repair_relay", "修复继电器。"),
    ("unlock_gate", "解锁门禁。"),
    ("stabilize_core", "稳定核心区并完成最终任务。"),
    ("report_status", "报告完整环境状态和任务进度。"),
]


def main() -> None:
    config = load_agent_config()
    policy = config.generator.prompt_for("decision").context_policy
    state = AgentState.new("context-continuity-demo")
    task_system = TaskSystem()
    root = task_system.create_task(
        state,
        title="持续完成外部环境目标",
        goal="在目标完成前持续选择有效动作",
        purpose=MULTI_STEP_OBJECTIVE_PURPOSE,
    )

    first_result = {
        "location": "workshop",
        "agent_location": "atrium",
        "resources": ["fuse", "copper_wire", "crystal"],
        "task_status": "in_progress",
    }
    for index in range(6):
        run = ActionRun(
            action_run_id=f"inspect_{index}",
            agent_id=state.agent_id,
            task_id=root.task_id,
            action_name="inspect_location",
            args={"location": "workshop"},
            mode="async",
            source="star_protocol",
            status="succeeded",
            result=dict(first_result),
        )
        state.action_runs[run.action_run_id] = run
    for index in range(5):
        run = ActionRun(
            action_run_id=f"collect_{index}",
            agent_id=state.agent_id,
            task_id=root.task_id,
            action_name="collect_resource",
            args={"resource_id": "fuse"},
            mode="async",
            source="star_protocol",
            status="failed",
            error={"message": "当前位置没有可收集资源: fuse"},
        )
        state.action_runs[run.action_run_id] = run

    root.progress = {"percent": 100, "message": "completed"}
    root.result = dict(first_result)
    task_system.reconcile(state)

    registry = ActionRegistry()
    for name, description in EXTERNAL_ACTIONS:
        registry.register(
            ActionSpec(
                name=name,
                description=description,
                input_schema={
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "动作目标；不需要时可留空。",
                        }
                    },
                    "additionalProperties": True,
                },
                source="star_protocol",
                target="demo-environment",
            )
        )

    trigger_run = state.action_runs["inspect_5"]
    event = AgentEvent.make(
        agent_id=state.agent_id,
        type="action.completed",
        source="star_protocol",
        task_id=root.task_id,
        action_run_id=trigger_run.action_run_id,
        payload={
            "action_name": trigger_run.action_name,
            "result": trigger_run.result,
        },
    )
    context = ContextBuilder(policy).build(
        state=state,
        event=event,
        action_specs=registry.list_specs(),
    )
    tool_names = [
        tool["function"]["name"]
        for tool in context["_model_tools"]
    ]
    external_names = context["runtime"]["action_guidance"][
        "candidate_external_action_names"
    ]
    repeated = context["runtime"]["execution_memory"]["repeated_attempts"]

    assert set(external_names) == {name for name, _ in EXTERNAL_ACTIONS}
    assert "move_to" in tool_names
    assert "inspect_location" in tool_names
    assert root.progress.get("percent") is None
    assert root.result is None
    assert any(
        item["action_name"] == "inspect_location" and item["count"] == 6
        for item in repeated
    )
    assert any(
        item["action_name"] == "collect_resource" and item["count"] == 5
        for item in repeated
    )

    console.rule("[bold blue]Context Continuity Regression[/bold blue]")
    console.print(
        Panel(
            JSON.from_data(
                {
                    "registered_tools": len(list(registry.list_specs())),
                    "model_request_tools": len(tool_names),
                    "external_catalog_complete": len(external_names),
                    "move_to_available": "move_to" in tool_names,
                    "completed_action_still_available": "inspect_location" in tool_names,
                }
            ),
            title="Capability boundary",
            border_style="green",
        )
    )
    console.print(
        Panel(
            JSON.from_data(
                {
                    "root_status": root.status,
                    "root_progress": root.progress,
                    "root_result": root.result,
                    "repeated_attempts": repeated,
                }
            ),
            title="Task truth and execution memory",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            "回归通过：完整小型环境动作目录保持可调用；多步 root task 不再继承单个 "
            "action 的 100% 状态；重复尝试进入紧凑工作记忆。",
            title="Demo result",
            border_style="blue",
        )
    )


if __name__ == "__main__":
    main()
