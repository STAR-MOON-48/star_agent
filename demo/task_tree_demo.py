"""Task tree scheduling demo. Run with: uv run python demo/task_tree_demo.py"""

from __future__ import annotations

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.tree import Tree

from agent.protocols import AgentState, AgentTask
from agent.runtime.action_systems.task_system import (
    MULTI_STEP_OBJECTIVE_PURPOSE,
    TaskSystem,
)


console = Console()


def task_label(task: AgentTask) -> str:
    scheduling = task.scheduling
    return (
        f"[bold]{task.title}[/bold]  "
        f"id=[cyan]{task.task_id}[/cyan]  "
        f"status=[yellow]{task.status}[/yellow]  "
        f"classification=[magenta]{scheduling.get('classification')}[/magenta]  "
        f"can_run={scheduling.get('can_run')}  "
        f"can_complete={scheduling.get('can_complete')}"
    )


def add_children(tree: Tree, state: AgentState, task: AgentTask) -> None:
    for child_id in task.child_task_ids:
        child = state.tasks[child_id]
        branch = tree.add(task_label(child))
        add_children(branch, state, child)


def show_state(title: str, state: AgentState, task_system: TaskSystem) -> None:
    task_system.reconcile(state)
    roots = [task for task in state.tasks.values() if not task.parent_task_id]
    console.rule(f"[bold blue]{title}[/bold blue]")
    for root in roots:
        tree = Tree(task_label(root))
        add_children(tree, state, root)
        console.print(tree)

    scheduler_view = []
    for task in state.tasks.values():
        scheduler_view.append(
            {
                "task_id": task.task_id,
                "dependencies": task.dependencies,
                "pending_dependencies": task.scheduling.get(
                    "pending_dependency_ids",
                    [],
                ),
                "completion_blockers": task.scheduling.get(
                    "completion_blockers",
                    [],
                ),
            }
        )
    console.print(
        Panel(
            JSON.from_data(scheduler_view),
            title="Scheduler view",
            border_style="blue",
        )
    )
    console.print(
        f"current_task_id=[bold cyan]{state.workspace.current_task_id}[/bold cyan]  "
        f"next_runnable_task_id=[bold green]"
        f"{task_system.next_runnable_task_id(state)}[/bold green]"
    )


def show_completion(title: str, result: dict[str, object]) -> None:
    console.print(
        Panel(
            JSON.from_data(result),
            title=f"[bold]{title}[/bold]",
            border_style="green" if result.get("completed") else "yellow",
        )
    )


def main() -> None:
    state = AgentState.new("task-tree-demo")
    task_system = TaskSystem()

    root = task_system.create_task(
        state,
        title="完成环境修复",
        goal="完成所有必要步骤并确认环境恢复",
        purpose=MULTI_STEP_OBJECTIVE_PURPOSE,
        continuation={"kind": "multi_step_objective"},
    )
    collect = task_system.create_task(
        state,
        title="收集修复材料",
        goal="获得修复所需材料",
        purpose="root task 的执行步骤",
        parent_task_id=root.task_id,
    )
    repair = task_system.create_task(
        state,
        title="执行修复",
        goal="使用材料完成修复",
        purpose="root task 的执行步骤",
        parent_task_id=root.task_id,
        dependencies=[collect.task_id],
    )

    show_state("1. 创建任务树", state, task_system)
    assert collect.status == "runnable"
    assert repair.status == "waiting"
    assert root.status == "waiting"
    assert task_system.next_runnable_task_id(state) == collect.task_id

    early_completion = task_system.complete_task(
        state,
        root.task_id,
        {"summary": "尝试提前结束 root"},
    )
    show_completion("2. root 提前完成请求", early_completion)
    show_state("root 保持未完成，调度器仍选择可运行叶子", state, task_system)
    assert early_completion["deferred"] is True
    assert root.status != "completed"

    collect_completion = task_system.complete_task(
        state,
        collect.task_id,
        {"materials": ["fuse", "copper_wire", "crystal"]},
    )
    show_completion("3. 完成依赖任务", collect_completion)
    show_state("依赖解除后，下一个叶子变为 runnable", state, task_system)
    assert repair.status == "runnable"
    assert task_system.next_runnable_task_id(state) == repair.task_id

    repair_completion = task_system.complete_task(
        state,
        repair.task_id,
        {"repair_verified": True},
    )
    show_completion("4. 完成最终执行步骤", repair_completion)
    show_state("所有子任务完成，root 回到 runnable 等待收口", state, task_system)
    assert root.status == "runnable"
    assert root.scheduling["can_complete"] is True
    assert task_system.next_runnable_task_id(state) == root.task_id

    root_completion = task_system.complete_task(
        state,
        root.task_id,
        {"summary": "所有必要步骤和验证均已完成"},
    )
    show_completion("5. root 完成", root_completion)
    show_state("最终状态", state, task_system)
    assert root.status == "completed"
    assert state.workspace.current_task_id is None
    assert task_system.next_runnable_task_id(state) is None

    console.print(
        Panel(
            "任务树已收敛：root 只在子树、依赖、等待和 action run 均满足完成条件后进入 completed。",
            title="Demo result",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
