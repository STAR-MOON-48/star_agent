from __future__ import annotations

import json
from typing import Any, Iterable

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agent.protocols import ActionSpec, JsonDict


class RichTraceRenderer:
    """Render baseline lifecycle events as compact, readable Rich components."""

    def __init__(self, console: Console, *, max_json_chars: int = 12_000) -> None:
        self.console = console
        self.max_json_chars = max(1000, max_json_chars)

    def __call__(self, event: str, data: JsonDict) -> None:
        renderer = getattr(self, f"_render_{event.replace('.', '_')}", None)
        if renderer is None:
            self.console.print(
                Text(event, style="dim cyan"),
                self._json(data),
            )
            return
        renderer(data)

    def ready(
        self,
        *,
        agent_id: str,
        env_id: str,
        model: str,
        specs: Iterable[ActionSpec],
    ) -> None:
        spec_list = list(specs)
        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold cyan")
        overview.add_column()
        overview.add_row("Agent", agent_id)
        overview.add_row("Environment", env_id)
        overview.add_row("Model", model)
        overview.add_row("Tools", str(len(spec_list)))
        self.console.print(Panel(overview, title="Baseline Ready", border_style="green"))

        if not spec_list:
            self.console.print("[yellow]No Star tools discovered.[/yellow]")
            return
        tools = Table(title="Star Protocol Tools", header_style="bold magenta")
        tools.add_column("Name", style="cyan", no_wrap=True)
        tools.add_column("Target", style="green")
        tools.add_column("Timeout", justify="right")
        tools.add_column("Description")
        for spec in spec_list:
            tools.add_row(
                spec.name,
                spec.target or "-",
                f"{spec.timeout_ms / 1000:g}s",
                spec.description or "-",
            )
        self.console.print(tools)

    def _render_request_started(self, data: JsonDict) -> None:
        self.console.rule("[bold cyan]New Baseline Request")
        self.console.print(
            Panel(
                Text(str(data.get("objective") or "")),
                title=str(data.get("request_id") or "objective"),
                border_style="cyan",
            )
        )

    def _render_context_usage(self, data: JsonDict) -> None:
        estimated = int(data.get("estimated_input_tokens") or 0)
        available = max(1, int(data.get("available_input_tokens") or 1))
        trigger = int(data.get("trigger_tokens") or 0)
        ratio = min(1.0, estimated / available)
        color = "green" if estimated < trigger else "yellow"
        if estimated > available:
            color = "red"
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            ProgressBar(total=1.0, completed=ratio, width=None, style=color),
            Text(
                f"{estimated:,} / {available:,} tokens  ({ratio:.1%})",
                style=color,
            ),
        )
        detail = Text(
            "step={step}  remaining={remaining:,}  trigger={trigger:,}  "
            "verbatim_rounds={rounds}  compactions={compactions}".format(
                step=data.get("step"),
                remaining=int(data.get("remaining_input_tokens") or 0),
                trigger=trigger,
                rounds=data.get("verbatim_rounds"),
                compactions=data.get("compaction_count"),
            ),
            style="dim",
        )
        self.console.print(
            Panel(Group(grid, detail), title="Context Budget", border_style=color)
        )

    def _render_context_compaction_started(self, data: JsonDict) -> None:
        body = Table.grid(padding=(0, 2))
        body.add_column(style="bold yellow")
        body.add_column()
        body.add_row(
            "Before",
            f"{int(data.get('before_estimated_input_tokens') or 0):,} tokens",
        )
        body.add_row("Archive", f"{data.get('archived_rounds')} old rounds")
        body.add_row(
            "Protect",
            f"{data.get('retained_verbatim_rounds')} newest rounds verbatim",
        )
        body.add_row("Target", f"{int(data.get('target_tokens') or 0):,} tokens")
        self.console.print(
            Panel(body, title="Context Compaction Started", border_style="yellow")
        )

    def _render_context_compaction_completed(self, data: JsonDict) -> None:
        before = int(data.get("before_estimated_input_tokens") or 0)
        after = int(data.get("after_estimated_input_tokens") or 0)
        ratio = float(data.get("compression_ratio") or 0.0)
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold green")
        table.add_column()
        table.add_row("Tokens", f"{before:,} → {after:,}")
        table.add_row("Reduced", f"{ratio:.1%}")
        table.add_row(
            "Summary",
            f"{int(data.get('summary_estimated_tokens') or 0):,} tokens "
            f"({data.get('summary_source')})",
        )
        table.add_row(
            "History",
            f"archived {data.get('archived_rounds')}, "
            f"kept {data.get('retained_verbatim_rounds')} verbatim",
        )
        renderables: list[Any] = [table]
        preview = str(data.get("summary_preview") or "").strip()
        if preview:
            renderables.append(
                Panel(Text(preview), title="Rolling Summary Preview", border_style="dim")
            )
        self.console.print(
            Panel(
                Group(*renderables),
                title="Context Compacted",
                border_style="green",
            )
        )

    def _render_context_compaction_summary_failed(self, data: JsonDict) -> None:
        self.console.print(
            Panel(
                Text(str(data.get("error") or "unknown summary error")),
                title="Summary Model Failed — Using Deterministic Fallback",
                border_style="red",
            )
        )

    def _render_model_response(self, data: JsonDict) -> None:
        calls = data.get("tool_calls")
        calls = calls if isinstance(calls, list) else []
        renderables: list[Any] = []
        text = str(data.get("text") or "").strip()
        if text:
            renderables.append(Panel(Text(text), title="Assistant Text"))
        if calls:
            table = Table(header_style="bold magenta", expand=True)
            table.add_column("#", justify="right", width=3)
            table.add_column("Tool", style="cyan", no_wrap=True)
            table.add_column("Arguments")
            for index, call in enumerate(calls, start=1):
                call = call if isinstance(call, dict) else {"value": call}
                table.add_row(
                    str(index),
                    str(call.get("name") or "unknown"),
                    self._json(call.get("arguments") or {}),
                )
            renderables.append(table)
        if not renderables:
            renderables.append(Text("Empty model response", style="yellow"))
        self.console.print(
            Panel(
                Group(*renderables),
                title=f"Model Response · Step {data.get('step')}",
                border_style="magenta",
            )
        )

    def _render_tool_started(self, data: JsonDict) -> None:
        self.console.print(
            Panel(
                self._json(data.get("arguments") or {}),
                title=f"Tool → {data.get('name')}",
                subtitle=str(data.get("action_run_id") or ""),
                border_style="blue",
            )
        )

    def _render_tool_completed(self, data: JsonDict) -> None:
        content = {
            key: value
            for key, value in data.items()
            if key not in {"action_run_id", "name"}
        }
        self.console.print(
            Panel(
                self._json(content),
                title=f"Tool ✓ {data.get('name')}",
                subtitle=str(data.get("action_run_id") or ""),
                border_style="green",
            )
        )

    def _render_tool_failed(self, data: JsonDict) -> None:
        self.console.print(
            Panel(
                self._json(data),
                title=f"Tool ✗ {data.get('name')}",
                border_style="red",
            )
        )

    def _render_request_completed(self, data: JsonDict) -> None:
        self.console.print(
            Panel(
                Text(str(data.get("answer") or "")),
                title=f"Final Answer · {data.get('steps')} steps",
                border_style="bold green",
            )
        )

    def _render_request_max_steps(self, data: JsonDict) -> None:
        self.console.print(
            Panel(
                Text(f"Reached {data.get('max_steps')} tool-loop steps."),
                title="Stopped",
                border_style="yellow",
            )
        )

    def _render_request_failed(self, data: JsonDict) -> None:
        self.console.print(
            Panel(
                Text(str(data.get("error") or "unknown error")),
                title="Request Failed",
                border_style="red",
            )
        )

    def _json(self, value: Any) -> Any:
        serialized = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        syntax = Syntax(
            serialized[: self.max_json_chars],
            "json",
            word_wrap=True,
            background_color="default",
        )
        if len(serialized) <= self.max_json_chars:
            return syntax
        return Group(
            syntax,
            Text(
                f"… {len(serialized) - self.max_json_chars:,} display characters "
                "hidden; full value remains in model context.",
                style="dim yellow",
            ),
        )
