from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


console = Console()


_CHANNELS: dict[str, tuple[str, str]] = {
    "app": ("app", "blue"),
    "event.bus": ("event bus", "cyan"),
    "runtime.policy": ("runtime policy", "yellow"),
    "agent.event": ("agent internal protocol:event", "bright_cyan"),
    "agent.context": ("agent internal protocol:context", "cyan"),
    "agent.decision": ("agent internal protocol:decision", "magenta"),
    "agent.reply": ("agent output:assistant reply", "green"),
    "model.request": ("model boundary:raw request", "bright_blue"),
    "model.response": ("model boundary:raw response", "yellow"),
    "generator": ("generator runtime", "magenta"),
    "dmn": ("dmn system", "bright_magenta"),
    "conversation.manager": ("conversation system:manager", "bright_cyan"),
    "conversation.wernicke": ("conversation system:wernicke", "bright_blue"),
    "conversation.broca": ("conversation system:broca", "bright_green"),
    "action.executor": ("action system:executor", "green"),
    "task.system": ("task system", "blue"),
    "protocol.star": ("protocol interface:star", "bright_green"),
    "error": ("runtime error", "red"),
}


def _channel(channel: str) -> tuple[str, str]:
    return _CHANNELS.get(channel, (channel, "white"))


def trace_title(channel: str, title: str = "") -> str:
    label, style = _channel(channel)
    suffix = f" {escape(title)}" if title else ""
    return (
        f"[bold {style}]{escape(label)}[/bold {style}] "
        f"[dim]channel={escape(channel)}[/dim]{suffix}"
    )


def trace_rule(channel: str, title: str = "") -> None:
    _, style = _channel(channel)
    console.rule(trace_title(channel, title), style=style)


def trace_line(channel: str, message: str) -> None:
    console.print(f"{trace_title(channel)} {message}")


def trace_text(
    channel: str,
    title: str,
    text: str,
    *,
    subtitle: str | None = None,
) -> None:
    _, style = _channel(channel)
    console.print(
        Panel(
            Text(str(text)),
            title=trace_title(channel, title),
            subtitle=escape(subtitle) if subtitle else None,
            border_style=style,
        )
    )


def trace_json(
    channel: str,
    title: str,
    value: Any,
    *,
    subtitle: str | None = None,
) -> None:
    _, style = _channel(channel)
    console.print(
        Panel(
            Syntax(
                json.dumps(value, ensure_ascii=False, indent=2),
                "json",
                word_wrap=True,
            ),
            title=trace_title(channel, title),
            subtitle=escape(subtitle) if subtitle else None,
            border_style=style,
        )
    )


def trace_event(channel: str, event: Any, *, note: str = "") -> None:
    event_type = str(getattr(event, "type", "unknown"))
    event_id = str(getattr(event, "event_id", "-"))
    source = str(getattr(event, "source", "-"))
    task_id = getattr(event, "task_id", None)
    run_id = getattr(event, "action_run_id", None)
    parts = [event_type, f"id={event_id}", f"source={source}"]
    if task_id:
        parts.append(f"task={task_id}")
    if run_id:
        parts.append(f"run={run_id}")
    if note:
        parts.append(note)
    trace_rule(channel, " ".join(parts))
