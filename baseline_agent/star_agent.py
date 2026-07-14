from __future__ import annotations

import argparse
import asyncio
import json
import signal
from typing import Any

from rich.console import Console

from agent.runtime.interfaces import StarModel, StarSession
from agent.runtime.interfaces.star_model import DEFAULT_MODEL_ID

from .tool_loop import ToolLoopAgent


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the minimal tool-loop baseline agent on Star Protocol."
    )
    parser.add_argument("--agent-id", default="baseline_agent")
    parser.add_argument("--hub-url", default="ws://localhost:8000")
    parser.add_argument("--env-id", default="ashfall-haven-7")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-config", default=None)
    parser.add_argument(
        "--objective",
        default=None,
        help="Optional objective to enqueue after Star tool discovery.",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=20480)
    parser.add_argument("--join-timeout", type=float, default=30.0)
    parser.add_argument("--retry-interval", type=float, default=2.0)
    parser.add_argument("--tool-discovery-timeout", type=float, default=5.0)
    parser.add_argument("--no-auto-rejoin", action="store_true")
    parser.add_argument("--monitorable", action="store_true")
    parser.add_argument("--no-trace", action="store_true")
    return parser


def _trace(event: str, data: dict[str, Any]) -> None:
    console.print(
        f"[cyan]{event}[/cyan] {json.dumps(data, ensure_ascii=False, default=str)}"
    )


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = StarSession(
        hub_url=args.hub_url,
        client_id=args.agent_id,
        env_id=args.env_id,
        join_timeout=args.join_timeout,
        retry_interval=args.retry_interval,
        auto_rejoin=not args.no_auto_rejoin,
        monitorable=args.monitorable,
    )
    agent = ToolLoopAgent(
        agent_id=args.agent_id,
        model=StarModel(
            default_model_id=args.model,
            config_path=args.model_config,
        ),
        protocol=session,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        trace=None if args.no_trace else _trace,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    serve_task: asyncio.Task[None] | None = None
    try:
        await agent.start()
        try:
            await session.rediscover()
        except (AttributeError, NotImplementedError):
            pass
        discovered = await agent.wait_for_tools(args.tool_discovery_timeout)
        tool_names = [spec.name for spec in session.list_action_specs()]
        console.print(
            f"[green]baseline ready[/green] agent={args.agent_id} env={args.env_id} "
            f"tools={tool_names if discovered else 'none'}"
        )
        if args.objective:
            await agent.submit(args.objective)
        serve_task = asyncio.create_task(agent.serve(), name="baseline-agent-serve")
        await stop_event.wait()
    finally:
        if serve_task is not None:
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass
        await agent.stop()
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
