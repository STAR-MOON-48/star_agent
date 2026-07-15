from __future__ import annotations

import argparse
import asyncio
import signal

from rich.console import Console

from agent.runtime.interfaces import StarModel, StarSession
from agent.runtime.interfaces.star_model import DEFAULT_MODEL_ID

from .rich_output import RichTraceRenderer
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
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=1_000_000,
        help="Model context window used by the local budget estimator.",
    )
    parser.add_argument(
        "--context-safety-margin-tokens",
        type=int,
        default=8192,
        help="Input tokens kept unused as an overflow safety margin.",
    )
    parser.add_argument(
        "--compaction-trigger-ratio",
        type=float,
        default=0.85,
        help="Compact when estimated input reaches this fraction of its budget.",
    )
    parser.add_argument(
        "--compaction-target-ratio",
        type=float,
        default=0.35,
        help="Target input fraction after old rounds are summarized.",
    )
    parser.add_argument(
        "--keep-recent-rounds",
        type=int,
        default=4,
        help="Newest assistant/tool rounds kept verbatim during compaction.",
    )
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=4096,
        help="Hard output limit for each rolling-summary model call.",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=2.0,
        help="Conservative serialized-character token estimate.",
    )
    parser.add_argument("--join-timeout", type=float, default=30.0)
    parser.add_argument("--retry-interval", type=float, default=2.0)
    parser.add_argument("--tool-discovery-timeout", type=float, default=5.0)
    parser.add_argument("--no-auto-rejoin", action="store_true")
    parser.add_argument("--monitorable", action="store_true")
    parser.add_argument("--no-trace", action="store_true")
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    renderer = RichTraceRenderer(console)
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
        context_window_tokens=args.context_window_tokens,
        context_safety_margin_tokens=args.context_safety_margin_tokens,
        compaction_trigger_ratio=args.compaction_trigger_ratio,
        compaction_target_ratio=args.compaction_target_ratio,
        keep_recent_rounds=args.keep_recent_rounds,
        summary_max_tokens=args.summary_max_tokens,
        chars_per_token=args.chars_per_token,
        trace=None if args.no_trace else renderer,
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
        await agent.wait_for_tools(args.tool_discovery_timeout)
        renderer.ready(
            agent_id=args.agent_id,
            env_id=args.env_id,
            model=args.model,
            specs=session.list_action_specs(),
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
