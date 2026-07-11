from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path

from rich.table import Table

from agent import JsonStateStore
from agent.runtime.console import console, trace_text
from agent.runtime.interfaces import StarSession
from agent.runtime.interfaces.star_model import DEFAULT_MODEL_ID
from agent.runtime.kernel.runtime import AgentRuntime
from agent.runtime.perception_systems import PerceptionSystem

from agent_ling.app import create_agent_runtime
from agent_ling.config import load_agent_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the agent_ling application on Star Protocol."
    )
    parser.add_argument(
        "--agent-id", default="agent_ling", help="Agent/client id on Star Protocol."
    )
    parser.add_argument(
        "--hub-url", default="ws://localhost:8000", help="Star Protocol hub URL."
    )
    parser.add_argument(
        "--env-id", default="ashfall-haven-7", help="Environment id to join."
    )
    parser.add_argument(
        "--state-dir", default="./.agent_state", help="State/checkpoint directory."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Model id. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--model-config", default=None, help="Optional MengLong config path."
    )
    parser.add_argument(
        "--agent-config",
        default=None,
        help="Optional agent_ling app config TOML path.",
    )
    parser.add_argument(
        "--join-timeout", type=float, default=30.0, help="Seconds to wait for env join."
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=2.0,
        help="Join retry interval in seconds.",
    )
    parser.add_argument(
        "--no-auto-rejoin", action="store_true", help="Disable Star env auto-rejoin."
    )
    parser.add_argument(
        "--monitorable",
        action="store_true",
        help="Enable Star Protocol monitorable mode.",
    )
    parser.add_argument("--no-trace", action="store_true", help="Hide runtime trace.")
    parser.add_argument(
        "--initial-message",
        default=None,
        help="Optional local user.message submitted after startup.",
    )
    parser.add_argument(
        "--startup-objective",
        default=None,
        help="Autonomous objective submitted after startup and tool discovery. Defaults to app config.",
    )
    parser.add_argument(
        "--no-startup-objective",
        action="store_true",
        help="Do not submit the configured autonomous startup objective.",
    )
    parser.add_argument(
        "--tool-discovery-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for Star Protocol tool specifications before startup objective.",
    )
    return parser


def render_action_specs(runtime: AgentRuntime) -> None:
    specs = list(runtime.registry.list_specs())
    table = Table(title="Available Actions", show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("Mode", style="magenta")
    table.add_column("Source", style="green")
    table.add_column("Target", style="white")
    table.add_column("Description", style="white")
    for spec in specs:
        table.add_row(
            spec.name,
            spec.mode,
            spec.source,
            spec.target or "-",
            spec.description or "-",
        )
    console.print(table)


async def wait_for_external_actions(runtime: AgentRuntime, *, timeout: float) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if any(
            spec.source == "star_protocol" for spec in runtime.registry.list_specs()
        ):
            return
        await asyncio.sleep(0.2)


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trace = not args.no_trace
    store_root = Path(args.state_dir)
    agent_config = load_agent_config(args.agent_config)
    startup_objective = (
        args.startup_objective
        if args.startup_objective is not None
        else agent_config.star.startup_objective
    )
    perception_system = PerceptionSystem()

    session = StarSession(
        hub_url=args.hub_url,
        client_id=args.agent_id,
        env_id=args.env_id,
        join_timeout=args.join_timeout,
        retry_interval=args.retry_interval,
        auto_rejoin=not args.no_auto_rejoin,
        monitorable=args.monitorable,
        perception_system=perception_system,
    )
    application = create_agent_runtime(
        agent_id=args.agent_id,
        store=JsonStateStore(store_root),
        protocol_interface=session,
        model_id=args.model,
        model_config_path=args.model_config,
        agent_config=agent_config,
        perception_system=perception_system,
        trace=trace,
    )
    runtime = application.runtime

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    trace_text(
        "app",
        "starting agent_ling Star App",
        "\n".join(
            [
                f"agent_id={args.agent_id}",
                f"hub_url={args.hub_url}",
                f"env_id={args.env_id}",
                f"model={args.model}",
                f"agent_config={agent_config.source}",
                f"state_dir={store_root.resolve()}",
                f"trace={trace}",
            ]
        ),
    )

    try:
        await runtime.start()
        await wait_for_external_actions(runtime, timeout=args.tool_discovery_timeout)
        await asyncio.sleep(0.2)
        render_action_specs(runtime)
        trace_text(
            "app",
            "ready",
            "Agent is running.\n"
            'Star user input can be sent as action `user_message content="..."` '
            f"to recipient `{args.agent_id}`.",
        )

        if args.initial_message:
            await runtime.submit_user_message(args.initial_message)

        if startup_objective and not args.no_startup_objective:
            trace_text("app", "startup objective", startup_objective)
            await runtime.submit_user_message(startup_objective)

        await stop_event.wait()
    finally:
        trace_text("app", "stopping", "Stopping agent runtime...")
        await runtime.stop()
        trace_text("app", "stopped", "Agent stopped.")

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
