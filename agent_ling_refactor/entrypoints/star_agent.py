from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path

from agent.runtime.interfaces import StarSession
from agent.runtime.perception_systems import PerceptionSystem
from agent.runtime.persistence_system import JsonStateStore

from ..app import create_refactored_runtime
from ..settings import load_refactor_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start Agent Ling Refactor on Star Protocol.")
    parser.add_argument("--agent-id", default="agent_ling_refactor")
    parser.add_argument("--hub-url", default="ws://localhost:8000")
    parser.add_argument("--env-id", default="ashfall-haven-7")
    parser.add_argument("--state-dir", default="./.agent_state_refactor")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--agent-config", default=None)
    parser.add_argument("--join-timeout", type=float, default=30)
    parser.add_argument("--retry-interval", type=float, default=2)
    parser.add_argument("--no-auto-rejoin", action="store_true")
    parser.add_argument("--monitorable", action="store_true")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--initial-message", default=None)
    parser.add_argument("--startup-objective", default=None)
    parser.add_argument("--no-startup-objective", action="store_true")
    parser.add_argument("--tool-discovery-timeout", type=float, default=5)
    return parser


async def _wait_for_tools(runtime: object, timeout: float) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        registry = getattr(runtime, "registry")
        if any(spec.source == "star_protocol" for spec in registry.list_specs()):
            return
        await asyncio.sleep(0.2)


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_refactor_settings(args.agent_config)
    perception = PerceptionSystem()
    session = StarSession(
        hub_url=args.hub_url,
        client_id=args.agent_id,
        env_id=args.env_id,
        join_timeout=args.join_timeout,
        retry_interval=args.retry_interval,
        auto_rejoin=not args.no_auto_rejoin,
        monitorable=args.monitorable,
        perception_system=perception,
    )
    application = create_refactored_runtime(
        agent_id=args.agent_id,
        store=JsonStateStore(Path(args.state_dir)),
        model_id=args.model,
        model_config_path=args.model_config,
        settings=settings,
        protocol=session,
        trace=not args.no_trace,
    )
    runtime = application.runtime
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass
    try:
        await runtime.start()
        await _wait_for_tools(runtime, args.tool_discovery_timeout)
        if args.initial_message:
            await runtime.submit_user_message(args.initial_message)
        startup = (
            args.startup_objective
            if args.startup_objective is not None
            else settings.star.startup_objective
        )
        if startup and not args.no_startup_objective:
            await runtime.submit_objective(startup)
        await stop_event.wait()
    finally:
        await runtime.stop()
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
