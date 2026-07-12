from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agent.runtime.persistence_system import JsonStateStore

from ..app import create_refactored_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with Agent Ling Refactor.")
    parser.add_argument("--agent-id", default="agent_ling_refactor")
    parser.add_argument("--state-dir", default="./.agent_state_refactor")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--agent-config", default=None)
    parser.add_argument("--no-trace", action="store_true")
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    async def show_reply(content: str) -> None:
        print(f"Ling> {content}")

    application = create_refactored_runtime(
        agent_id=args.agent_id,
        store=JsonStateStore(Path(args.state_dir)),
        model_id=args.model,
        model_config_path=args.model_config,
        settings_path=args.agent_config,
        on_reply=show_reply,
        trace=not args.no_trace,
    )
    await application.runtime.start()
    print("输入消息；/quit 退出。")
    try:
        while True:
            content = await asyncio.to_thread(input, "You> ")
            if content.strip() in {"/quit", "/exit"}:
                break
            if content.strip():
                await application.runtime.submit_user_message(content)
    finally:
        await application.runtime.stop()
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
