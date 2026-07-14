from __future__ import annotations

import argparse
from pathlib import Path

from ..control import ControlInbox


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inject one natural-language internal directive into a running Agent Ling.",
    )
    parser.add_argument("text", help="Natural-language thought, direction, or private note.")
    parser.add_argument(
        "--target",
        choices=("decision", "dmn", "note"),
        default="decision",
        help="Cognitive destination. Default: decision.",
    )
    parser.add_argument("--agent-id", default="agent_ling_refactor")
    parser.add_argument("--state-dir", default="./.agent_state_refactor")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    inbox = ControlInbox(Path(args.state_dir), args.agent_id)
    directive = inbox.enqueue(target=args.target, text=args.text)
    print(
        f"queued internal directive id={directive.directive_id} "
        f"agent={directive.agent_id} target={directive.target}"
    )


if __name__ == "__main__":
    main()
