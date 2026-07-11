"""Real-model ConversationSystem demo.

Run with: uv run python demo/conversation_demo.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

from agent import ConversationStore, JsonStateStore
from agent_ling.app import create_agent_runtime


console = Console()


async def main() -> None:
    agent_id = f"conversation_demo_{uuid4().hex[:8]}"
    state_root = Path(".agent_state/conversation_demo")
    application = create_agent_runtime(
        agent_id=agent_id,
        store=JsonStateStore(state_root),
        trace=True,
    )
    runtime = application.runtime

    console.rule("[bold blue]ConversationSystem Real Model Demo[/bold blue]")
    try:
        await runtime.start()
        await runtime.submit_user_message("你好，我叫林舟。今天有点累，不过见到你还挺高兴的。")
        await runtime.event_bus.join()
        await runtime.submit_user_message("我刚才说我叫什么？")
        await runtime.event_bus.join()
    finally:
        await runtime.stop()

    conversation_state = ConversationStore(state_root).load_state(agent_id)
    turns = [turn.to_dict() for turn in conversation_state.turns.values()]
    console.print(
        Panel(
            JSON.from_data(
                {
                    "session_count": len(conversation_state.sessions),
                    "turn_count": len(turns),
                    "latest_turns": turns[-2:],
                }
            ),
            title="ConversationStore",
            border_style="cyan",
        )
    )

    latest = turns[-2:]
    assert len(latest) == 2
    assert all(turn["status"] == "completed" for turn in latest)
    assert all(turn.get("understanding") for turn in latest)
    assert all(turn.get("response_text") for turn in latest)
    assert all(turn.get("outbound_utterances") for turn in latest)
    console.print(
        Panel(
            "真实模型已完成两轮 Wernicke → Broca 对话，理解与最终话语分别持久化。",
            title="Demo result",
            border_style="green",
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
