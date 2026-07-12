"""Web scene entrypoint shared with the Star Protocol implementation.

The browser scene is already runtime-independent: it talks to the agent through
Star Hub.  Reusing it avoids forking frontend code while preserving the same UI
scope for the refactored agent.
"""

from agent_ling.entrypoints.web_ui import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
