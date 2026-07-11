from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2] / "web_ui"
    if not project_dir.is_dir():
        raise SystemExit(f"Web UI project not found: {project_dir}")
    if shutil.which("npm") is None:
        raise SystemExit("Node.js 20+ and npm are required to run the Web UI.")
    if not (project_dir / "node_modules").is_dir():
        raise SystemExit(
            f"Web UI dependencies are not installed. Run: cd {project_dir} && npm install"
        )
    command = ["npm", "run", "start", "--", *sys.argv[1:]]
    environment = _environment_with_mimo_credentials()
    try:
        completed = subprocess.run(
            command,
            cwd=project_dir,
            check=False,
            env=environment,
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    raise SystemExit(completed.returncode)


def _environment_with_mimo_credentials() -> dict[str, str]:
    environment = dict(os.environ)
    if environment.get("MIMO_API_KEY") or environment.get("XIAOMI_API_KEY"):
        return environment
    try:
        from menglong.utils.config.config_loader import load_config

        config = load_config(environment.get("MIMO_CONFIG_PATH"))
        provider = config.providers.get("xiaomi")
        if provider and provider.api_key:
            environment["XIAOMI_API_KEY"] = str(provider.api_key)
        if (
            provider
            and provider.base_url
            and not environment.get("MIMO_BASE_URL")
            and not environment.get("XIAOMI_BASE_URL")
        ):
            environment["XIAOMI_BASE_URL"] = str(provider.base_url)
    except Exception:
        # The Web server will report voice as unavailable without exposing config details.
        pass
    return environment


if __name__ == "__main__":
    main()
