"""Command-line entrypoint for the Textual client."""

from __future__ import annotations

import argparse
from typing import Optional

from src.client.tui.textual_app import TextualClientApp
from src.core_engine.api import DirectRuntime


def run_textual_client(
    config_path: Optional[str] = None,
    *,
    context_id: str = "tui_default",
) -> None:
    """Run the Textual client against a direct runtime."""

    runtime = DirectRuntime.from_config(config_path)
    app = TextualClientApp(runtime, context_id=context_id)
    app.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuro-Like Textual terminal client.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json. Defaults to the project-level config.json.",
    )
    parser.add_argument(
        "--context-id",
        type=str,
        default="tui_default",
        help="Context id used for the terminal session.",
    )
    args = parser.parse_args()
    run_textual_client(args.config, context_id=args.context_id)


if __name__ == "__main__":
    main()
