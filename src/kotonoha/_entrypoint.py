"""Load process environment files before constructing the command-line interface."""

from __future__ import annotations

from kotonoha._env import load_env_file


def main() -> None:
    """Load `.env`, then delegate to the Typer application."""
    load_env_file()

    # Typer localizes help while the module is imported, so the environment must
    # be complete before importing the application rather than before invoking it.
    from kotonoha._cli import main as command_line_main

    command_line_main()
