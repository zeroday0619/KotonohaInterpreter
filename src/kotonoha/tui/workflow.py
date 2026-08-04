"""Sequential workflow for the integrated terminal interface."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import Settings, load_settings
from ..logging_setup import setup_logging
from .app import KotonohaApp
from .config_app import ConfigApp
from .history_app import HistoryApp
from .license_app import LicenseApp
from .menu_app import TuiMenuApp
from .tools_app import ToolsApp


async def run_unified_tui(
    config_path: Path | None,
    build_orchestrator: Callable[[Settings], Any],
) -> None:
    """Run the menu and selected applications on one event loop."""
    while True:
        settings = load_settings(config_path)
        selection = await TuiMenuApp(settings).run_async()
        if selection == "interpreter":
            setup_logging(
                settings.logging.level,
                settings.resolve(settings.logging.log_path),
                settings.logging.console,
                "orch",
                terminal_interface=True,
            )
            await KotonohaApp(build_orchestrator(settings)).run_async()
        elif selection == "configuration":
            await ConfigApp(config_path=config_path).run_async()
        elif selection == "history":
            await HistoryApp(config_path=config_path).run_async()
        elif selection == "tools":
            await ToolsApp(config_path=config_path).run_async()
        elif selection == "license":
            await LicenseApp().run_async()
        else:
            return
