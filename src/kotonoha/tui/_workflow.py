"""Sequential workflow for the integrated terminal interface."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kotonoha._config import Settings, load_settings, local_config_path, read_yaml
from kotonoha._logging_setup import setup_logging
from kotonoha._shmring import prepare_shared_memory_tracking
from kotonoha.tui._app import KotonohaApp
from kotonoha.tui._config_app import ConfigApp
from kotonoha.tui._history_app import HistoryApp
from kotonoha.tui._license_app import LicenseApp
from kotonoha.tui._menu_app import TuiMenuApp
from kotonoha.tui._tools_app import ToolsApp


def _read_optional_yaml(
    path: Path,
    /,
) -> dict:
    return read_yaml(path) if path.exists() else {}


async def run_unified_tui(
    config_path: Path | None,
    /,
    build_orchestrator: Callable[[Settings], Any],
) -> None:
    """Run the menu and selected applications on one event loop."""
    # The menu is itself a terminal application. Start the process tracker before
    # Textual can replace standard-error file descriptors used by multiprocessing.
    prepare_shared_memory_tracking()
    while True:
        settings = await asyncio.to_thread(load_settings, config_path)
        selection = await TuiMenuApp(settings).run_async()
        if selection == "interpreter":
            setup_logging(
                settings.logging.level,
                settings.resolve(settings.logging.log_path),
                settings.logging.console,
                "orchestrator",
                terminal_interface=True,
                maximum_bytes=settings.logging.max_bytes,
                backup_count=settings.logging.backup_count,
            )
            orchestrator = await asyncio.to_thread(build_orchestrator, settings)
            await KotonohaApp(orchestrator).run_async()
        elif selection == "configuration":
            local_path = local_config_path()
            overrides = await asyncio.to_thread(_read_optional_yaml, local_path)
            await ConfigApp(
                config_path=config_path,
                local_path=local_path,
                settings=settings,
                overrides=overrides,
            ).run_async()
        elif selection == "history":
            await HistoryApp(config_path=config_path, settings=settings).run_async()
        elif selection == "tools":
            await ToolsApp(config_path=config_path).run_async()
        elif selection == "license":
            await LicenseApp().run_async()
        else:
            return
