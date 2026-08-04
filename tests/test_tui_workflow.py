"""Integrated TUI sequencing and configuration reload behavior."""

from __future__ import annotations

from pathlib import Path

from kotonoha.config import load_settings
from kotonoha.tui import workflow


async def test_unified_workflow_returns_to_menu_and_reloads_settings(monkeypatch) -> None:
    selections = iter(("configuration", "tools", "license", "interpreter", None))
    events: list[str] = []
    settings = load_settings()
    load_count = 0

    def load(_config_path: Path | None):
        nonlocal load_count
        load_count += 1
        return settings

    class MenuApplication:
        def __init__(self, _settings) -> None:
            pass

        async def run_async(self):
            return next(selections)

    class ConfigApplication:
        def __init__(
            self,
            config_path: Path | None,
            local_path: Path,
            settings,
            overrides: dict,
        ) -> None:
            assert config_path == Path("device.yaml")
            assert local_path.name == "local.yaml"
            assert settings is not None
            assert isinstance(overrides, dict)

        async def run_async(self) -> None:
            events.append("configuration")

    class InterpreterApplication:
        def __init__(self, orchestrator) -> None:
            assert orchestrator == "orchestrator"

        async def run_async(self) -> None:
            events.append("interpreter")

    class ToolsApplication:
        def __init__(self, config_path: Path | None) -> None:
            assert config_path == Path("device.yaml")

        async def run_async(self) -> None:
            events.append("tools")

    class LicenseApplication:
        async def run_async(self) -> None:
            events.append("license")

    monkeypatch.setattr(workflow, "load_settings", load)
    monkeypatch.setattr(workflow, "TuiMenuApp", MenuApplication)
    monkeypatch.setattr(workflow, "ConfigApp", ConfigApplication)
    monkeypatch.setattr(workflow, "KotonohaApp", InterpreterApplication)
    monkeypatch.setattr(workflow, "ToolsApp", ToolsApplication)
    monkeypatch.setattr(workflow, "LicenseApp", LicenseApplication)
    monkeypatch.setattr(
        workflow,
        "setup_logging",
        lambda *arguments, **keyword_arguments: None,
    )

    await workflow.run_unified_tui(
        Path("device.yaml"),
        lambda active_settings: (
            "orchestrator" if active_settings is settings else "unexpected"
        ),
    )

    assert events == ["configuration", "tools", "license", "interpreter"]
    assert load_count == 5
