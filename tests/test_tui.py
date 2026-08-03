"""Terminal interfaces compose and pick up the active locale.

These exercise the Textual API surface the applications depend on — binding
registration, CSS parsing, widget construction — which unit tests of the underlying
logic do not reach. A Textual upgrade that changes any of it fails here rather than on
the device.
"""

from __future__ import annotations

import sys
import wave

import numpy as np
import pytest
import yaml
from textual.widgets import Button

from kotonoha.clients.config_admin import RemoteConfigSnapshot
from kotonoha.config import load_settings
from kotonoha.config_store import set_path
from kotonoha.i18n import CATALOGS, set_locale
from kotonoha.services.config_admin import REMOTE_EDITABLE_PATHS
from kotonoha.tui import tools_app
from kotonoha.tui.app import KotonohaApp
from kotonoha.tui.config_app import FIELDS, SECTIONS, ConfigApp
from kotonoha.tui.menu_app import TuiMenuApp
from kotonoha.tui.tools_app import OPERATION_FIELDS, OPERATIONS, ToolsApp


@pytest.fixture(autouse=True)
def _reset_locale():
    yield
    set_locale(None)


@pytest.fixture
def wav_path(tmp_path):
    path = tmp_path / "probe.wav"
    sr = 16000
    t = np.arange(int(1.5 * sr)) / sr
    x = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return path


# -- integrated control center ---------------------------------------------
async def test_control_center_composes_with_localized_actions():
    set_locale("ja")
    app = TuiMenuApp(load_settings())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.title == CATALOGS["ja"]["tui.title"]
        assert str(app.query_one("#interpreter", Button).label) == CATALOGS["ja"][
            "tui.menu.interpreter"
        ]
        assert str(app.query_one("#configuration", Button).label) == CATALOGS["ja"][
            "tui.menu.configuration"
        ]
        assert str(app.query_one("#tools", Button).label) == CATALOGS["ja"]["tui.menu.tools"]
        assert [binding.description for binding in app._bindings.shown_keys] == [
            CATALOGS["ja"]["tui.menu.key.interpreter"],
            CATALOGS["ja"]["tui.menu.key.configuration"],
            CATALOGS["ja"]["tui.menu.key.tools"],
            CATALOGS["ja"]["tui.menu.key.quit"],
        ]


async def test_control_center_keyboard_action_selects_interpreter():
    app = TuiMenuApp(load_settings())
    async with app.run_test() as pilot:
        await pilot.press("i")
    assert app.return_value == "interpreter"


async def test_operations_screen_composes_every_cli_operation():
    set_locale("ko")
    app = ToolsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        operation_select = app.query_one("#tool-operation")
        assert len(operation_select._options) == len(OPERATIONS)
        assert app.title == CATALOGS["ko"]["tools.title"]
        assert app.query_one("#field-wav").display
        assert not app.query_one("#field-host").display
        assert [binding.description for binding in app._bindings.shown_keys] == [
            CATALOGS["ko"]["tools.key.run"],
            CATALOGS["ko"]["tools.key.stop"],
            CATALOGS["ko"]["tools.key.clear"],
            CATALOGS["ko"]["tools.key.back"],
        ]
        app._write("styled output", "red")


async def test_operations_screen_switches_command_fields():
    app = ToolsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#tool-operation").value = "serve"
        await pilot.pause()
        assert app.query_one("#field-service").display
        assert app.query_one("#field-host").display
        assert app.query_one("#field-port").display
        assert not app.query_one("#field-wav").display

        app.query_one("#tool-operation").value = "doctor"
        await pilot.pause()
        assert not any(
            app.query_one(f"#field-{field_id}").display
            for fields in OPERATION_FIELDS.values()
            for field_id in fields
        )


async def test_operations_screen_streams_a_child_process(monkeypatch):
    monkeypatch.setattr(
        tools_app,
        "build_tool_command",
        lambda operation, values, config_path: [  # noqa: ARG005
            sys.executable,
            "-c",
            "print('operation output')",
        ],
    )
    app = ToolsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._execute_tool()
        assert app.process is None
        assert "0" in str(app.query_one("#tool-status").render())
        assert not app.query_one("#tool-run", Button).disabled
        assert app.query_one("#tool-stop", Button).disabled


# -- configuration editor ---------------------------------------------------
async def test_config_editor_composes_one_row_per_field(tmp_path):
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app._rows) == len(FIELDS)
        assert {r.spec.path for r in app._rows} == {f.path for f in FIELDS}


async def test_config_editor_titles_follow_the_locale(tmp_path):
    set_locale("ja")
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.title == CATALOGS["ja"]["cfg.title"]
        assert [b.description for b in app._bindings.shown_keys] == [
            CATALOGS["ja"]["cfg.key.save"],
            CATALOGS["ja"]["cfg.key.reload"],
            CATALOGS["ja"]["cfg.key.menu"],
            CATALOGS["ja"]["cfg.key.quit"],
        ]


async def test_config_editor_shows_one_category_at_a_time(tmp_path):
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.current_section == "interface"
        assert app.query_one("#panel-interface").display
        assert not app.query_one("#panel-llm").display

        category_list = app.query_one("#category-list")
        category_list.focus()
        category_list.index = SECTIONS.index("llm")
        await pilot.press("enter")
        await pilot.pause()
        assert app.current_section == "llm"
        assert app.query_one("#panel-llm").display
        assert not app.query_one("#panel-interface").display


async def test_category_switch_preserves_unsaved_input(tmp_path):
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._show_section("remote")
        row = next(r for r in app._rows if r.spec.path == "remote.services.llm")
        row.editor.value = "http://a6000.internal:8003"

        app._show_section("interface")
        app._show_section("remote")
        assert row.editor.value == "http://a6000.internal:8003"


async def test_menu_action_focuses_category_navigation(tmp_path):
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_menu()
        await pilot.pause()
        assert app.focused is app.query_one("#category-list")


async def test_saving_without_edits_writes_nothing(tmp_path):
    target = tmp_path / "local.yaml"
    set_locale("en")
    app = ConfigApp(local_path=target)
    said: list[str] = []
    async with app.run_test() as pilot:
        await pilot.pause()
        app._say = lambda message, style: said.append(message)  # noqa: ARG005
        await app.action_save()
    assert said == [CATALOGS["en"]["cfg.no_changes"]]
    assert not target.exists()


async def test_editing_a_field_persists_it(tmp_path):
    target = tmp_path / "local.yaml"
    set_locale("en")
    app = ConfigApp(local_path=target)
    async with app.run_test() as pilot:
        await pilot.pause()
        row = next(r for r in app._rows if r.spec.path == "perf_mode")
        row.editor.value = "hybrid"
        app._say = lambda message, style: None  # noqa: ARG005
        await app.action_save()
    assert target.exists()
    assert "hybrid" in target.read_text(encoding="utf-8")


async def test_reload_restores_the_stored_values(tmp_path):
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        row = next(r for r in app._rows if r.spec.path == "perf_mode")
        original = row.editor.value
        row.editor.value = "remote"
        await app.action_reload()
        assert row.editor.value == original


class FakeRemoteConfigClient:
    def __init__(self):
        settings = load_settings("config/remote-server.yaml")
        self.config = settings.model_dump(mode="json", exclude={"root"})
        self.overrides: dict = {}
        self.changes: dict = {}

    async def read(self) -> RemoteConfigSnapshot:
        return self.snapshot()

    async def update(self, changes: dict) -> RemoteConfigSnapshot:
        self.changes = changes
        for path, value in changes.items():
            set_path(self.config, path, value)
            set_path(self.overrides, path, value)
        return self.snapshot()

    async def aclose(self) -> None:
        return None

    def snapshot(self) -> RemoteConfigSnapshot:
        return RemoteConfigSnapshot(
            config=self.config,
            editable_paths=sorted(REMOTE_EDITABLE_PATHS),
            overrides=self.overrides,
            path="/app/config/remote-server.local.yaml",
            restart_required=True,
        )


async def test_remote_target_loads_and_saves_through_the_admin_client(tmp_path):
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    remote_client = FakeRemoteConfigClient()
    app.remote_client = remote_client
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#target-select").value = "remote"
        await pilot.pause()
        assert app.target == "remote"
        assert app.settings.llm.profile == "moe"
        assert app.query_one("#category-asr").display
        assert not app.query_one("#category-session").display

        row = next(row for row in app._rows if row.spec.path == "llm.n_ctx")
        row.editor.value = "8192"
        await app.action_save()

        assert remote_client.changes == {"llm.n_ctx": 8192}
        assert app.settings.llm.n_ctx == 8192
        assert not (tmp_path / "local.yaml").exists()


async def test_collection_fields_accept_yaml_flow_values(tmp_path):
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        row = next(row for row in app._rows if row.spec.path == "session.pair")
        row.editor.value = "[ja, zh-TW]"
        await app.action_save()
    written = yaml.safe_load((tmp_path / "local.yaml").read_text(encoding="utf-8"))
    assert written["session"]["pair"] == ["ja", "zh-TW"]


# -- main interface ---------------------------------------------------------
async def test_main_interface_composes_with_localized_labels(wav_path, tmp_path, monkeypatch):
    monkeypatch.setenv("KOTONOHA__FRONTEND__VAD__BACKEND", "energy")
    monkeypatch.setenv("KOTONOHA__SHM__NAME", "kotonoha_test_tui")
    monkeypatch.setenv("KOTONOHA__STORE__PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("KOTONOHA__LOGGING__LOG_PATH", str(tmp_path / "application.jsonl"))
    monkeypatch.setenv("KOTONOHA__LOGGING__TURN_LOG_PATH", str(tmp_path / "turns.jsonl"))
    set_locale("ko")

    from kotonoha.cli import _build
    from kotonoha.config import load_settings
    from kotonoha.logging_setup import (
        get_logger,
        reset_terminal_interface_logs,
        setup_logging,
    )

    settings = load_settings()
    setup_logging(
        settings.logging.level,
        settings.resolve(settings.logging.log_path),
        settings.logging.console,
        "orch",
        terminal_interface=True,
    )
    orch = _build(settings, wav=wav_path)
    app = KotonohaApp(orch)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.title == CATALOGS["ko"]["tui.title"]
        assert app.src._title == CATALOGS["ko"]["tui.pane.source"]
        assert app.tgt._title == CATALOGS["ko"]["tui.pane.target"]
        assert str(app.query_one("#log-title").render()) == CATALOGS["ko"]["tui.panel.logs"]
        get_logger().info("tui.test", readable=True)
        await pilot.pause(0.2)
        log_text = "\n".join(line.text for line in app.log_output.lines)
        assert "tui.test" in log_text
        assert "readable=true" in log_text
        assert [b.description for b in app._bindings.shown_keys] == [
            CATALOGS["ko"]["tui.key.talk"],
            CATALOGS["ko"]["tui.key.mode"],
            CATALOGS["ko"]["tui.key.routing"],
            CATALOGS["ko"]["tui.key.clear"],
            CATALOGS["ko"]["tui.key.quit"],
        ]
    setup_logging()
    reset_terminal_interface_logs()
