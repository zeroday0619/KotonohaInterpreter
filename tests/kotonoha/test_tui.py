"""Terminal interfaces compose and pick up the active locale.

These exercise the Textual API surface the applications depend on — binding
registration, CSS parsing, widget construction — which unit tests of the underlying
logic do not reach. A Textual upgrade that changes any of it fails here rather than on
the device.
"""

from __future__ import annotations

import sys
import wave
from typing import Any, ClassVar

import numpy as np
import pytest
import yaml
from textual.widgets import Button, Select

from kotonoha._config import load_settings
from kotonoha._config_store import set_path
from kotonoha._i18n import set_locale, translate_to
from kotonoha.audio._devices import AudioDevice, AudioProbeResult
from kotonoha.clients._config_admin import RemoteConfigSnapshot
from kotonoha.core._events import UiEvent
from kotonoha.services._config_admin import REMOTE_EDITABLE_PATHS
from kotonoha.tui import _config_app as config_app
from kotonoha.tui import _tools_app as tools_app
from kotonoha.tui._app import KotonohaApp
from kotonoha.tui._config_app import FIELDS, SECTIONS, ConfigApp
from kotonoha.tui._license_app import LicenseApp
from kotonoha.tui._menu_app import TuiMenuApp
from kotonoha.tui._tools_app import OPERATION_FIELDS, OPERATIONS, ToolsApp


@pytest.fixture(autouse=True)
def _reset_locale() -> Any:
    yield
    set_locale(None)


@pytest.fixture
def wav_path(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> Any:
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
async def test_control_center_composes_with_localized_actions() -> None:
    set_locale("ja")
    app = TuiMenuApp(load_settings())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.title == translate_to("ja", "Kotonoha Interpreter")
        assert str(app.query_one("#interpreter", Button).label) == translate_to(
            "ja", "Interpreter"
        )
        assert str(app.query_one("#configuration", Button).label) == translate_to(
            "ja", "Configuration"
        )
        assert str(app.query_one("#tools", Button).label) == translate_to("ja", "Operations")
        assert str(app.query_one("#license", Button).label) == translate_to("ja", "License")
        assert [binding.description for binding in app._bindings.shown_keys] == [
            translate_to("ja", "Interpreter"),
            translate_to("ja", "Configuration"),
            translate_to("ja", "Interpretation history"),
            translate_to("ja", "Operations"),
            translate_to("ja", "License"),
            translate_to("ja", "Exit"),
        ]


async def test_control_center_keyboard_action_selects_interpreter() -> None:
    app = TuiMenuApp(load_settings())
    async with app.run_test() as pilot:
        await pilot.press("i")
    assert app.return_value == "interpreter"


async def test_control_center_keyboard_action_selects_license() -> None:
    app = TuiMenuApp(load_settings())
    async with app.run_test() as pilot:
        await pilot.press("l")
    assert app.return_value == "license"


async def test_license_screen_composes_with_localized_tabs() -> None:
    set_locale("zh-TW")
    app = LicenseApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.title == translate_to("zh-TW", "License information")
        assert app.license_text is not None
        assert "MIT License" in app.license_text
        assert app.query_one("#dependency-table").row_count == len(app.dependencies)
        assert [binding.description for binding in app._bindings.shown_keys] == [
            translate_to("zh-TW", "Project"),
            translate_to("zh-TW", "Dependencies"),
            translate_to("zh-TW", "Back"),
        ]

        await pilot.press("d")
        assert app.query_one("#license-tabs").active == "dependencies"


async def test_operations_screen_composes_every_cli_operation() -> None:
    set_locale("ko")
    app = ToolsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        operation_select = app.query_one("#tool-operation")
        assert len(operation_select._options) == len(OPERATIONS)
        assert app.title == translate_to("ko", "Kotonoha operations")
        assert app.query_one("#field-wav").display
        assert not app.query_one("#field-host").display
        assert [binding.description for binding in app._bindings.shown_keys] == [
            translate_to("ko", "Run"),
            translate_to("ko", "Stop"),
            translate_to("ko", "Clear output"),
            translate_to("ko", "Back"),
        ]
        app._write("styled output", "red")


async def test_operations_screen_switches_command_fields() -> None:
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


async def test_operations_screen_streams_a_child_process(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: Any,
) -> None:
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
async def test_config_editor_composes_one_row_per_field(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app._rows) == len(FIELDS)
        assert {r.specification.path for r in app._rows} == {f.path for f in FIELDS}


async def test_config_editor_uses_device_selectors(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    def fake_query_audio_devices() -> tuple[AudioDevice, ...]:
        return (
            AudioDevice(1, "Microphone", "ALSA", 2, 0, 48000.0),
            AudioDevice(2, "Speaker", "ALSA", 0, 2, 48000.0),
        )

    monkeypatch.setattr(config_app, "query_audio_devices", fake_query_audio_devices)
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        input_row = next(r for r in app._rows if r.specification.path == "audio.input_device")
        output_row = next(r for r in app._rows if r.specification.path == "audio.output_device")
        assert input_row.specification.kind == "device"
        assert output_row.specification.kind == "device"
        assert isinstance(input_row.editor, Select)
        assert isinstance(output_row.editor, Select)
        assert {value for _, value in input_row.editor._options} == {
            "",
            "Microphone, ALSA",
        }
        assert {value for _, value in output_row.editor._options} == {
            "",
            "Speaker, ALSA",
        }


async def test_custom_mode_exposes_per_role_placement_selectors(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    settings = load_settings()
    settings.perf_mode = "custom"
    settings.placement = {"llm": "remote"}
    app = ConfigApp(local_path=tmp_path / "local.yaml", settings=settings)
    async with app.run_test() as pilot:
        await pilot.pause()
        row = next(r for r in app._rows if r.specification.path == "placement")
        assert row.specification.kind == "placement"
        assert row.editor is None
        assert row.placement_editors["asr"].value == "local"
        assert row.placement_editors["llm"].value == "remote"

        row.placement_editors["llm"].value = "local"
        row.placement_editors["asr"].value = "remote"
        assert row.value() == {"asr": "remote"}


async def test_audio_device_selection_is_saved_as_a_stable_selector(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    monkeypatch.setattr(
        config_app,
        "query_audio_devices",
        lambda: (AudioDevice(3, "Microphone", "ALSA", 1, 0, 48000.0),),
    )
    target = tmp_path / "local.yaml"
    app = ConfigApp(local_path=target)
    async with app.run_test() as pilot:
        await pilot.pause()
        row = next(r for r in app._rows if r.specification.path == "audio.input_device")
        row.editor.value = "Microphone, ALSA"
        app._say = lambda message, style: None  # noqa: ARG005
        await app.action_save()
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["audio"]["input_device"] == (
        "Microphone, ALSA"
    )


async def test_audio_device_button_runs_the_stream_probe(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    monkeypatch.setattr(
        config_app,
        "query_audio_devices",
        lambda: (
            AudioDevice(3, "Microphone", "ALSA", 1, 0, 48000.0),
            AudioDevice(4, "Speaker", "ALSA", 0, 2, 48000.0),
        ),
    )
    captured: dict[str, Any] = {}

    def fake_probe(
        input_device: Any,
        output_device: Any,
        /,
        **settings: Any,
    ) -> AudioProbeResult:
        captured.update(
            input_device=input_device,
            output_device=output_device,
            settings=settings,
        )
        return AudioProbeResult(True, True)

    monkeypatch.setattr(config_app, "probe_audio_devices", fake_probe)
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._show_section("audio")
        await pilot.pause()
        await app.audio_test_pressed(Button.Pressed(app.query_one("#audio-test", Button)))
        await pilot.pause()
        assert str(app.query_one("#audio-test-status").render())
    assert captured["input_device"] is None
    assert captured["output_device"] is None
    assert captured["settings"]["channels"] == 1


async def test_config_editor_titles_follow_the_locale(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    set_locale("ja")
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.title == translate_to("ja", "Kotonoha configuration")
        assert [b.description for b in app._bindings.shown_keys] == [
            translate_to("ja", "Save"),
            translate_to("ja", "Reload"),
            translate_to("ja", "Categories"),
            translate_to("ja", "Quit"),
        ]


async def test_config_editor_shows_one_category_at_a_time(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
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


async def test_category_switch_preserves_unsaved_input(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._show_section("remote")
        row = next(r for r in app._rows if r.specification.path == "remote.services.llm")
        row.editor.value = "http://a6000.internal:8003"

        app._show_section("interface")
        app._show_section("remote")
        assert row.editor.value == "http://a6000.internal:8003"


async def test_menu_action_focuses_category_navigation(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_menu()
        await pilot.pause()
        assert app.focused is app.query_one("#category-list")


async def test_saving_without_edits_writes_nothing(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    target = tmp_path / "local.yaml"
    set_locale("en")
    app = ConfigApp(local_path=target)
    said: list[str] = []
    async with app.run_test() as pilot:
        await pilot.pause()
        app._say = lambda message, style: said.append(message)  # noqa: ARG005
        await app.action_save()
    assert said == [translate_to("en", "No changes to save")]
    assert not target.exists()


async def test_editing_a_field_persists_it(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    target = tmp_path / "local.yaml"
    set_locale("en")
    app = ConfigApp(local_path=target)
    async with app.run_test() as pilot:
        await pilot.pause()
        row = next(r for r in app._rows if r.specification.path == "perf_mode")
        row.editor.value = "hybrid"
        app._say = lambda message, style: None  # noqa: ARG005
        await app.action_save()
    assert target.exists()
    assert "hybrid" in target.read_text(encoding="utf-8")


async def test_reload_restores_the_stored_values(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        row = next(r for r in app._rows if r.specification.path == "perf_mode")
        original = row.editor.value
        row.editor.value = "remote"
        await app.action_reload()
        assert row.editor.value == original


class FakeRemoteConfigClient:
    __slots__: ClassVar[tuple[str, ...]] = (
        "changes",
        "config",
        "overrides",
    )
    def __init__(
        self,
        /,
    ) -> None:
        settings = load_settings("config/remote-server.yaml")
        self.config = settings.model_dump(mode="json", exclude={"root"})
        self.overrides: dict = {}
        self.changes: dict = {}

    async def read(
        self,
        /,
    ) -> RemoteConfigSnapshot:
        return self.snapshot()

    async def update(
        self,
        /,
        changes: dict,
    ) -> RemoteConfigSnapshot:
        self.changes = changes
        for path, value in changes.items():
            set_path(self.config, path, value)
            set_path(self.overrides, path, value)
        return self.snapshot()

    async def aclose(
        self,
        /,
    ) -> None:
        return None

    def snapshot(
        self,
        /,
    ) -> RemoteConfigSnapshot:
        return RemoteConfigSnapshot(
            config=self.config,
            editable_paths=sorted(REMOTE_EDITABLE_PATHS),
            overrides=self.overrides,
            path="/app/config/remote-server.local.yaml",
            restart_required=True,
        )


async def test_remote_target_loads_and_saves_through_the_admin_client(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    remote_client = FakeRemoteConfigClient()
    app.remote_client = remote_client
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#target-select").value = "remote"
        await pilot.pause()
        assert app.target == "remote"
        assert app.settings.llm.profile == "translategemma"
        assert app.query_one("#category-asr").display
        assert not app.query_one("#category-session").display

        row = next(
            row for row in app._rows if row.specification.path == "llm.max_model_len"
        )
        row.editor.value = "8192"
        await app.action_save()

        assert remote_client.changes == {"llm.max_model_len": 8192}
        assert app.settings.llm.max_model_len == 8192
        assert not (tmp_path / "local.yaml").exists()


async def test_collection_fields_accept_yaml_flow_values(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        row = next(row for row in app._rows if row.specification.path == "session.pair")
        row.editor.value = "[ja, zh-TW]"
        await app.action_save()
    written = yaml.safe_load((tmp_path / "local.yaml").read_text(encoding="utf-8"))
    assert written["session"]["pair"] == ["ja", "zh-TW"]


# -- main interface ---------------------------------------------------------
async def test_main_interface_composes_with_localized_labels(
    _positional_only: object | None = None,
    /,
    *,
    wav_path: Any,
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("KOTONOHA__FRONTEND__VAD__BACKEND", "energy")
    monkeypatch.setenv("KOTONOHA__SHM__NAME", "kotonoha_test_tui")
    monkeypatch.setenv("KOTONOHA__STORE__PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("KOTONOHA__LOGGING__LOG_PATH", str(tmp_path / "application.jsonl"))
    monkeypatch.setenv("KOTONOHA__LOGGING__TURN_LOG_PATH", str(tmp_path / "turns.jsonl"))
    set_locale("ko")

    from kotonoha._cli import _build
    from kotonoha._config import load_settings
    from kotonoha._logging_setup import (
        get_logger,
        reset_terminal_interface_logs,
        setup_logging,
    )

    settings = load_settings()
    setup_logging(
        settings.logging.level,
        settings.resolve(settings.logging.log_path),
        settings.logging.console,
        "orchestrator",
        terminal_interface=True,
    )
    orchestrator = _build(settings, wave_path=wav_path)
    app = KotonohaApp(orchestrator)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.title == translate_to("ko", "Kotonoha Interpreter")
        assert app.source_pane._title == translate_to("ko", "Source (ASR)")
        assert app.translation_pane._title == translate_to("ko", "Translation")
        assert app.source_pane.parent.id == "current-turn"
        assert app.translation_pane.parent.id == "current-turn"
        assert app.history_pane.parent.id == "panes"
        assert list(app.query_one("#panes").children)[-1] is app.history_pane
        current_turn = app.query_one("#current-turn")
        assert app.history_pane.region.y == current_turn.region.bottom
        assert app.history_pane.region.width == app.query_one("#panes").region.width
        assert str(app.query_one("#log-title").render()) == translate_to("ko", "Application logs")
        get_logger().info("tui.test", readable=True)
        await pilot.pause(0.2)
        log_text = "\n".join(line.text for line in app.log_output.lines)
        assert "tui.test" in log_text
        assert "readable=true" in log_text
        assert [b.description for b in app._bindings.shown_keys] == [
            translate_to("ko", "Talk (toggle)"),
            translate_to("ko", "PTT/auto"),
            translate_to("ko", "Routing"),
            translate_to("ko", "Clear"),
            translate_to("ko", "History"),
            translate_to("ko", "Text input"),
            translate_to("ko", "Leave text input"),
            translate_to("ko", "Quit"),
        ]
        app.source_pane.push("previous source")
        app.translation_pane.push("previous translation")
        app._frame_accumulator.push_translation("stale pending translation")
        app._apply(UiEvent("state", {"state": "LISTENING"}))
        assert app.source_pane._lines == []
        assert app.translation_pane._lines == []
        assert not app._frame_accumulator.advance().translation_changed
    setup_logging()
    reset_terminal_interface_logs()
