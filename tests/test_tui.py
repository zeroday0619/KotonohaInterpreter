"""Terminal interfaces compose and pick up the active locale.

These exercise the Textual API surface the applications depend on — binding
registration, CSS parsing, widget construction — which unit tests of the underlying
logic do not reach. A Textual upgrade that changes any of it fails here rather than on
the device.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from kotonoha.i18n import CATALOGS, set_locale
from kotonoha.tui.app import KotonohaApp
from kotonoha.tui.config_app import FIELDS, ConfigApp


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
        assert not app.query_one("#panel-models").display

        await pilot.click("#category-models")
        await pilot.pause()
        assert app.current_section == "models"
        assert app.query_one("#panel-models").display
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
        assert app.focused is app.query_one("#category-list")


async def test_saving_without_edits_writes_nothing(tmp_path):
    target = tmp_path / "local.yaml"
    set_locale("en")
    app = ConfigApp(local_path=target)
    said: list[str] = []
    async with app.run_test() as pilot:
        await pilot.pause()
        app._say = lambda message, style: said.append(message)  # noqa: ARG005
        app.action_save()
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
        app.action_save()
    assert target.exists()
    assert "hybrid" in target.read_text(encoding="utf-8")


async def test_reload_restores_the_stored_values(tmp_path):
    app = ConfigApp(local_path=tmp_path / "local.yaml")
    async with app.run_test() as pilot:
        await pilot.pause()
        row = next(r for r in app._rows if r.spec.path == "perf_mode")
        original = row.editor.value
        row.editor.value = "remote"
        app.action_reload()
        assert row.editor.value == original


# -- main interface ---------------------------------------------------------
async def test_main_interface_composes_with_localized_labels(wav_path, tmp_path, monkeypatch):
    monkeypatch.setenv("KOTONOHA__FRONTEND__VAD__BACKEND", "energy")
    monkeypatch.setenv("KOTONOHA__SHM__NAME", "kotonoha_test_tui")
    monkeypatch.setenv("KOTONOHA__STORE__PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("KOTONOHA__LOGGING__TURN_LOG_PATH", str(tmp_path / "turns.jsonl"))
    set_locale("ko")

    from kotonoha.cli import _build
    from kotonoha.config import load_settings

    orch = _build(load_settings(), wav=wav_path)
    app = KotonohaApp(orch)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.title == CATALOGS["ko"]["tui.title"]
        assert app.src._title == CATALOGS["ko"]["tui.pane.source"]
        assert app.tgt._title == CATALOGS["ko"]["tui.pane.target"]
        assert [b.description for b in app._bindings.shown_keys] == [
            CATALOGS["ko"]["tui.key.talk"],
            CATALOGS["ko"]["tui.key.mode"],
            CATALOGS["ko"]["tui.key.routing"],
            CATALOGS["ko"]["tui.key.clear"],
            CATALOGS["ko"]["tui.key.quit"],
        ]
