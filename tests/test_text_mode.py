"""Typed input: script detection, the state path, and the interpreter's input bar."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from kotonoha.config import load_settings
from kotonoha.core.lid import decide_typed_language, detect_script
from kotonoha.core.state import IllegalTransition, Machine, State
from kotonoha.i18n import set_locale, translate_to
from kotonoha.metrics import TurnMetrics
from kotonoha.tui.app import KotonohaApp

ALLOWED = ["ko", "en", "zh-TW", "ja"]


@pytest.fixture(autouse=True)
def _reset_locale():
    yield
    set_locale(None)


# -- script detection -------------------------------------------------------
def test_hangul_is_korean():
    lang, confidence = detect_script("다음 주 화요일까지 보내주세요")
    assert lang == "ko" and confidence == 1.0


def test_kana_is_japanese_even_with_kanji():
    lang, confidence = detect_script("資料を共有します")
    assert lang == "ja" and confidence == 1.0


def test_han_without_kana_is_traditional_chinese():
    """Documented heuristic: kanji-only Japanese is not what this mode receives."""
    lang, _ = detect_script("軟體資訊已經更新了")
    assert lang == "zh-TW"


def test_latin_is_english():
    lang, confidence = detect_script("Please send the list by Tuesday")
    assert lang == "en" and confidence == 1.0


def test_korean_with_hanja_stays_korean():
    lang, _ = detect_script("會議는 세 시입니다")
    assert lang == "ko"


def test_punctuation_and_digits_are_not_decisive():
    assert detect_script("12:30 — 100%") == (None, 0.0)
    assert detect_script("") == (None, 0.0)


def test_mixed_script_reports_a_partial_share():
    lang, confidence = detect_script("회의는 3pm start")
    assert lang == "ko"
    assert 0.0 < confidence < 1.0


# -- language decision ------------------------------------------------------
def test_explicit_setting_overrides_the_script():
    decision = decide_typed_language("Hello", "ja", None, ALLOWED)
    assert decision.language == "ja" and decision.source == "forced"


def test_script_decides_when_the_setting_is_auto():
    decision = decide_typed_language("資料を共有します", "auto", "ko", ALLOWED)
    assert decision.language == "ja" and decision.source == "script"


def test_undecidable_text_inherits_the_previous_language():
    """Same rule §5 applies to a short spoken utterance."""
    decision = decide_typed_language("...", "auto", "zh-TW", ALLOWED)
    assert decision.language == "zh-TW" and decision.source == "inherited"


def test_weak_script_share_inherits():
    decision = decide_typed_language("ok 확인", "auto", "en", ALLOWED, min_confidence=0.9)
    assert decision.source == "inherited"
    assert decision.language == "en"


def test_unusable_setting_falls_back_to_the_first_allowed_language():
    decision = decide_typed_language("Hello", "de", None, ALLOWED)
    assert decision.language == "ko" and decision.source == "inherited"


# -- state machine ----------------------------------------------------------
def test_typed_turn_goes_straight_from_idle_to_processing():
    """There is no utterance to segment, so LISTENING is skipped."""
    machine = Machine()
    machine.to(State.PROCESSING, "text")
    assert machine.state is State.PROCESSING
    machine.to(State.SPEAKING)
    machine.to(State.IDLE)


def test_speaking_is_still_unreachable_from_idle():
    machine = Machine()
    with pytest.raises(IllegalTransition):
        machine.to(State.SPEAKING, "skip")


def test_metrics_record_the_input_mode():
    m = TurnMetrics()
    assert m.to_dict()["input_mode"] == "voice"
    m.input_mode = "text"
    assert m.to_dict()["input_mode"] == "text"


# -- configuration ----------------------------------------------------------
def test_text_is_a_valid_session_mode():
    settings = load_settings()
    settings.session.mode = "text"
    assert settings.session.text_source_language == "auto"


# -- interpreter interface --------------------------------------------------
@pytest.fixture
def wav_path(tmp_path):
    path = tmp_path / "probe.wav"
    sr = 16000
    t = np.arange(int(0.5 * sr)) / sr
    x = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return path


@pytest.fixture
def interpreter(wav_path, tmp_path, monkeypatch):
    monkeypatch.setenv("KOTONOHA__FRONTEND__VAD__BACKEND", "energy")
    monkeypatch.setenv("KOTONOHA__SHM__NAME", "kotonoha_test_text")
    monkeypatch.setenv("KOTONOHA__STORE__PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("KOTONOHA__LOGGING__TURN_LOG_PATH", str(tmp_path / "turns.jsonl"))

    from kotonoha.cli import _build

    return _build(load_settings(), wave_path=wav_path)


async def test_input_bar_is_hidden_until_text_mode(interpreter):
    app = KotonohaApp(interpreter)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app.text_input.display

        app.action_text_mode()
        await pilot.pause()
        assert app.text_input.display
        assert interpreter.settings.session.mode == "text"
        assert app.status.mode == "text"


async def test_the_t_key_enters_text_mode(interpreter):
    """Press the key, not the action.

    A hidden Input is still focusable and would take focus on mount, swallowing
    every single-letter binding including the one that reveals it.
    """
    app = KotonohaApp(interpreter)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.focused is not app.text_input
        await pilot.press("t")
        await pilot.pause()
        assert interpreter.settings.session.mode == "text"
        assert app.text_input.display
        assert app.focused is app.text_input


async def test_escape_leaves_text_mode_from_the_focused_field(interpreter):
    """`t` becomes a character once the field has focus, so escape is the way out."""
    app = KotonohaApp(interpreter)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()
        assert interpreter.settings.session.mode == "text"

        await pilot.press("escape")
        await pilot.pause()
        assert interpreter.settings.session.mode == "push_to_talk"
        assert not app.text_input.display
        assert app.focused is not app.text_input


async def test_typing_t_into_the_field_does_not_leave_text_mode(interpreter):
    app = KotonohaApp(interpreter)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()
        assert interpreter.settings.session.mode == "text"
        assert app.text_input.value == "t"


async def test_leaving_text_mode_restores_the_previous_voice_mode(interpreter):
    interpreter.settings.session.mode = "auto"
    app = KotonohaApp(interpreter)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_text_mode()
        await pilot.pause()
        app.action_text_mode()
        await pilot.pause()
        assert interpreter.settings.session.mode == "auto"
        assert not app.text_input.display


async def test_mode_key_cycles_through_all_three(interpreter):
    app = KotonohaApp(interpreter)
    async with app.run_test() as pilot:
        await pilot.pause()
        seen = [interpreter.settings.session.mode]
        for _step in range(3):
            app.action_toggle_mode()
            await pilot.pause()
            seen.append(interpreter.settings.session.mode)
    assert seen == ["push_to_talk", "auto", "text", "push_to_talk"]


async def test_text_mode_closes_the_microphone(interpreter):
    """The VAD would otherwise segment room noise while the operator types."""
    app = KotonohaApp(interpreter)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_text_mode()
        await pilot.pause()
        assert not interpreter.capture.gate_open


async def test_submitting_the_input_starts_a_turn(interpreter, monkeypatch):
    submitted: list[str] = []

    async def fake_submit(text, src_lang=None):
        submitted.append(text)
        return True

    monkeypatch.setattr(interpreter, "submit_text", fake_submit)
    app = KotonohaApp(interpreter)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_text_mode()
        await pilot.pause()
        app.text_input.value = "  다음 주에 뵙겠습니다  "
        await pilot.press("enter")
        await pilot.pause()
    assert submitted == ["다음 주에 뵙겠습니다"]


async def test_blank_submission_is_ignored(interpreter, monkeypatch):
    submitted: list[str] = []

    async def fake_submit(text, src_lang=None):
        submitted.append(text)
        return True

    monkeypatch.setattr(interpreter, "submit_text", fake_submit)
    app = KotonohaApp(interpreter)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_text_mode()
        await pilot.pause()
        app.text_input.value = "   "
        await pilot.press("enter")
        await pilot.pause()
    assert submitted == []


async def test_placeholder_follows_the_locale(interpreter):
    set_locale("ja")
    app = KotonohaApp(interpreter)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.text_input.placeholder == translate_to(
            "ja", "Type an utterance and press Enter. Press t to return to voice."
        )


# -- orchestrator guards ----------------------------------------------------
async def test_blank_text_is_refused(interpreter):
    assert await interpreter.submit_text("   ") is False


async def test_submission_is_refused_while_a_turn_runs(interpreter):
    interpreter.machine.to(State.LISTENING, "test")
    assert await interpreter.submit_text("hello") is False
    interpreter.machine.force_idle()
