"""Typed input script detection, state transitions, and orchestrator guards."""

from __future__ import annotations

import wave
from typing import Any

import numpy as np
import pytest

from kotonoha._config import load_settings
from kotonoha._metrics import TurnMetrics
from kotonoha.core._lid import decide_typed_language, detect_script
from kotonoha.core._state import IllegalTransition, Machine, State

ALLOWED = ["ko", "en", "zh-TW", "ja"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("다음 주 화요일까지 보내주세요", "ko"),
        ("資料を共有します", "ja"),
        ("軟體資訊已經更新了", "zh-TW"),
        ("Please send the list by Tuesday", "en"),
        ("會議는 세 시입니다", "ko"),
    ],
)
def test_script_detection(
    _positional_only: object | None = None,
    /,
    *,
    text: str,
    expected: str,
) -> None:
    language, confidence = detect_script(text)
    assert language == expected
    assert confidence > 0.0


def test_punctuation_and_digits_are_not_decisive() -> None:
    assert detect_script("12:30 — 100%") == (None, 0.0)
    assert detect_script("") == (None, 0.0)


def test_explicit_language_overrides_script() -> None:
    decision = decide_typed_language("Hello", "ja", None, ALLOWED)
    assert decision.language == "ja" and decision.source == "forced"


def test_script_and_inherited_language_decisions() -> None:
    scripted = decide_typed_language("資料を共有します", "auto", "ko", ALLOWED)
    inherited = decide_typed_language("...", "auto", "zh-TW", ALLOWED)
    assert scripted.language == "ja" and scripted.source == "script"
    assert inherited.language == "zh-TW" and inherited.source == "inherited"


def test_typed_turn_skips_listening() -> None:
    machine = Machine()
    machine.to(State.PROCESSING, "text")
    assert machine.state is State.PROCESSING
    machine.to(State.SPEAKING)
    machine.to(State.IDLE)


def test_speaking_is_unreachable_from_idle() -> None:
    with pytest.raises(IllegalTransition):
        Machine().to(State.SPEAKING, "skip")


def test_metrics_record_text_input() -> None:
    metrics = TurnMetrics()
    metrics.input_mode = "text"
    assert metrics.to_dict()["input_mode"] == "text"


def test_text_is_a_valid_session_mode() -> None:
    settings = load_settings()
    settings.session.mode = "text"
    assert settings.session.text_source_language == "auto"


@pytest.fixture
def interpreter(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
    monkeypatch: Any,
) -> Any:
    wave_path = tmp_path / "probe.wav"
    sample_rate = 16000
    timeline = np.arange(int(0.5 * sample_rate)) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * 200 * timeline)).astype(np.float32)
    with wave.open(str(wave_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((samples * 32767).astype("<i2").tobytes())
    monkeypatch.setenv("KOTONOHA__FRONTEND__VAD__BACKEND", "energy")
    monkeypatch.setenv("KOTONOHA__SHM__NAME", "kotonoha_test_text")
    monkeypatch.setenv("KOTONOHA__STORE__PATH", str(tmp_path / "turns.db"))
    monkeypatch.setenv("KOTONOHA__LOGGING__TURN_LOG_PATH", str(tmp_path / "turns.jsonl"))
    from kotonoha._cli import _build

    return _build(load_settings(), wave_path=wave_path)


def test_target_language_rejects_an_unsupported_language(
    _positional_only: object | None = None,
    /,
    *,
    interpreter: Any,
) -> None:
    with pytest.raises(ValueError, match="unsupported target language"):
        interpreter.set_target_language("de")


async def test_blank_text_is_refused(
    _positional_only: object | None = None,
    /,
    *,
    interpreter: Any,
) -> None:
    assert await interpreter.submit_text("   ") is False


async def test_submission_is_refused_while_a_turn_runs(
    _positional_only: object | None = None,
    /,
    *,
    interpreter: Any,
) -> None:
    interpreter.machine.to(State.LISTENING, "test")
    assert await interpreter.submit_text("hello") is False
    interpreter.machine.force_idle()
