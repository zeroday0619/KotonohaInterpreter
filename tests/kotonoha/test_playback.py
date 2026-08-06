"""Playback completion behavior for TTS turns."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kotonoha._config import load_settings
from kotonoha.audio._playback import NullPlayback
from kotonoha.core._state import State


async def test_null_playback_finishes_a_turn_without_audio(
    _positional_only: object | None = None,
    /,
) -> None:
    settings = load_settings()
    playback = NullPlayback(settings.audio, settings.tts)
    playback.start(asyncio.get_running_loop())
    playback.begin_turn()

    playback.finish_turn()

    assert await playback.wait_drained(timeout=0.01)


async def test_text_turn_returns_to_idle_after_processing_failure(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("KOTONOHA__FRONTEND__VAD__BACKEND", "energy")
    monkeypatch.setenv("KOTONOHA__SHM__NAME", "kotonoha_test_playback")
    monkeypatch.setenv("KOTONOHA__STORE__PATH", str(tmp_path / "store.db"))
    monkeypatch.setenv("KOTONOHA__LOGGING__TURN_LOG_PATH", str(tmp_path / "turns.jsonl"))

    from kotonoha._cli import _build

    interpreter = _build(load_settings(), text_only=True)

    async def fail_processing(
        text: str,
        /,
        source_language: str | None,
    ) -> None:
        del text, source_language
        interpreter.machine.to(State.PROCESSING, "test")
        interpreter.machine.to(State.SPEAKING, "test")
        raise RuntimeError("test failure")

    monkeypatch.setattr(interpreter, "_process_text", fail_processing)

    with pytest.raises(RuntimeError, match="test failure"):
        await interpreter.submit_text("hello")

    assert interpreter.machine.state is State.IDLE
