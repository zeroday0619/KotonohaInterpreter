"""Playback completion behavior for TTS turns."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from kotonoha._config import load_settings
from kotonoha.audio import _playback
from kotonoha.audio._playback import NullPlayback, Playback
from kotonoha.core._state import State


class FakeOutputStream:
    __slots__: ClassVar[tuple[str, ...]] = (
        "callback",
        "closed",
        "started",
    )

    def __init__(
        self,
        /,
        **options: Any,
    ) -> None:
        self.callback = options["callback"]
        self.closed = False
        self.started = False

    def start(
        self,
        /,
    ) -> None:
        self.started = True

    def stop(
        self,
        /,
    ) -> None:
        self.started = False

    def close(
        self,
        /,
    ) -> None:
        self.closed = True

    def render(
        self,
        frame_count: int,
        /,
    ) -> np.ndarray:
        output = np.empty((frame_count, 1), dtype=np.float32)
        self.callback(output, frame_count, None, None)
        return output


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


async def test_transient_empty_queue_does_not_finish_while_tts_is_producing(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    streams: list[FakeOutputStream] = []

    def output_stream(
        _positional_only: object | None = None,
        /,
        **options: Any,
    ) -> FakeOutputStream:
        del _positional_only
        stream = FakeOutputStream(**options)
        streams.append(stream)
        return stream

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(OutputStream=output_stream),
    )
    monkeypatch.setattr(
        _playback,
        "resolve_audio_stream",
        lambda *_arguments, **_options: SimpleNamespace(
            sample_rate=settings.tts.sample_rate,
            channels=1,
            device_index=None,
            selector="test-output",
        ),
    )
    playback = Playback(settings.audio, settings.tts)
    playback.start(asyncio.get_running_loop())
    try:
        playback.begin_turn()
        playback.enqueue(np.ones(4, dtype=np.float32), settings.tts.sample_rate)
        streams[0].render(8)
        await asyncio.sleep(0)

        assert not playback.drained.is_set()

        playback.enqueue(np.ones(4, dtype=np.float32), settings.tts.sample_rate)
        playback.finish_turn()
        streams[0].render(8)

        assert await playback.wait_drained(timeout=0.1)
    finally:
        playback.stop()


async def test_playback_backpressure_bounds_queued_pcm(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    streams: list[FakeOutputStream] = []

    def output_stream(
        _positional_only: object | None = None,
        /,
        **options: Any,
    ) -> FakeOutputStream:
        del _positional_only
        stream = FakeOutputStream(**options)
        streams.append(stream)
        return stream

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(OutputStream=output_stream),
    )
    monkeypatch.setattr(
        _playback,
        "resolve_audio_stream",
        lambda *_arguments, **_options: SimpleNamespace(
            sample_rate=settings.tts.sample_rate,
            channels=1,
            device_index=None,
            selector="test-output",
        ),
    )
    playback = Playback(settings.audio, settings.tts)
    playback.start(asyncio.get_running_loop())
    playback.begin_turn()
    maximum_seconds = 0.001
    try:
        enqueue_task = asyncio.create_task(
            playback.enqueue_bounded(
                np.ones(48, dtype=np.float32),
                rate=settings.tts.sample_rate,
                maximum_seconds=maximum_seconds,
            )
        )
        await asyncio.sleep(0)

        assert not enqueue_task.done()
        assert playback.pending_seconds <= maximum_seconds

        streams[0].render(24)
        await asyncio.wait_for(enqueue_task, timeout=0.1)

        assert playback.pending_seconds <= maximum_seconds
    finally:
        playback.flush()
        playback.stop()


def test_playback_stop_releases_pending_audio(
    _positional_only: object | None = None,
    /,
) -> None:
    settings = load_settings()
    playback = Playback(settings.audio, settings.tts)
    playback.begin_turn()
    playback.enqueue(np.ones(24000, dtype=np.float32), settings.tts.sample_rate)

    assert playback.pending_seconds == 1.0

    playback.stop()

    assert playback.pending_seconds == 0.0


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
