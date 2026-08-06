"""TTS playback queue.

Stitches together the audio chunks that arrive clause by clause (§5.4). The
"first audio packet" and "queue drained" timestamps come from here — they are
only meaningful if they mark when audio actually reached the speaker, not when
the orchestrator enqueued it.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Any, ClassVar

import numpy as np

from kotonoha._config import AudioConfig, TextToSpeechConfig
from kotonoha._logging_setup import get_logger
from kotonoha._typing import override
from kotonoha.audio._resample import Resampler

log = get_logger(__name__)


class Playback:
    __slots__: ClassVar[tuple[str, ...]] = (
        "_closing",
        "_current",
        "_lock",
        "_loop",
        "_position",
        "_queue",
        "_resampler",
        "_stream",
        "audio",
        "drained",
        "first_packet",
        "text_to_speech",
    )
    audio: AudioConfig
    text_to_speech: TextToSpeechConfig
    first_packet: asyncio.Event
    drained: asyncio.Event
    _queue: deque[np.ndarray]
    _lock: threading.Lock
    _current: np.ndarray | None
    _position: int
    _stream: Any | None
    _resampler: Resampler
    _loop: asyncio.AbstractEventLoop | None
    _closing: bool

    @override
    def __init__(
        self,
        /,
        audio: AudioConfig,
        text_to_speech: TextToSpeechConfig,
    ) -> None:
        self.audio = audio
        self.text_to_speech = text_to_speech
        self._queue = deque()
        self._lock = threading.Lock()
        self._current = None
        self._position = 0
        self._stream = None
        self._resampler = Resampler(
            text_to_speech.sample_rate,
            audio.playback_sample_rate,
        )

        self._loop = None
        self.first_packet = asyncio.Event()
        self.drained = asyncio.Event()
        self.drained.set()
        self._closing = False

    # -- lifecycle -------------------------------------------------------
    def start(
        self,
        /,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        import sounddevice

        self._loop = loop or asyncio.get_event_loop()

        def playback_callback(
            output_data: Any,
            /,
            frame_count: Any,
            time_info: Any,
            status: Any,
        ) -> None:
            if status:
                log.debug("playback.status", status=str(status))
            written = 0
            while written < frame_count:
                if self._current is None or self._position >= self._current.size:
                    with self._lock:
                        self._current = self._queue.popleft() if self._queue else None
                    self._position = 0
                    if self._current is None:
                        output_data[written:, 0] = 0.0
                        if written > 0 or not self.drained.is_set():
                            self._signal_drained()
                        return
                write_count = min(
                    frame_count - written,
                    self._current.size - self._position,
                )
                output_data[written : written + write_count, 0] = self._current[
                    self._position : self._position + write_count
                ]
                self._position += write_count
                written += write_count
                if not self.first_packet.is_set():
                    self._signal_first()

        self._stream = sounddevice.OutputStream(
            samplerate=self.audio.playback_sample_rate,
            channels=1,
            dtype="float32",
            device=self.audio.output_device,
            callback=playback_callback,
        )
        self._stream.start()
        log.info("playback.started", rate=self.audio.playback_sample_rate)

    def stop(
        self,
        /,
    ) -> None:
        self._closing = True
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # -- queue -----------------------------------------------------------
    def begin_turn(
        self,
        /,
    ) -> None:
        """New turn — rearm the instrumentation events."""
        self.first_packet.clear()
        self.drained.clear()

    def enqueue(
        self,
        /,
        pcm: np.ndarray,
        rate: int | None = None,
    ) -> None:
        """Push a TTS chunk. Given a rate, resample from it to the output rate."""
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        source_rate = rate or self.text_to_speech.sample_rate
        if source_rate != self.audio.playback_sample_rate:
            resampler = (
                self._resampler
                if source_rate == self.text_to_speech.sample_rate
                else Resampler(source_rate, self.audio.playback_sample_rate)
            )
            samples = resampler(samples)
        with self._lock:
            self._queue.append(samples)
        self.drained.clear()

    def finish_turn(
        self,
        /,
    ) -> None:
        """Mark TTS production complete when no samples remain to be played.

        TTS can complete without producing samples, for example after an empty
        clause or a backend failure. In that case the audio callback has no
        future invocation that can discover an empty queue, so the producer
        must complete the drained event explicitly.
        """
        with self._lock:
            queue_empty = not self._queue
        if queue_empty and self._current is None:
            self._signal_drained()

    def flush(
        self,
        /,
    ) -> None:
        """Abort playback (cancellation or error) and empty the queue."""
        with self._lock:
            self._queue.clear()
        self._current = None
        self._position = 0
        self._signal_drained()

    @property
    def pending_seconds(
        self,
        /,
    ) -> float:
        with self._lock:
            sample_count = sum(chunk.size for chunk in self._queue)
        if self._current is not None:
            sample_count += max(0, self._current.size - self._position)
        return sample_count / float(self.audio.playback_sample_rate)

    async def wait_drained(
        self,
        /,
        timeout: float | None = None,
    ) -> bool:
        try:
            await asyncio.wait_for(self.drained.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # -- callback thread -> event loop -----------------------------------
    def _signal_first(
        self,
        /,
    ) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.first_packet.set)

    def _signal_drained(
        self,
        /,
    ) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.drained.set)


class NullPlayback(Playback):
    """For environments with no audio output device (CI, remote shells)."""
    __slots__: ClassVar[tuple[str, ...]] = (
        "_loop",
    )

    _loop: asyncio.AbstractEventLoop | None

    @override
    def start(
        self,
        /,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        log.warning("playback.null", reason="no output device")

    @override
    def stop(
        self,
        /,
    ) -> None:
        return None

    @override
    def enqueue(
        self,
        /,
        pcm: np.ndarray,
        rate: int | None = None,
    ) -> None:
        if not self.first_packet.is_set():
            self._signal_first()
        self._signal_drained()
