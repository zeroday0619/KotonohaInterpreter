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
from kotonoha.audio._devices import resolve_audio_stream
from kotonoha.audio._resample import Resampler

log = get_logger(__name__)


class Playback:
    __slots__: ClassVar[tuple[str, ...]] = (
        "_current",
        "_lock",
        "_loop",
        "_output_channels",
        "_output_device",
        "_output_sample_rate",
        "_pending_samples",
        "_position",
        "_production_finished",
        "_queue_progress",
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
    _queue_progress: asyncio.Event
    _queue: deque[np.ndarray]
    _lock: threading.Lock
    _current: np.ndarray | None
    _position: int
    _pending_samples: int
    _stream: Any | None
    _resampler: Resampler
    _loop: asyncio.AbstractEventLoop | None
    _production_finished: bool
    _output_channels: int
    _output_device: int | None
    _output_sample_rate: int

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
        self._pending_samples = 0
        self._stream = None
        self._output_channels = 1
        self._output_device = None
        self._output_sample_rate = audio.playback_sample_rate
        self._resampler = Resampler(
            text_to_speech.sample_rate,
            self._output_sample_rate,
        )

        self._loop = None
        self.first_packet = asyncio.Event()
        self.drained = asyncio.Event()
        self._queue_progress = asyncio.Event()
        self._queue_progress.set()
        self.drained.set()
        self._production_finished = True

    # -- lifecycle -------------------------------------------------------
    def start(
        self,
        /,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        import sounddevice

        if self._stream is not None:
            return
        self._loop = loop or asyncio.get_event_loop()
        stream_settings = resolve_audio_stream(
            self.audio.output_device,
            "output",
            requested_sample_rate=self.audio.playback_sample_rate,
            requested_channels=1,
        )
        self._output_sample_rate = stream_settings.sample_rate
        self._output_channels = stream_settings.channels
        self._output_device = stream_settings.device_index
        self._resampler = Resampler(
            self.text_to_speech.sample_rate,
            self._output_sample_rate,
        )

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
                with self._lock:
                    if self._current is None or self._position >= self._current.size:
                        self._current = self._queue.popleft() if self._queue else None
                        self._position = 0
                    if self._current is None:
                        output_data[written:, :] = 0.0
                        if self._production_finished and not self.drained.is_set():
                            self._signal_drained()
                        self._signal_queue_progress()
                        return
                    write_count = min(
                        frame_count - written,
                        self._current.size - self._position,
                    )
                    samples = self._current[
                        self._position : self._position + write_count
                    ]
                    output_data[written : written + write_count, :] = samples[:, np.newaxis]
                    self._position += write_count
                    self._pending_samples -= write_count
                written += write_count
                if not self.first_packet.is_set():
                    self._signal_first()
            self._signal_queue_progress()

        stream = sounddevice.OutputStream(
            samplerate=self._output_sample_rate,
            channels=self._output_channels,
            dtype="float32",
            device=self._output_device,
            callback=playback_callback,
        )
        try:
            stream.start()
        except Exception:
            try:
                stream.close()
            except Exception as close_error:  # noqa: BLE001
                log.warning("playback.close_failed", error=repr(close_error))
            raise
        self._stream = stream
        log.info(
            "playback.started",
            rate=self._output_sample_rate,
            requested_rate=self.audio.playback_sample_rate,
            channels=self._output_channels,
            device=stream_settings.selector,
        )

    def stop(
        self,
        /,
    ) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception as error:  # noqa: BLE001
                log.warning("playback.stop_failed", error=repr(error))
            try:
                stream.close()
            except Exception as error:  # noqa: BLE001
                log.warning("playback.close_failed", error=repr(error))
        self.flush()

    # -- queue -----------------------------------------------------------
    def begin_turn(
        self,
        /,
    ) -> None:
        """New turn — rearm the instrumentation events."""
        with self._lock:
            self._queue.clear()
            self._current = None
            self._position = 0
            self._pending_samples = 0
            self._production_finished = False
        self._resampler.reset()
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
        if source_rate != self._output_sample_rate:
            resampler = (
                self._resampler
                if source_rate == self.text_to_speech.sample_rate
                else Resampler(source_rate, self._output_sample_rate)
            )
            samples = resampler(samples, last=resampler is not self._resampler)
        if samples.size == 0:
            return
        with self._lock:
            self._queue.append(samples)
            self._pending_samples += samples.size
        self.drained.clear()

    async def enqueue_bounded(
        self,
        pcm: np.ndarray,
        /,
        *,
        rate: int | None = None,
        maximum_seconds: float,
    ) -> None:
        """Enqueue PCM while limiting audio waiting ahead of the speaker."""
        if maximum_seconds <= 0:
            raise ValueError("maximum_seconds must be positive")
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        source_rate = rate or self.text_to_speech.sample_rate
        segment_sample_count = max(
            1,
            int(source_rate * min(0.25, maximum_seconds)),
        )
        for offset in range(0, samples.size, segment_sample_count):
            segment = samples[offset : offset + segment_sample_count]
            segment_seconds = segment.size / float(source_rate)
            while True:
                self._queue_progress.clear()
                if self.pending_seconds + segment_seconds <= maximum_seconds:
                    break
                await self._queue_progress.wait()
            # A slice otherwise retains the complete source chunk until playback
            # reaches this segment, defeating the queue's memory bound.
            if segment.size != samples.size:
                segment = segment.copy()
            self.enqueue(segment, source_rate)

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
        tail = self._resampler(np.zeros(0, dtype=np.float32), last=True)
        self._resampler.reset()
        with self._lock:
            if tail.size:
                self._queue.append(tail)
                self._pending_samples += tail.size
            self._production_finished = True
            queue_empty = not self._queue and self._current is None
        if queue_empty:
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
            self._pending_samples = 0
            self._production_finished = True
        self._resampler.reset()
        self._signal_drained()
        self._signal_queue_progress()

    @property
    def pending_seconds(
        self,
        /,
    ) -> float:
        with self._lock:
            pending_samples = self._pending_samples
        return pending_samples / float(self._output_sample_rate)

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
        if self._loop is None or self._loop.is_closed():
            return
        try:
            if asyncio.get_running_loop() is self._loop:
                self.first_packet.set()
                return
        except RuntimeError:
            pass
        self._loop.call_soon_threadsafe(self.first_packet.set)

    def _signal_drained(
        self,
        /,
    ) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        try:
            if asyncio.get_running_loop() is self._loop:
                self.drained.set()
                return
        except RuntimeError:
            pass
        self._loop.call_soon_threadsafe(self.drained.set)

    def _signal_queue_progress(
        self,
        /,
    ) -> None:
        if self._queue_progress.is_set():
            return
        if self._loop is None or self._loop.is_closed():
            return
        try:
            if asyncio.get_running_loop() is self._loop:
                self._queue_progress.set()
                return
        except RuntimeError:
            pass
        self._loop.call_soon_threadsafe(self._queue_progress.set)


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
