"""Capture and playback backed by a browser rather than a local audio device.

The orchestrator is unchanged by this: it consumes `capture.frames` and calls
`playback.enqueue`, so a session whose audio arrives over a WebSocket presents the
same surface as one reading PortAudio. Half-duplex gating stays authoritative on
this side. A browser that keeps sending while the gate is shut has its frames
dropped here, exactly as the device capture drops blocks, so a client that ignores
the mute signal still cannot feed synthesized speech back into the microphone.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Final

import numpy as np

from kotonoha.audio._capture import Frame
from kotonoha.audio._resample import resample_once

# The segmenter steps once per Silero window.
FRAME_SAMPLES: Final = 512


class BrowserCapture:
    """Assemble WebSocket audio into the frames the segmenter expects."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "_buffer",
        "_fill",
        "_gate_open",
        "_sample_index",
        "_source_sample_rate",
        "_work_sample_rate",
        "dropped_blocks",
        "frames",
        "loop",
        "overflows",
    )

    frames: asyncio.Queue[Frame]
    loop: asyncio.AbstractEventLoop | None
    dropped_blocks: int
    overflows: int
    _buffer: np.ndarray
    _fill: int
    _gate_open: bool
    _sample_index: int
    _source_sample_rate: int
    _work_sample_rate: int

    def __init__(
        self,
        /,
        work_sample_rate: int = 16000,
        **_ignored: Any,
    ) -> None:
        # One frame of slack only. A browser that stalls must lose the backlog
        # rather than replay stale audio into a later turn.
        self.frames: asyncio.Queue[Frame] = asyncio.Queue(maxsize=8)
        self.loop = None
        self.dropped_blocks = 0
        self.overflows = 0
        self._work_sample_rate = work_sample_rate
        self._source_sample_rate = work_sample_rate
        self._buffer = np.empty(FRAME_SAMPLES, dtype=np.float32)
        self._fill = 0
        self._sample_index = 0
        self._gate_open = False

    @property
    def gate_open(
        self,
        /,
    ) -> bool:
        return self._gate_open

    def close_gate(
        self,
        /,
    ) -> None:
        """Entering SPEAKING. Everything still in flight belongs to the old turn."""
        self._gate_open = False
        self._fill = 0
        self._drain()

    def open_gate(
        self,
        /,
    ) -> None:
        self._gate_open = False
        self._fill = 0
        self._sample_index = 0
        self._drain()
        self._gate_open = True

    def start(
        self,
        /,
    ) -> None:
        self.loop = None
        self.open_gate()

    def stop(
        self,
        /,
    ) -> None:
        self._gate_open = False
        self._drain()

    def tail48(
        self,
        /,
        work_sample_count: int,
    ) -> np.ndarray:
        """No original-rate ring exists for browser audio, so denoise is skipped.

        The orchestrator treats a short return as "nothing to clean" and keeps the
        segmenter's own audio. Browsers already apply echo cancellation and noise
        suppression to a `getUserMedia` stream.
        """
        del work_sample_count
        return np.zeros(0, dtype=np.float32)

    def set_source_rate(
        self,
        /,
        sample_rate: int,
    ) -> None:
        """Adopt the rate the client reports.

        A browser is free to ignore a requested AudioContext rate. Reading its
        declared rate and resampling here is what keeps the pitch correct, rather
        than assuming the request was honoured.
        """
        if sample_rate > 0:
            self._source_sample_rate = sample_rate

    def push(
        self,
        /,
        pcm: np.ndarray,
    ) -> None:
        """Accept one block of client audio and emit whole frames from it."""
        if not self._gate_open or pcm.size == 0:
            return
        if self._source_sample_rate != self._work_sample_rate:
            pcm = resample_once(pcm, self._source_sample_rate, self._work_sample_rate)

        offset = 0
        while offset < pcm.size:
            take = min(FRAME_SAMPLES - self._fill, pcm.size - offset)
            self._buffer[self._fill : self._fill + take] = pcm[offset : offset + take]
            self._fill += take
            offset += take
            if self._fill < FRAME_SAMPLES:
                continue
            frame = Frame(index=self._sample_index, pcm=self._buffer.copy())
            self._sample_index += FRAME_SAMPLES
            self._fill = 0
            try:
                self.frames.put_nowait(frame)
            except asyncio.QueueFull:
                # Falling behind is reported rather than silently smoothed over:
                # a dropped frame is a hole in the utterance.
                self.overflows += 1
                try:
                    self.frames.get_nowait()
                    self.frames.put_nowait(frame)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    self.dropped_blocks += 1

    def _drain(
        self,
        /,
    ) -> None:
        while True:
            try:
                self.frames.get_nowait()
            except asyncio.QueueEmpty:
                return


class BrowserPlayback:
    """Hand synthesized audio to the client instead of an output device.

    The device playback tracks what remains to be heard from its own audio
    callback. Here the speaker is in the browser, so the client reports how much
    it has played and this class keeps the same accounting from those reports.
    That accounting is load-bearing rather than cosmetic: the orchestrator reopens
    the microphone only once playback has drained, and a server that believed
    audio was finished early would reopen it while the browser was still speaking
    and feed synthesized speech straight back into the next turn.

    A client that stops acknowledging therefore stalls the turn instead of
    breaking half-duplex; `wait_drained` carries the timeout that bounds it.
    """

    __slots__: ClassVar[tuple[str, ...]] = (
        "_played_samples",
        "_production_finished",
        "_progress",
        "_send_audio",
        "_send_control",
        "_sent_samples",
        "drained",
        "first_packet",
        "sample_rate",
    )

    first_packet: asyncio.Event
    drained: asyncio.Event
    sample_rate: int
    _send_audio: Any
    _send_control: Any
    _sent_samples: int
    _played_samples: int
    _production_finished: bool
    _progress: asyncio.Event

    def __init__(
        self,
        /,
        send_audio: Any,
        send_control: Any,
        sample_rate: int = 24000,
    ) -> None:
        self._send_audio = send_audio
        self._send_control = send_control
        self.sample_rate = sample_rate
        self.first_packet = asyncio.Event()
        self.drained = asyncio.Event()
        self.drained.set()
        self._progress = asyncio.Event()
        self._sent_samples = 0
        self._played_samples = 0
        self._production_finished = True

    def start(
        self,
        /,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        del loop

    def stop(
        self,
        /,
    ) -> None:
        self.flush()

    def begin_turn(
        self,
        /,
    ) -> None:
        self._sent_samples = 0
        self._played_samples = 0
        self._production_finished = False
        self.first_packet.clear()
        self.drained.clear()
        self._progress.set()
        self._send_control({"type": "playback_begin", "rate": self.sample_rate})

    def enqueue(
        self,
        /,
        pcm: np.ndarray,
        rate: int | None = None,
    ) -> None:
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        source_rate = rate or self.sample_rate
        if source_rate != self.sample_rate:
            samples = resample_once(samples, source_rate, self.sample_rate)
            if samples.size == 0:
                return
        self._sent_samples += samples.size
        self.drained.clear()
        if not self.first_packet.is_set():
            self.first_packet.set()
        self._send_audio(samples, self.sample_rate)

    async def enqueue_bounded(
        self,
        pcm: np.ndarray,
        /,
        *,
        rate: int | None = None,
        maximum_seconds: float,
    ) -> None:
        """Enqueue while limiting how much audio waits ahead of the speaker."""
        if maximum_seconds <= 0:
            raise ValueError("maximum_seconds must be positive")
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        source_rate = rate or self.sample_rate
        segment_sample_count = max(1, int(source_rate * min(0.25, maximum_seconds)))
        for offset in range(0, samples.size, segment_sample_count):
            segment = samples[offset : offset + segment_sample_count]
            while self.pending_seconds >= maximum_seconds:
                self._progress.clear()
                await self._progress.wait()
            self.enqueue(segment, source_rate)

    def acknowledge(
        self,
        /,
        played_samples: int,
    ) -> None:
        """The client reports cumulative samples it has actually played."""
        if played_samples < 0:
            return
        self._played_samples = min(self._sent_samples, max(self._played_samples, played_samples))
        self._progress.set()
        if self._production_finished and self._played_samples >= self._sent_samples:
            self.drained.set()

    def finish_turn(
        self,
        /,
    ) -> None:
        self._production_finished = True
        self._send_control({"type": "playback_end"})
        if self._played_samples >= self._sent_samples:
            self.drained.set()

    def flush(
        self,
        /,
    ) -> None:
        """Abort playback and tell the client to discard what it has queued."""
        self._production_finished = True
        self._played_samples = self._sent_samples
        self._progress.set()
        self.drained.set()
        self._send_control({"type": "playback_flush"})

    @property
    def pending_seconds(
        self,
        /,
    ) -> float:
        return max(0, self._sent_samples - self._played_samples) / float(self.sample_rate)

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
