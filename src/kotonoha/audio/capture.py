"""Microphone capture and half-duplex gating.

Capture runs at 48 kHz for two reasons:
  1. DeepFilterNet3 is 48 kHz only. Downsample to 16k first and it is unusable.
  2. Most USB microphones default to 48k, so resampling happens exactly once
     outside the kernel.

The VAD runs on 16k frames while the original 48k audio is kept in a separate
ring. When EOU fires, the utterance length in 16k samples is multiplied by three
and that many samples are pulled from the 48k ring for noise suppression. (The
streaming resampler retains bounded frame carry. The preroll remains attached to
the segmented utterance.)

Half-duplex gating (§4): closing the gate in SPEAKING drops incoming blocks on
the spot and resets the resampler and VAD state. If TTS output leaks back into
the microphone and is heard as a new utterance, the result is an infinite loop —
this gate is not negotiable.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from kotonoha.audio.resample import Resampler
from kotonoha.config import AudioConfig, VadConfig
from kotonoha.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Frame:
    """One 16k frame, sized for a single VAD step."""

    index: int  # start index in 16k samples, counted since the gate opened
    pcm: np.ndarray


class RawRing:
    """Fixed-length ring holding the original 48k audio, for pulling the tail back."""

    _buffer: np.ndarray
    _capacity: int
    _written: int

    def __init__(self, capacity: int):
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._written = 0

    def push(self, samples: np.ndarray) -> None:
        sample_count = samples.shape[0]
        if sample_count >= self._capacity:
            self._buffer[:] = samples[-self._capacity :]
            self._written += sample_count
            return
        position = self._written % self._capacity
        end = position + sample_count
        if end <= self._capacity:
            self._buffer[position:end] = samples
        else:
            first_part_length = self._capacity - position
            self._buffer[position:] = samples[:first_part_length]
            self._buffer[: end - self._capacity] = samples[first_part_length:]
        self._written += sample_count

    def tail(self, sample_count: int) -> np.ndarray:
        sample_count = min(sample_count, self._capacity, self._written)
        if sample_count == 0:
            return np.zeros(0, dtype=np.float32)
        position = self._written % self._capacity
        start = position - sample_count
        if start >= 0:
            return self._buffer[start:position].copy()
        return np.concatenate([self._buffer[start:], self._buffer[:position]])

    def clear(self) -> None:
        self._written = 0
        self._buffer[:] = 0.0


class MicCapture:
    audio: AudioConfig
    loop: asyncio.AbstractEventLoop | None
    frames: asyncio.Queue[Frame]
    dropped_blocks: int
    overflows: int
    _raw_queue: queue.Queue[np.ndarray | None]
    _stream: Any | None
    _worker_thread: threading.Thread | None
    _stop_event: threading.Event
    _gate_open: threading.Event
    _resampler: Resampler
    _pending_samples: np.ndarray
    _sample_index: int
    _raw_capture: RawRing
    _window_size: int

    def __init__(
        self,
        audio: AudioConfig,
        vad: VadConfig,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.audio = audio
        self.loop = loop
        self._window_size = 512
        self.frames = asyncio.Queue(maxsize=256)

        self._raw_queue = queue.Queue(maxsize=64)
        self._stream = None
        self._worker_thread = None
        self._stop_event = threading.Event()

        self._gate_open = threading.Event()
        self._gate_open.set()

        self._resampler = Resampler(audio.capture_sample_rate, audio.work_sample_rate)
        self._pending_samples = np.zeros(0, dtype=np.float32)
        self._sample_index = 0

        ring_seconds = (vad.max_utterance_ms + vad.preroll_ms) / 1000.0 + 2.0
        self._raw_capture = RawRing(int(ring_seconds * audio.capture_sample_rate))
        self.dropped_blocks = 0
        self.overflows = 0

    # -- gating (§4) -----------------------------------------------------
    @property
    def gate_open(self) -> bool:
        return self._gate_open.is_set()

    def close_gate(self) -> None:
        """Entering SPEAKING. Microphone shut."""
        self._gate_open.clear()

    def open_gate(self) -> None:
        """Back to IDLE. Discard any TTS tail still queued, then open."""
        self._drain_raw_queue()
        while not self.frames.empty():
            try:
                self.frames.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._pending_samples = np.zeros(0, dtype=np.float32)
        self._raw_capture.clear()
        self._resampler.reset()
        self._gate_open.set()

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        import sounddevice as sd

        self.loop = self.loop or asyncio.get_event_loop()

        def capture_callback(  # noqa: ANN001 - PortAudio callback signature
            input_data,
            frame_count,
            time_info,
            status,
        ):
            if status:
                self.overflows += 1
            if not self._gate_open.is_set():
                self.dropped_blocks += 1
                return
            try:
                self._raw_queue.put_nowait(
                    np.array(input_data[:, 0], dtype=np.float32, copy=True)
                )
            except queue.Full:
                self.overflows += 1

        self._stream = sd.InputStream(
            samplerate=self.audio.capture_sample_rate,
            blocksize=self.audio.capture_block_frames,
            channels=self.audio.channels,
            dtype="float32",
            device=self.audio.input_device,
            callback=capture_callback,
        )
        self._stream.start()
        self._worker_thread = threading.Thread(
            target=self._run,
            name="mic-worker",
            daemon=True,
        )
        self._worker_thread.start()
        log.info(
            "mic.started",
            rate=self.audio.capture_sample_rate,
            block_ms=self.audio.capture_block_ms,
            device=str(self.audio.input_device),
        )

    def stop(self) -> None:
        self._stop_event.set()
        self._raw_queue.put(None)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        log.info("mic.stopped", dropped_blocks=self.dropped_blocks, overflows=self.overflows)

    # -- pulling the original 48k back -----------------------------------
    def tail48(self, work_sample_count: int) -> np.ndarray:
        ratio = self.audio.capture_sample_rate / self.audio.work_sample_rate
        return self._raw_capture.tail(int(work_sample_count * ratio))

    # -- internals -------------------------------------------------------
    def _drain_raw_queue(self) -> None:
        while True:
            try:
                self._raw_queue.get_nowait()
            except queue.Empty:
                return

    def _run(self) -> None:
        while not self._stop_event.is_set():
            block = self._raw_queue.get()
            if block is None:
                return
            if not self._gate_open.is_set():
                continue
            self._raw_capture.push(block)
            resampled = self._resampler(block)
            if resampled.size == 0:
                continue
            self._pending_samples = (
                resampled
                if self._pending_samples.size == 0
                else np.concatenate([self._pending_samples, resampled])
            )
            while self._pending_samples.size >= self._window_size:
                frame = self._pending_samples[: self._window_size]
                self._pending_samples = self._pending_samples[self._window_size :]
                output_frame = Frame(index=self._sample_index, pcm=frame)
                self._sample_index += self._window_size
                if self.loop is not None and not self.loop.is_closed():
                    self.loop.call_soon_threadsafe(self._emit, output_frame)

    def _emit(self, frame: Frame) -> None:
        try:
            self.frames.put_nowait(frame)
        except asyncio.QueueFull:
            self.overflows += 1


class NullCapture:
    """Opens no device and never yields a frame.

    Used by text-only runs, where the keyboard is the input source and touching
    PortAudio would fail on hosts without a microphone.
    """

    frames: asyncio.Queue[Frame]
    loop: asyncio.AbstractEventLoop | None
    dropped_blocks: int
    overflows: int
    _gate_open: bool

    def __init__(self, **_ignored):
        self.frames: asyncio.Queue[Frame] = asyncio.Queue()
        self.loop: asyncio.AbstractEventLoop | None = None
        self._gate_open = False
        self.dropped_blocks = 0
        self.overflows = 0

    @property
    def gate_open(self) -> bool:
        return self._gate_open

    def close_gate(self) -> None:
        self._gate_open = False

    def open_gate(self) -> None:
        self._gate_open = False  # there is nothing to open

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def tail48(self, work_sample_count: int) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


class FileCapture:
    """Streams a WAV file as if it were the microphone.

    Used for EOU regression tests and for development on macOS.
    """

    frames: asyncio.Queue[Frame]
    dropped_blocks: int
    overflows: int
    _pcm: np.ndarray
    _window: int
    _gate_open: bool

    def __init__(self, pcm: np.ndarray, window: int = 512):
        self.frames: asyncio.Queue[Frame] = asyncio.Queue()
        self._pcm = pcm.astype(np.float32, copy=False)
        self._window = window
        self._gate_open = True
        self.dropped_blocks = 0
        self.overflows = 0

    @property
    def gate_open(self) -> bool:
        return self._gate_open

    def close_gate(self) -> None:
        self._gate_open = False

    def open_gate(self) -> None:
        self._gate_open = True

    def start(self) -> None:
        index = 0
        while index + self._window <= self._pcm.size:
            self.frames.put_nowait(
                Frame(index, self._pcm[index : index + self._window])
            )
            index += self._window
        # Append a silent tail so EOU definitely fires.
        for _ in range(40):
            self.frames.put_nowait(
                Frame(index, np.zeros(self._window, dtype=np.float32))
            )
            index += self._window

    def stop(self) -> None:
        return None

    def tail48(self, work_sample_count: int) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)
    frames: asyncio.Queue[Frame]
    loop: asyncio.AbstractEventLoop | None
    dropped_blocks: int
    overflows: int
    _gate_open: bool

    frames: asyncio.Queue[Frame]
    dropped_blocks: int
    overflows: int
    _pcm: np.ndarray
    _window: int
    _gate_open: bool
