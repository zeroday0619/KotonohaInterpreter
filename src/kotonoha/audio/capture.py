"""Microphone capture and half-duplex gating.

Capture runs at 48 kHz for two reasons:
  1. DeepFilterNet3 is 48 kHz only. Downsample to 16k first and it is unusable.
  2. Most USB microphones default to 48k, so resampling happens exactly once
     outside the kernel.

The VAD runs on 16k frames while the original 48k audio is kept in a separate
ring. When EOU fires, the utterance length in 16k samples is multiplied by three
and that many samples are pulled from the 48k ring for noise suppression. (The
streaming resampler adds a few ms of skew, which is comfortably absorbed by the
300 ms preroll.)

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

import numpy as np

from ..config import AudioCfg, VadCfg
from ..logging_setup import get_logger
from .resample import Resampler

log = get_logger(__name__)


@dataclass
class Frame:
    """One 16k frame, sized for a single VAD step."""

    index: int  # start index in 16k samples, counted since the gate opened
    pcm: np.ndarray


class RawRing:
    """Fixed-length ring holding the original 48k audio, for pulling the tail back."""

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._written = 0

    def push(self, x: np.ndarray) -> None:
        n = x.shape[0]
        if n >= self._cap:
            self._buf[:] = x[-self._cap :]
            self._written += n
            return
        pos = self._written % self._cap
        end = pos + n
        if end <= self._cap:
            self._buf[pos:end] = x
        else:
            k = self._cap - pos
            self._buf[pos:] = x[:k]
            self._buf[: end - self._cap] = x[k:]
        self._written += n

    def tail(self, n: int) -> np.ndarray:
        n = min(n, self._cap, self._written)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        pos = self._written % self._cap
        start = pos - n
        if start >= 0:
            return self._buf[start:pos].copy()
        return np.concatenate([self._buf[start:], self._buf[:pos]])

    def clear(self) -> None:
        self._written = 0
        self._buf[:] = 0.0


class MicCapture:
    def __init__(self, audio: AudioCfg, vad: VadCfg, loop: asyncio.AbstractEventLoop | None = None):
        self.audio = audio
        self.vad_cfg = vad
        self.loop = loop
        self.window16 = 512  # silero window size
        self.frames: asyncio.Queue[Frame] = asyncio.Queue(maxsize=256)

        self._raw_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=64)
        self._stream = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

        self._gate_open = threading.Event()
        self._gate_open.set()

        self._resampler = Resampler(audio.capture_sample_rate, audio.work_sample_rate)
        self._pending16 = np.zeros(0, dtype=np.float32)
        self._idx16 = 0

        ring_seconds = (vad.max_utterance_ms + vad.preroll_ms) / 1000.0 + 2.0
        self._raw48 = RawRing(int(ring_seconds * audio.capture_sample_rate))
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
        self._pending16 = np.zeros(0, dtype=np.float32)
        self._raw48.clear()
        self._resampler.reset()
        self._gate_open.set()

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        import sounddevice as sd

        self.loop = self.loop or asyncio.get_event_loop()

        def _cb(indata, frames_n, time_info, status):  # noqa: ANN001 - portaudio signature
            if status:
                self.overflows += 1
            if not self._gate_open.is_set():
                self.dropped_blocks += 1
                return
            try:
                self._raw_q.put_nowait(np.array(indata[:, 0], dtype=np.float32, copy=True))
            except queue.Full:
                self.overflows += 1

        self._stream = sd.InputStream(
            samplerate=self.audio.capture_sample_rate,
            blocksize=self.audio.capture_block_frames,
            channels=self.audio.channels,
            dtype="float32",
            device=self.audio.input_device,
            callback=_cb,
        )
        self._stream.start()
        self._worker = threading.Thread(target=self._run, name="mic-worker", daemon=True)
        self._worker.start()
        log.info(
            "mic.started",
            rate=self.audio.capture_sample_rate,
            block_ms=self.audio.capture_block_ms,
            device=str(self.audio.input_device),
        )

    def stop(self) -> None:
        self._stop.set()
        self._raw_q.put(None)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        log.info("mic.stopped", dropped_blocks=self.dropped_blocks, overflows=self.overflows)

    # -- pulling the original 48k back -----------------------------------
    def tail48(self, samples16: int) -> np.ndarray:
        ratio = self.audio.capture_sample_rate / self.audio.work_sample_rate
        return self._raw48.tail(int(samples16 * ratio))

    # -- internals -------------------------------------------------------
    def _drain_raw_queue(self) -> None:
        while True:
            try:
                self._raw_q.get_nowait()
            except queue.Empty:
                return

    def _run(self) -> None:
        while not self._stop.is_set():
            blk = self._raw_q.get()
            if blk is None:
                return
            if not self._gate_open.is_set():
                continue
            self._raw48.push(blk)
            y = self._resampler(blk)
            if y.size == 0:
                continue
            self._pending16 = (
                y if self._pending16.size == 0 else np.concatenate([self._pending16, y])
            )
            while self._pending16.size >= self.window16:
                frame = self._pending16[: self.window16]
                self._pending16 = self._pending16[self.window16 :]
                f = Frame(index=self._idx16, pcm=frame)
                self._idx16 += self.window16
                if self.loop is not None and not self.loop.is_closed():
                    self.loop.call_soon_threadsafe(self._emit, f)

    def _emit(self, f: Frame) -> None:
        try:
            self.frames.put_nowait(f)
        except asyncio.QueueFull:
            self.overflows += 1


class NullCapture:
    """Opens no device and never yields a frame.

    Used by text-only runs, where the keyboard is the input source and touching
    PortAudio would fail on hosts without a microphone.
    """

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

    def tail48(self, samples16: int) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


class FileCapture:
    """Streams a WAV file as if it were the microphone.

    Used for EOU regression tests and for development on macOS.
    """

    def __init__(self, pcm16k: np.ndarray, window: int = 512):
        self.frames: asyncio.Queue[Frame] = asyncio.Queue()
        self._pcm = pcm16k.astype(np.float32, copy=False)
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
        idx = 0
        while idx + self._window <= self._pcm.size:
            self.frames.put_nowait(Frame(idx, self._pcm[idx : idx + self._window]))
            idx += self._window
        # Append a silent tail so EOU definitely fires.
        for _ in range(40):
            self.frames.put_nowait(Frame(idx, np.zeros(self._window, dtype=np.float32)))
            idx += self._window

    def stop(self) -> None:
        return None

    def tail48(self, samples16: int) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)
