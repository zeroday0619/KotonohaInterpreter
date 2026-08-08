"""Microphone capture and half-duplex gating.

Capture prefers 48 kHz for two reasons:
  1. DeepFilterNet3 is 48 kHz only. Downsample to 16k first and it is unusable.
  2. Most USB microphones default to 48k, so resampling happens exactly once
     outside the kernel.

The selected device can reject 48 kHz. In that case PortAudio opens the device at
its native rate and the capture path resamples to 16 kHz for VAD. The original-rate
audio remains in a separate ring and is converted to 48 kHz only when DeepFilterNet
needs it. This avoids losing the microphone entirely because an ALSA endpoint does
not expose the configured rate.

Half-duplex gating (§4): closing the gate in SPEAKING drops incoming blocks on
the spot and resets the resampler and VAD state. If TTS output leaks back into
the microphone and is heard as a new utterance, the result is an infinite loop —
this gate is not negotiable.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
from dataclasses import dataclass
from typing import Any, ClassVar, Final

import numpy as np

from kotonoha._config import AudioConfig, VadConfig
from kotonoha._logging_setup import get_logger
from kotonoha._typing import override
from kotonoha.audio._devices import resolve_audio_stream, select_mono_input
from kotonoha.audio._resample import Resampler

log = get_logger(__name__)
FRAME_QUEUE_CAPACITY: Final[int] = 256


@dataclass(slots=True)
class Frame:
    """One 16k frame, sized for a single VAD step."""

    index: int  # start index in 16k samples, counted since the gate opened
    pcm: np.ndarray


class RawRing:
    """Fixed-length ring holding the original 48k audio, for pulling the tail back."""
    __slots__: ClassVar[tuple[str, ...]] = (
        "_buffer",
        "_capacity",
        "_lock",
        "_written",
    )

    _buffer: np.ndarray
    _capacity: int
    _written: int
    _lock: threading.Lock

    @override
    def __init__(
        self,
        /,
        capacity: int,
    ) -> None:
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._written = 0
        self._lock = threading.Lock()

    def push(
        self,
        /,
        samples: np.ndarray,
    ) -> None:
        with self._lock:
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

    def tail(
        self,
        /,
        sample_count: int,
    ) -> np.ndarray:
        with self._lock:
            sample_count = min(sample_count, self._capacity, self._written)
            if sample_count == 0:
                return np.zeros(0, dtype=np.float32)
            position = self._written % self._capacity
            start = position - sample_count
            if start >= 0:
                return self._buffer[start:position].copy()
            output = np.empty(sample_count, dtype=np.float32)
            first_part_length = -start
            output[:first_part_length] = self._buffer[start:]
            output[first_part_length:] = self._buffer[:position]
            return output

    def clear(
        self,
        /,
    ) -> None:
        with self._lock:
            self._written = 0


class MicCapture:
    __slots__: ClassVar[tuple[str, ...]] = (
        "_gate_open",
        "_capture_channels",
        "_capture_device",
        "_capture_sample_rate",
        "_emit_slots",
        "_frame_buffer",
        "_frame_fill",
        "_gate_generation",
        "_processing_lock",
        "_raw_capture",
        "_raw_queue",
        "_ring_seconds",
        "_resampler",
        "_sample_index",
        "_stop_event",
        "_stream",
        "_window_size",
        "_worker_thread",
        "audio",
        "dropped_blocks",
        "frames",
        "loop",
        "overflows",
    )
    audio: AudioConfig
    loop: asyncio.AbstractEventLoop | None
    frames: asyncio.Queue[Frame]
    dropped_blocks: int
    overflows: int
    _raw_queue: queue.Queue[tuple[int, np.ndarray] | None]
    _stream: Any | None
    _worker_thread: threading.Thread | None
    _stop_event: threading.Event
    _gate_open: threading.Event
    _capture_channels: int
    _capture_device: int | None
    _capture_sample_rate: int
    _emit_slots: threading.BoundedSemaphore
    _resampler: Resampler
    _frame_buffer: np.ndarray
    _frame_fill: int
    _gate_generation: int
    _processing_lock: threading.Lock
    _sample_index: int
    _raw_capture: RawRing
    _ring_seconds: float
    _window_size: int

    @override
    def __init__(
        self,
        /,
        audio: AudioConfig,
        vad: VadConfig,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.audio = audio
        self.loop = loop
        self._window_size = 512
        self.frames = asyncio.Queue(maxsize=FRAME_QUEUE_CAPACITY)

        self._raw_queue = queue.Queue(maxsize=64)
        self._emit_slots = threading.BoundedSemaphore(self.frames.maxsize)
        self._stream = None
        self._worker_thread = None
        self._stop_event = threading.Event()

        self._gate_open = threading.Event()
        self._gate_open.set()
        self._gate_generation = 0

        self._capture_sample_rate = audio.capture_sample_rate
        self._capture_channels = audio.channels
        self._capture_device = None
        self._resampler = Resampler(self._capture_sample_rate, audio.work_sample_rate)
        self._frame_buffer = np.empty(self._window_size, dtype=np.float32)
        self._frame_fill = 0
        self._processing_lock = threading.Lock()
        self._sample_index = 0

        self._ring_seconds = (vad.max_utterance_ms + vad.preroll_ms) / 1000.0 + 2.0
        self._raw_capture = RawRing(int(self._ring_seconds * self._capture_sample_rate))
        self.dropped_blocks = 0
        self.overflows = 0

    # -- gating (§4) -----------------------------------------------------
    @property
    def gate_open(
        self,
        /,
    ) -> bool:
        return self._gate_open.is_set()

    def close_gate(
        self,
        /,
    ) -> None:
        """Entering SPEAKING. Microphone shut."""
        self._gate_open.clear()
        # Fence the worker after clearing the gate. A block that passed the
        # callback check before this transition must not survive into the next turn.
        with self._processing_lock:
            self._gate_generation += 1
        self._drain_raw_queue()

    def open_gate(
        self,
        /,
    ) -> None:
        """Back to IDLE. Discard any TTS tail still queued, then open."""
        self._gate_open.clear()
        with self._processing_lock:
            self._gate_generation += 1
            self._drain_raw_queue()
            while True:
                try:
                    self.frames.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._frame_buffer = np.empty(self._window_size, dtype=np.float32)
            self._frame_fill = 0
            self._sample_index = 0
            self._raw_capture.clear()
            self._resampler.reset()
        self._gate_open.set()

    # -- lifecycle -------------------------------------------------------
    def start(
        self,
        /,
    ) -> None:
        import sounddevice as sd

        if self._stream is not None:
            return
        worker_thread = self._worker_thread
        if worker_thread is not None:
            if worker_thread.is_alive():
                raise RuntimeError("previous microphone worker is still stopping")
            self._worker_thread = None
        self.loop = self.loop or asyncio.get_event_loop()
        self._stop_event.clear()
        self._emit_slots = threading.BoundedSemaphore(self.frames.maxsize)
        self.open_gate()
        stream_settings = resolve_audio_stream(
            self.audio.input_device,
            "input",
            requested_sample_rate=self.audio.capture_sample_rate,
            requested_channels=self.audio.channels,
        )
        self._capture_sample_rate = stream_settings.sample_rate
        self._capture_channels = stream_settings.channels
        self._capture_device = stream_settings.device_index
        self._resampler = Resampler(
            self._capture_sample_rate,
            self.audio.work_sample_rate,
        )
        self._raw_capture = RawRing(int(self._ring_seconds * self._capture_sample_rate))
        block_frames = int(
            self._capture_sample_rate * self.audio.capture_block_ms / 1000
        )

        def capture_callback(
            input_data: Any,
            /,
            frame_count: Any,
            time_info: Any,
            status: Any,
        ) -> None:
            if status:
                self.overflows += 1
            if not self._gate_open.is_set():
                self.dropped_blocks += 1
                return
            generation = self._gate_generation
            try:
                self._raw_queue.put_nowait(
                    (generation, np.array(input_data, dtype=np.float32, copy=True))
                )
            except queue.Full:
                self.overflows += 1

        stream = sd.InputStream(
            samplerate=self._capture_sample_rate,
            blocksize=block_frames,
            channels=self._capture_channels,
            dtype="float32",
            device=self._capture_device,
            callback=capture_callback,
        )
        try:
            stream.start()
        except Exception:
            try:
                stream.close()
            except Exception as close_error:  # noqa: BLE001
                log.warning("mic.close_failed", error=repr(close_error))
            raise
        worker_thread = threading.Thread(
            target=self._run,
            name="mic-worker",
            daemon=True,
        )
        try:
            worker_thread.start()
        except Exception:
            try:
                stream.stop()
            except Exception as stop_error:  # noqa: BLE001
                log.warning("mic.stop_failed", error=repr(stop_error))
            try:
                stream.close()
            except Exception as close_error:  # noqa: BLE001
                log.warning("mic.close_failed", error=repr(close_error))
            raise
        self._stream = stream
        self._worker_thread = worker_thread
        log.info(
            "mic.started",
            rate=self._capture_sample_rate,
            requested_rate=self.audio.capture_sample_rate,
            channels=self._capture_channels,
            block_ms=self.audio.capture_block_ms,
            device=stream_settings.selector,
        )

    def stop(
        self,
        /,
    ) -> None:
        self._gate_open.clear()
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception as error:  # noqa: BLE001
                log.warning("mic.stop_failed", error=repr(error))
            try:
                stream.close()
            except Exception as error:  # noqa: BLE001
                log.warning("mic.close_failed", error=repr(error))
        self._stop_event.set()
        while True:
            try:
                self._raw_queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    self._raw_queue.get_nowait()
                except queue.Empty:
                    continue
        worker_thread = self._worker_thread
        if worker_thread is not None:
            worker_thread.join(timeout=2.0)
            if worker_thread.is_alive():
                log.warning("mic.worker_stop_timeout")
            else:
                self._worker_thread = None
        if self._worker_thread is None:
            self._drain_raw_queue()
        log.info("mic.stopped", dropped_blocks=self.dropped_blocks, overflows=self.overflows)

    # -- pulling the original 48k back -----------------------------------
    def tail48(
        self,
        /,
        work_sample_count: int,
    ) -> np.ndarray:
        ratio = self._capture_sample_rate / self.audio.work_sample_rate
        raw_audio = self._raw_capture.tail(int(work_sample_count * ratio))
        if self._capture_sample_rate == 48000:
            return raw_audio
        from kotonoha.audio._resample import resample_once

        return resample_once(raw_audio, self._capture_sample_rate, 48000)

    # -- internals -------------------------------------------------------
    def _drain_raw_queue(
        self,
        /,
    ) -> None:
        while True:
            try:
                self._raw_queue.get_nowait()
            except queue.Empty:
                return

    def _run(
        self,
        /,
    ) -> None:
        try:
            while not self._stop_event.is_set():
                queued_block = self._raw_queue.get()
                if queued_block is None:
                    return
                generation, block = queued_block
                if not self._gate_open.is_set():
                    continue
                with self._processing_lock:
                    if (
                        not self._gate_open.is_set()
                        or generation != self._gate_generation
                    ):
                        continue
                    mono_block = select_mono_input(block)
                    self._raw_capture.push(mono_block)
                    resampled = self._resampler(mono_block)
                    offset = 0
                    while offset < resampled.size:
                        copy_count = min(
                            self._window_size - self._frame_fill,
                            resampled.size - offset,
                        )
                        self._frame_buffer[
                            self._frame_fill : self._frame_fill + copy_count
                        ] = resampled[offset : offset + copy_count]
                        self._frame_fill += copy_count
                        offset += copy_count
                        if self._frame_fill != self._window_size:
                            continue
                        output_frame = Frame(
                            index=self._sample_index,
                            pcm=self._frame_buffer,
                        )
                        self._sample_index += self._window_size
                        self._frame_buffer = np.empty(
                            self._window_size,
                            dtype=np.float32,
                        )
                        self._frame_fill = 0
                        self._schedule_emit(output_frame, generation)
        except Exception as error:  # noqa: BLE001
            self._stop_event.set()
            log.exception("mic.worker_failed", error=repr(error))

    def _schedule_emit(
        self,
        /,
        frame: Frame,
        generation: int,
    ) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        if not self._emit_slots.acquire(blocking=False):
            self.overflows += 1
            return
        emit_slots = self._emit_slots
        try:
            loop.call_soon_threadsafe(
                self._emit_scheduled,
                frame,
                generation,
                emit_slots,
            )
        except RuntimeError:
            emit_slots.release()

    def _emit_scheduled(
        self,
        /,
        frame: Frame,
        generation: int,
        emit_slots: threading.BoundedSemaphore,
    ) -> None:
        try:
            self._emit(frame, generation)
        finally:
            emit_slots.release()

    def _emit(
        self,
        /,
        frame: Frame,
        generation: int,
    ) -> None:
        if not self._gate_open.is_set() or generation != self._gate_generation:
            return
        try:
            self.frames.put_nowait(frame)
        except asyncio.QueueFull:
            self.overflows += 1


class NullCapture:
    """Opens no device and never yields a frame.

    Used by text-only runs, where the keyboard is the input source and touching
    PortAudio would fail on hosts without a microphone.
    """
    __slots__: ClassVar[tuple[str, ...]] = (
        "_gate_open",
        "dropped_blocks",
        "frames",
        "loop",
        "overflows",
    )

    frames: asyncio.Queue[Frame]
    loop: asyncio.AbstractEventLoop | None
    dropped_blocks: int
    overflows: int
    _gate_open: bool

    @override
    def __init__(
        self,
        /,
        **_ignored: Any,
    ) -> None:
        self.frames: asyncio.Queue[Frame] = asyncio.Queue(maxsize=1)
        self.loop: asyncio.AbstractEventLoop | None = None
        self._gate_open = False
        self.dropped_blocks = 0
        self.overflows = 0

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
        self._gate_open = False

    def open_gate(
        self,
        /,
    ) -> None:
        self._gate_open = False  # there is nothing to open

    def start(
        self,
        /,
    ) -> None:
        return None

    def stop(
        self,
        /,
    ) -> None:
        return None

    def tail48(
        self,
        /,
        work_sample_count: int,
    ) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


class FileCapture:
    """Streams a WAV file as if it were the microphone.

    Used for EOU regression tests and for development on macOS.
    """
    __slots__: ClassVar[tuple[str, ...]] = (
        "_gate_open",
        "_pcm",
        "_producer_future",
        "_window",
        "dropped_blocks",
        "frames",
        "loop",
        "overflows",
    )

    frames: asyncio.Queue[Frame]
    loop: asyncio.AbstractEventLoop | None
    dropped_blocks: int
    overflows: int
    _pcm: np.ndarray
    _producer_future: concurrent.futures.Future[None] | None
    _window: int
    _gate_open: bool

    @override
    def __init__(
        self,
        /,
        pcm: np.ndarray,
        window: int = 512,
    ) -> None:
        self.frames = asyncio.Queue(maxsize=FRAME_QUEUE_CAPACITY)
        self.loop = None
        self._pcm = pcm.astype(np.float32, copy=False)
        self._producer_future = None
        self._window = window
        self._gate_open = True
        self.dropped_blocks = 0
        self.overflows = 0

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
        self._gate_open = False

    def open_gate(
        self,
        /,
    ) -> None:
        self._gate_open = True

    def start(
        self,
        /,
    ) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            raise RuntimeError("file capture requires a running event loop")
        if self._producer_future is not None and not self._producer_future.done():
            return
        self._producer_future = asyncio.run_coroutine_threadsafe(
            self._produce(),
            loop,
        )

    async def _produce(
        self,
        /,
    ) -> None:
        index = 0
        while index + self._window <= self._pcm.size:
            await self.frames.put(
                Frame(
                    index,
                    self._pcm[index : index + self._window],
                )
            )
            index += self._window
        # Append a silent tail so EOU definitely fires.
        silent_frame = np.zeros(self._window, dtype=np.float32)
        for _ in range(40):
            await self.frames.put(
                Frame(
                    index,
                    silent_frame,
                )
            )
            index += self._window

    def stop(
        self,
        /,
    ) -> None:
        producer_future = self._producer_future
        self._producer_future = None
        if producer_future is None:
            return
        producer_future.cancel()
        try:
            producer_future.result(timeout=1.0)
        except concurrent.futures.CancelledError:
            pass
        except concurrent.futures.TimeoutError:
            log.warning("file_capture.stop_timeout")
        except Exception as error:  # noqa: BLE001
            log.warning("file_capture.producer_failed", error=repr(error))

    def tail48(
        self,
        /,
        work_sample_count: int,
    ) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)
