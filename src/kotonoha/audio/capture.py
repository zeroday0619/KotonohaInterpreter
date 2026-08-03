"""마이크 캡처 + 반이중 게이팅.

캡처는 48kHz 로 받는다. 이유는 두 가지다.
  1. DeepFilterNet3 는 48kHz 전용이다. 16k로 먼저 내리면 쓸 수 없다.
  2. 대부분의 USB 마이크 기본 레이트가 48k라 리샘플이 커널 밖에서 한 번만 일어난다.

VAD 는 16k 프레임으로 돌리고, 원본 48k 는 별도 링에 남겨둔다. EOU 가 나면
16k 인덱스 기준 발화 길이를 3배 해서 48k 링에서 잘라내 잡음 제거에 넘긴다.
(스트리밍 리샘플러의 몇 ms 지연이 있지만 프리롤 300ms 안에 충분히 묻힌다.)

§4 반이중 게이팅: SPEAKING 상태에서 gate 를 닫으면 들어오는 블록을 그 자리에서
버리고 리샘플러·VAD 상태까지 리셋한다. TTS 출력이 마이크로 되돌아와 새 발화로
인식되면 무한 루프가 된다 — 이 게이팅은 타협 불가다.
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
    """VAD 한 스텝 분량의 16k 프레임."""

    index: int  # 16k 샘플 기준 시작 인덱스 (게이트 열린 이후 누적)
    pcm: np.ndarray


class RawRing:
    """48k 원본을 담아두는 고정 길이 링. 발화 뒤쪽 N 샘플을 꺼내는 용도."""

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
        self.window16 = 512  # silero 창 크기
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

    # ── 게이팅 (§4) ─────────────────────────────────────────────────────
    @property
    def gate_open(self) -> bool:
        return self._gate_open.is_set()

    def close_gate(self) -> None:
        """SPEAKING 진입. 마이크 차단."""
        self._gate_open.clear()

    def open_gate(self) -> None:
        """IDLE 복귀. 큐에 남은 TTS 잔향까지 버리고 연다."""
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

    # ── 수명주기 ────────────────────────────────────────────────────────
    def start(self) -> None:
        import sounddevice as sd

        self.loop = self.loop or asyncio.get_event_loop()

        def _cb(indata, frames_n, time_info, status):  # noqa: ANN001 - portaudio 시그니처
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

    # ── 원본 48k 회수 ───────────────────────────────────────────────────
    def tail48(self, samples16: int) -> np.ndarray:
        ratio = self.audio.capture_sample_rate / self.audio.work_sample_rate
        return self._raw48.tail(int(samples16 * ratio))

    # ── 내부 ────────────────────────────────────────────────────────────
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


class FileCapture:
    """WAV 파일을 마이크처럼 흘려보낸다. EOU 회귀 테스트와 macOS 개발용."""

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
        # 무음 꼬리를 붙여 EOU 가 확실히 발동하게 한다
        for _ in range(40):
            self.frames.put_nowait(Frame(idx, np.zeros(self._window, dtype=np.float32)))
            idx += self._window

    def stop(self) -> None:
        return None

    def tail48(self, samples16: int) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)
