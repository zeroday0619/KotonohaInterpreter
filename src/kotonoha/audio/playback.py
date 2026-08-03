"""TTS 재생 큐.

절 단위로 도착하는 오디오 청크를 끊김 없이 이어 붙인다(§5.4). 계측에 필요한
'첫 오디오 패킷' 시각과 '큐 소진' 시각은 여기서 나온다 — 오케스트레이터가
enqueue 한 시각이 아니라, 실제로 스피커로 나간 시각이어야 의미가 있다.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque

import numpy as np

from ..config import AudioCfg, TtsCfg
from ..logging_setup import get_logger
from .resample import Resampler

log = get_logger(__name__)


class Playback:
    def __init__(self, audio: AudioCfg, tts: TtsCfg):
        self.audio = audio
        self.tts = tts
        self._q: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._cur: np.ndarray | None = None
        self._pos = 0
        self._stream = None
        self._resampler = Resampler(tts.sample_rate, audio.playback_sample_rate)

        self._loop: asyncio.AbstractEventLoop | None = None
        self.first_packet = asyncio.Event()
        self.drained = asyncio.Event()
        self.drained.set()
        self._closing = False

    # ── 수명주기 ────────────────────────────────────────────────────────
    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        import sounddevice as sd

        self._loop = loop or asyncio.get_event_loop()

        def _cb(outdata, frames_n, time_info, status):  # noqa: ANN001 - portaudio 시그니처
            if status:
                log.debug("playback.status", status=str(status))
            written = 0
            while written < frames_n:
                if self._cur is None or self._pos >= self._cur.size:
                    with self._lock:
                        self._cur = self._q.popleft() if self._q else None
                    self._pos = 0
                    if self._cur is None:
                        outdata[written:, 0] = 0.0
                        if written > 0 or not self.drained.is_set():
                            self._signal_drained()
                        return
                take = min(frames_n - written, self._cur.size - self._pos)
                outdata[written : written + take, 0] = self._cur[self._pos : self._pos + take]
                self._pos += take
                written += take
                if not self.first_packet.is_set():
                    self._signal_first()

        self._stream = sd.OutputStream(
            samplerate=self.audio.playback_sample_rate,
            channels=1,
            dtype="float32",
            device=self.audio.output_device,
            callback=_cb,
        )
        self._stream.start()
        log.info("playback.started", rate=self.audio.playback_sample_rate)

    def stop(self) -> None:
        self._closing = True
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ── 큐 ──────────────────────────────────────────────────────────────
    def begin_turn(self) -> None:
        """새 턴 시작 — 계측 이벤트를 되돌린다."""
        self.first_packet.clear()
        self.drained.clear()

    def enqueue(self, pcm: np.ndarray, rate: int | None = None) -> None:
        """TTS 청크 투입. rate 를 주면 그 레이트에서 출력 레이트로 맞춘다."""
        x = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        src = rate or self.tts.sample_rate
        if src != self.audio.playback_sample_rate:
            r = (
                self._resampler
                if src == self.tts.sample_rate
                else Resampler(src, self.audio.playback_sample_rate)
            )
            x = r(x)
        with self._lock:
            self._q.append(x)
        self.drained.clear()

    def flush(self) -> None:
        """재생 중단(취소·오류). 큐를 비운다."""
        with self._lock:
            self._q.clear()
        self._cur = None
        self._pos = 0
        self._signal_drained()

    @property
    def pending_seconds(self) -> float:
        with self._lock:
            n = sum(a.size for a in self._q)
        if self._cur is not None:
            n += max(0, self._cur.size - self._pos)
        return n / float(self.audio.playback_sample_rate)

    async def wait_drained(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self.drained.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ── 콜백 스레드 → 이벤트 루프 ───────────────────────────────────────
    def _signal_first(self) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.first_packet.set)

    def _signal_drained(self) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.drained.set)


class NullPlayback(Playback):
    """오디오 장치 없는 환경(CI, 원격 셸)용."""

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_event_loop()
        log.warning("playback.null", reason="no output device")

    def stop(self) -> None:
        return None

    def enqueue(self, pcm: np.ndarray, rate: int | None = None) -> None:
        if not self.first_packet.is_set():
            self._signal_first()
        self._signal_drained()
