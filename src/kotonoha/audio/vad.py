"""Silero VAD + 프리롤 + 발화 종료(EOU) 판정.

§5.1 은 타협 불가다. 프리롤 200~300ms 가 없으면 한국어 초성 파열음(ㅃ/ㄲ/ㅌ)과
일본어 촉음(っ) 앞의 짧은 무음 구간이 통째로 잘려 나가고, 그러면 ASR 품질 문제로
오진하게 된다. 그래서 프리롤 버퍼는 VAD가 발화 시작을 알리기 *이전* 프레임을
항상 들고 있다가 발화 앞에 붙인다.

VAD 자체는 순수 상태 기계로 두었다(I/O 없음). 덕분에 마이크 없이 WAV 파일만으로도
EOU 오작동 패턴을 재현·회귀 테스트할 수 있다 — Phase 1의 목표가 바로 그것이다.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np

from ..logging_setup import get_logger

log = get_logger(__name__)

SILERO_WINDOW = 512  # 16kHz 기준 32ms. silero v5 는 이 크기를 강제한다.


class VadModel(Protocol):
    name: str
    window: int

    def prob(self, frame: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class SileroVadOnnx:
    """onnxruntime CPU 로 silero-vad 를 직접 돌린다 (torch 의존 없음)."""

    name = "silero_onnx"
    window = SILERO_WINDOW

    def __init__(self, model_path: Path, sample_rate: int = 16000, threads: int = 1):
        import onnxruntime as ort  # type: ignore[import-not-found]

        if not model_path.exists():
            raise FileNotFoundError(
                f"silero_vad.onnx 없음: {model_path}\n"
                f"  scripts/fetch_models.sh 를 실행해 받아둘 것"
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = threads
        opts.intra_op_num_threads = threads
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.sample_rate = sample_rate
        names = {i.name for i in self.sess.get_inputs()}
        self._v5 = "state" in names  # v5: state / v4: h,c
        self.reset()
        log.info("vad.loaded", backend=self.name, v5=self._v5, path=str(model_path))

    def reset(self) -> None:
        if self._v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        x = frame.astype(np.float32, copy=False).reshape(1, -1)
        if x.shape[1] != self.window:
            # 마지막 자투리는 0 패딩
            pad = np.zeros((1, self.window), dtype=np.float32)
            pad[0, : x.shape[1]] = x[0, : self.window]
            x = pad
        sr = np.array(self.sample_rate, dtype=np.int64)
        if self._v5:
            out, self._state = self.sess.run(None, {"input": x, "state": self._state, "sr": sr})
        else:
            out, self._h, self._c = self.sess.run(
                None, {"input": x, "h": self._h, "c": self._c, "sr": sr}
            )
        return float(np.asarray(out).reshape(-1)[0])


class EnergyVad:
    """개발 PC 폴백. onnxruntime 없이 파이프라인을 굴려보기 위한 것.

    실기에서는 절대 쓰지 않는다 — 잡음에 취약해 EOU 가 엉망이 된다.
    """

    name = "energy"
    window = SILERO_WINDOW

    def __init__(self, floor_db: float = -45.0):
        self.floor_db = floor_db

    def reset(self) -> None:
        return None

    def prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))
        db = 20.0 * np.log10(rms + 1e-12)
        # floor_db 에서 0, floor+25dB 에서 1 로 선형 매핑
        return float(np.clip((db - self.floor_db) / 25.0, 0.0, 1.0))


def build_vad(backend: str, model_path: Path, sample_rate: int = 16000) -> VadModel:
    if backend == "energy":
        log.warning("vad.energy_fallback", reason="explicitly configured")
        return EnergyVad()
    try:
        return SileroVadOnnx(model_path, sample_rate=sample_rate)
    except Exception as e:  # noqa: BLE001
        log.error("vad.silero_unavailable", error=repr(e), fallback="energy")
        return EnergyVad()


# ── 발화 절단 ────────────────────────────────────────────────────────────


class SegState(str, Enum):
    IDLE = "idle"
    SPEECH = "speech"
    TRAILING = "trailing"  # 발화 중 침묵 카운트 중


@dataclass
class Utterance:
    pcm: np.ndarray
    sample_rate: int
    speech_ms: float
    preroll_ms: float
    ended_by: str  # silence | max_len | manual


@dataclass
class SegEvent:
    kind: str  # speech_start | speech_end | none
    prob: float = 0.0
    utterance: Utterance | None = None


@dataclass
class UtteranceSegmenter:
    """프레임 단위로 먹여주면 발화를 잘라 돌려주는 순수 상태 기계."""

    vad: VadModel
    sample_rate: int = 16000
    threshold: float = 0.5
    neg_threshold: float = 0.35
    preroll_ms: int = 300
    min_speech_ms: int = 120
    silence_ms: int = 800
    max_utterance_ms: int = 30000

    state: SegState = SegState.IDLE
    _preroll: deque = field(default_factory=deque, init=False)
    _buf: list = field(default_factory=list, init=False)
    _speech_ms: float = 0.0
    _silence_ms_acc: float = 0.0
    _preroll_used_ms: float = 0.0

    def __post_init__(self) -> None:
        self.frame_ms = 1000.0 * self.vad.window / self.sample_rate
        # 올림한다. 프리롤은 '최소' 요구사항이라 프레임 양자화로 200ms 아래로
        # 내려가면 안 된다. +1 은 발화를 촉발한 프레임이 링의 한 칸을 쓰기 때문 —
        # 그 프레임은 프리롤이 아니라 발화의 일부다.
        n_preroll = max(1, math.ceil(self.preroll_ms / self.frame_ms)) + 1
        self._preroll = deque(maxlen=n_preroll)

    # ── 공개 API ───────────────────────────────────────────────────────
    def reset(self) -> None:
        self.state = SegState.IDLE
        self._preroll.clear()
        self._buf.clear()
        self._speech_ms = 0.0
        self._silence_ms_acc = 0.0
        self._preroll_used_ms = 0.0
        self.vad.reset()

    def prime_preroll(self, frame: np.ndarray) -> None:
        """push-to-talk 모드에서 IDLE 동안 프리롤 링만 채운다.

        PTT 라도 프리롤은 필요하다. 사람은 키를 누르는 것과 거의 동시에,
        때로는 조금 먼저 말하기 시작한다.
        """
        if self.state is SegState.IDLE:
            self._preroll.append(frame)

    def feed(self, frame: np.ndarray) -> SegEvent:
        """길이 vad.window 의 16k float32 프레임 하나."""
        p = self.vad.prob(frame)
        speaking = p >= self.threshold if self.state is SegState.IDLE else p >= self.neg_threshold

        if self.state is SegState.IDLE:
            self._preroll.append(frame)
            if speaking:
                self._start(frame)
                return SegEvent("speech_start", p)
            return SegEvent("none", p)

        # SPEECH / TRAILING
        self._buf.append(frame)
        total_ms = len(self._buf) * self.frame_ms

        if speaking:
            self.state = SegState.SPEECH
            self._speech_ms += self.frame_ms
            self._silence_ms_acc = 0.0
        else:
            self.state = SegState.TRAILING
            self._silence_ms_acc += self.frame_ms
            if self._silence_ms_acc >= self.silence_ms:
                return self._finish("silence", p)

        if total_ms >= self.max_utterance_ms:
            return self._finish("max_len", p)

        return SegEvent("none", p)

    def force_end(self) -> SegEvent:
        """push-to-talk 에서 키를 뗐을 때."""
        if self.state is SegState.IDLE:
            return SegEvent("none")
        return self._finish("manual", 0.0)

    def force_start(self) -> SegEvent:
        """push-to-talk 에서 키를 눌렀을 때. 프리롤은 그대로 살려 쓴다."""
        if self.state is not SegState.IDLE:
            return SegEvent("none")
        self._start(None)
        return SegEvent("speech_start", 1.0)

    # ── 내부 ───────────────────────────────────────────────────────────
    def _start(self, first_frame: np.ndarray | None) -> None:
        # §5.1 프리롤: VAD 가 반응하기 *전* 프레임을 발화 앞에 붙인다.
        pre = list(self._preroll)
        if first_frame is not None and pre and pre[-1] is first_frame:
            pre = pre[:-1]
        self._buf = pre + ([first_frame] if first_frame is not None else [])
        self._preroll_used_ms = len(pre) * self.frame_ms
        self._preroll.clear()
        self._speech_ms = self.frame_ms if first_frame is not None else 0.0
        self._silence_ms_acc = 0.0
        self.state = SegState.SPEECH

    def _finish(self, reason: str, p: float) -> SegEvent:
        pcm = (
            np.concatenate(self._buf).astype(np.float32, copy=False)
            if self._buf
            else np.zeros(0, dtype=np.float32)
        )
        speech_ms = self._speech_ms
        preroll_ms = self._preroll_used_ms
        self.reset()

        if speech_ms < self.min_speech_ms and reason != "manual":
            # 기침·문 닫는 소리 같은 짧은 잡음. 발화로 세지 않는다.
            log.debug("vad.too_short", speech_ms=round(speech_ms, 1))
            return SegEvent("none", p)

        return SegEvent(
            "speech_end",
            p,
            Utterance(
                pcm=pcm,
                sample_rate=self.sample_rate,
                speech_ms=speech_ms,
                preroll_ms=preroll_ms,
                ended_by=reason,
            ),
        )
