"""Silero VAD, preroll, and end-of-utterance detection.

§5.1 is non-negotiable. Without 200-300 ms of preroll, the Korean tense-stop
onsets and the short pause before a Japanese sokuon get cut off entirely, and
the symptom looks exactly like an ASR quality problem. So the preroll buffer
always holds the frames from *before* the VAD reports speech, and prepends them.

The VAD itself is a pure state machine with no I/O. That means EOU misbehaviour
can be reproduced and regression-tested from WAV files with no microphone —
which is precisely the Phase 1 goal.
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

SILERO_WINDOW = 512  # 32 ms at 16 kHz; silero v5 requires exactly this size.


class VadModel(Protocol):
    name: str
    window: int

    def prob(self, frame: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class SileroVadOnnx:
    """Runs silero-vad directly through onnxruntime on CPU, with no torch dependency."""

    name = "silero_onnx"
    window = SILERO_WINDOW

    def __init__(self, model_path: Path, sample_rate: int = 16000, threads: int = 1):
        import onnxruntime as ort  # type: ignore[import-not-found]

        if not model_path.exists():
            raise FileNotFoundError(
                f"silero_vad.onnx missing: {model_path}\n"
                f"  fetch it with scripts/fetch_models.sh"
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
        self._v5 = "state" in names  # v5 takes `state`; v4 takes `h` and `c`.
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
            # Zero-pad a short trailing frame.
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
    """Development-machine fallback, so the pipeline can run without onnxruntime.

    Never use this on the device — it is far too noise-sensitive and EOU falls apart.
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
        # Map linearly: 0 at floor_db, 1 at floor_db + 25 dB.
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


# -- segmentation --------------------------------------------------------


class SegState(str, Enum):
    IDLE = "idle"
    SPEECH = "speech"
    TRAILING = "trailing"  # in speech, counting silence towards EOU


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
    """Pure state machine: feed it frames, get utterances back."""

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
        # Round up: preroll is a *minimum*, and frame quantisation must not drag
        # it below 200 ms. The +1 is for the frame that triggered speech onset —
        # that one belongs to the utterance, not to the preroll.
        n_preroll = max(1, math.ceil(self.preroll_ms / self.frame_ms)) + 1
        self._preroll = deque(maxlen=n_preroll)

    # -- public API ------------------------------------------------------
    def reset(self) -> None:
        self.state = SegState.IDLE
        self._preroll.clear()
        self._buf.clear()
        self._speech_ms = 0.0
        self._silence_ms_acc = 0.0
        self._preroll_used_ms = 0.0
        self.vad.reset()

    def prime_preroll(self, frame: np.ndarray) -> None:
        """Keep the preroll ring filled while idle in push-to-talk mode.

        Preroll matters even with PTT: people start speaking at the same moment
        they press the key, sometimes a little before.
        """
        if self.state is SegState.IDLE:
            self._preroll.append(frame)

    def feed(self, frame: np.ndarray) -> SegEvent:
        """One 16 kHz float32 frame of length vad.window."""
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
        """Push-to-talk key released."""
        if self.state is SegState.IDLE:
            return SegEvent("none")
        return self._finish("manual", 0.0)

    def force_start(self) -> SegEvent:
        """Push-to-talk key pressed. The preroll is kept and used as-is."""
        if self.state is not SegState.IDLE:
            return SegEvent("none")
        self._start(None)
        return SegEvent("speech_start", 1.0)

    # -- internals -------------------------------------------------------
    def _start(self, first_frame: np.ndarray | None) -> None:
        # §5.1 preroll: prepend the frames from *before* the VAD reacted.
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
            # A cough, a door closing. Not an utterance.
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
