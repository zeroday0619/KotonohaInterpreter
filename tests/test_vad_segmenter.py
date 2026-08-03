"""Regression tests for §5.1 preroll and for EOU.

Preroll is a non-negotiable item, so a regression has to surface immediately.
A clipped first syllable looks like an ASR quality problem, and tracking that
down takes days.
"""

from __future__ import annotations

import math

import numpy as np

from kotonoha.audio.vad import SILERO_WINDOW, UtteranceSegmenter

FRAME_MS = 1000.0 * SILERO_WINDOW / 16000  # 32 ms


class ScriptedVad:
    """Fake VAD that reports the frame's first sample as the probability."""

    name = "scripted"
    window = SILERO_WINDOW

    def reset(self) -> None:
        return None

    def prob(self, frame: np.ndarray) -> float:
        return float(frame[0])


def frame(value: float) -> np.ndarray:
    return np.full(SILERO_WINDOW, value, dtype=np.float32)


def make(**kw) -> UtteranceSegmenter:
    opts = dict(
        vad=ScriptedVad(),
        preroll_ms=300,
        min_speech_ms=60,
        silence_ms=800,
        threshold=0.5,
        neg_threshold=0.35,
    )
    opts.update(kw)
    return UtteranceSegmenter(**opts)


def test_preroll_is_prepended():
    seg = make()
    n_pre = math.ceil(300 / FRAME_MS)  # 10 frames = 320 ms >= 300 ms

    # 20 frames of silence before speech; the ring should hold only the last few.
    for _ in range(20):
        assert seg.feed(frame(0.1)).kind == "none"

    ev = seg.feed(frame(0.9))
    assert ev.kind == "speech_start"

    for _ in range(9):
        seg.feed(frame(0.9))

    end = None
    for _ in range(int(800 / FRAME_MS) + 2):
        e = seg.feed(frame(0.0))
        if e.kind == "speech_end":
            end = e
            break

    assert end is not None, "EOU never fired"
    u = end.utterance
    assert u is not None

    # The preroll frames (0.1) must actually be in front of the utterance.
    head = u.pcm[: n_pre * SILERO_WINDOW]
    assert np.allclose(head, 0.1), "preroll was dropped — the first syllable gets clipped"
    assert 200 <= u.preroll_ms <= 320


def test_preroll_never_exceeds_configured_window():
    seg = make(preroll_ms=200)
    for _ in range(100):
        seg.feed(frame(0.1))
    seg.feed(frame(0.9))
    assert seg._preroll_used_ms <= 200 + FRAME_MS


def test_eou_fires_at_configured_silence():
    seg = make(silence_ms=800)
    seg.feed(frame(0.9))
    for _ in range(6):
        seg.feed(frame(0.9))

    silent = 0
    while True:
        e = seg.feed(frame(0.0))
        silent += 1
        if e.kind == "speech_end":
            break
        assert silent < 100

    observed = silent * FRAME_MS
    assert 800 <= observed <= 800 + 2 * FRAME_MS, observed


def test_short_noise_is_not_an_utterance():
    """A single cough must not start a turn."""
    seg = make(min_speech_ms=200)
    seg.feed(frame(0.9))  # one 32 ms burst
    for _ in range(int(800 / FRAME_MS) + 2):
        e = seg.feed(frame(0.0))
        if e.kind == "speech_end":
            raise AssertionError("a short noise was accepted as an utterance")
    assert seg.state.value == "idle"


def test_max_utterance_cuts():
    seg = make(max_utterance_ms=320)
    seg.feed(frame(0.9))
    end = None
    for _ in range(30):
        e = seg.feed(frame(0.9))
        if e.kind == "speech_end":
            end = e
            break
    assert end is not None and end.utterance.ended_by == "max_len"


def test_ptt_keeps_preroll():
    """Preroll must survive in PTT mode too — people speak before the key."""
    seg = make()
    for _ in range(20):
        seg.prime_preroll(frame(0.1))
    ev = seg.force_start()
    assert ev.kind == "speech_start"
    for _ in range(5):
        seg.feed(frame(0.9))
    end = seg.force_end()
    assert end.utterance is not None
    assert np.isclose(end.utterance.pcm[0], 0.1), "preroll was lost on the PTT path"
    assert end.utterance.ended_by == "manual"
