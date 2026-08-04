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


class ScriptedVoiceActivityDetector:
    """Fake VAD that reports the frame's first sample as the probability."""

    name = "scripted"
    window = SILERO_WINDOW

    def reset(self) -> None:
        return None

    def probability(self, frame: np.ndarray) -> float:
        return float(frame[0])


def frame(value: float) -> np.ndarray:
    return np.full(SILERO_WINDOW, value, dtype=np.float32)


def create_segmenter(**overrides) -> UtteranceSegmenter:
    options = dict(
        vad=ScriptedVoiceActivityDetector(),
        preroll_ms=300,
        min_speech_ms=60,
        silence_ms=800,
        threshold=0.5,
        neg_threshold=0.35,
    )
    options.update(overrides)
    return UtteranceSegmenter(**options)


def test_preroll_is_prepended():
    segmenter = create_segmenter()
    preroll_frame_count = math.ceil(300 / FRAME_MS)  # 10 frames = 320 ms >= 300 ms

    # 20 frames of silence before speech; the ring should hold only the last few.
    for _ in range(20):
        assert segmenter.feed(frame(0.1)).kind == "none"

    event = segmenter.feed(frame(0.9))
    assert event.kind == "speech_start"

    for _ in range(9):
        segmenter.feed(frame(0.9))

    end = None
    for _ in range(int(800 / FRAME_MS) + 2):
        event = segmenter.feed(frame(0.0))
        if event.kind == "speech_end":
            end = event
            break

    assert end is not None, "EOU never fired"
    utterance = end.utterance
    assert utterance is not None

    # The preroll frames (0.1) must actually be in front of the utterance.
    head = utterance.pcm[: preroll_frame_count * SILERO_WINDOW]
    assert np.allclose(head, 0.1), "preroll was dropped — the first syllable gets clipped"
    assert 200 <= utterance.preroll_ms <= 320


def test_preroll_never_exceeds_configured_window():
    segmenter = create_segmenter(preroll_ms=200)
    for _ in range(100):
        segmenter.feed(frame(0.1))
    segmenter.feed(frame(0.9))
    assert segmenter._preroll_used_ms <= 200 + FRAME_MS


def test_eou_fires_at_configured_silence():
    segmenter = create_segmenter(silence_ms=800)
    segmenter.feed(frame(0.9))
    for _ in range(6):
        segmenter.feed(frame(0.9))

    silence_frame_count = 0
    while True:
        event = segmenter.feed(frame(0.0))
        silence_frame_count += 1
        if event.kind == "speech_end":
            break
        assert silence_frame_count < 100

    observed_silence_ms = silence_frame_count * FRAME_MS
    assert 800 <= observed_silence_ms <= 800 + 2 * FRAME_MS, observed_silence_ms


def test_short_noise_is_not_an_utterance():
    """A single cough must not start a turn."""
    segmenter = create_segmenter(min_speech_ms=200)
    segmenter.feed(frame(0.9))  # one 32 ms burst
    for _ in range(int(800 / FRAME_MS) + 2):
        event = segmenter.feed(frame(0.0))
        if event.kind == "speech_end":
            raise AssertionError("a short noise was accepted as an utterance")
    assert segmenter.state.value == "idle"


def test_max_utterance_cuts():
    segmenter = create_segmenter(max_utterance_ms=320)
    segmenter.feed(frame(0.9))
    end = None
    for _ in range(30):
        event = segmenter.feed(frame(0.9))
        if event.kind == "speech_end":
            end = event
            break
    assert end is not None and end.utterance.ended_by == "max_len"


def test_ptt_keeps_preroll():
    """PTT preroll must include speech that begins before the key event."""
    segmenter = create_segmenter()
    for _ in range(20):
        segmenter.prime_preroll(frame(0.1))
    event = segmenter.force_start()
    assert event.kind == "speech_start"
    for _ in range(5):
        segmenter.feed(frame(0.9))
    end = segmenter.force_end()
    assert end.utterance is not None
    assert np.isclose(end.utterance.pcm[0], 0.1), "preroll was lost on the PTT path"
    assert end.utterance.ended_by == "manual"
