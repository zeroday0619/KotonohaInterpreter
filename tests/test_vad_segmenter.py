"""§5.1 프리롤과 EOU 회귀 테스트.

프리롤은 타협 불가 항목이라 회귀가 나면 즉시 알아야 한다. 첫 음절 절단은
'ASR 품질 문제'처럼 보이기 때문에 원인을 찾는 데 며칠이 걸린다.
"""

from __future__ import annotations

import math

import numpy as np

from kotonoha.audio.vad import SILERO_WINDOW, UtteranceSegmenter

FRAME_MS = 1000.0 * SILERO_WINDOW / 16000  # 32ms


class ScriptedVad:
    """프레임의 첫 샘플 값을 그대로 확률로 쓰는 가짜 VAD."""

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
    n_pre = math.ceil(300 / FRAME_MS)  # 10 프레임 = 320ms ≥ 300ms

    # 발화 전 무음 20프레임. 프리롤 링은 마지막 9개만 들고 있어야 한다.
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

    assert end is not None, "EOU 가 발동하지 않았다"
    u = end.utterance
    assert u is not None

    # 프리롤 프레임(0.1)이 발화 앞에 실제로 붙어 있어야 한다.
    head = u.pcm[: n_pre * SILERO_WINDOW]
    assert np.allclose(head, 0.1), "프리롤이 버려졌다 — 첫 음절이 잘린다"
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
    """기침 한 번으로 턴이 시작되면 안 된다."""
    seg = make(min_speech_ms=200)
    seg.feed(frame(0.9))  # 32ms 짜리 소리 하나
    for _ in range(int(800 / FRAME_MS) + 2):
        e = seg.feed(frame(0.0))
        if e.kind == "speech_end":
            raise AssertionError("짧은 잡음이 발화로 인정됐다")
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
    """PTT 라도 프리롤은 살아 있어야 한다. 사람은 키보다 먼저 말한다."""
    seg = make()
    for _ in range(20):
        seg.prime_preroll(frame(0.1))
    ev = seg.force_start()
    assert ev.kind == "speech_start"
    for _ in range(5):
        seg.feed(frame(0.9))
    end = seg.force_end()
    assert end.utterance is not None
    assert np.isclose(end.utterance.pcm[0], 0.1), "PTT 경로에서 프리롤이 빠졌다"
    assert end.utterance.ended_by == "manual"
