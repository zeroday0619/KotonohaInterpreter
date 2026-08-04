from __future__ import annotations

import numpy as np
import pytest

from kotonoha._shmring import AudioRef, AudioRing, StaleSlotError

NAME = "kotonoha_test_ring"


def test_publish_and_read_across_attach() -> None:
    ring = AudioRing.create(NAME, slots=3, slot_seconds=1, sample_rate=16000)
    try:
        pcm = np.linspace(-1, 1, 8000, dtype=np.float32)
        ref = ring.publish(pcm)

        # A service receiving only the reference and attaching to the segment.
        consumer = AudioRing.attach(NAME)
        output = consumer.read(AudioRef.from_json(ref.to_json()))
        assert output.shape == pcm.shape
        assert np.allclose(output, pcm)
        assert ref.seconds == pytest.approx(0.5)
    finally:
        ring.close()


def test_overwrite_is_detected_not_silently_wrong() -> None:
    ring = AudioRing.create(NAME, slots=2, slot_seconds=1, sample_rate=16000)
    try:
        old = ring.publish(np.zeros(1000, dtype=np.float32))
        for _ in range(2):
            ring.publish(np.ones(1000, dtype=np.float32))
        with pytest.raises(StaleSlotError):
            ring.read(old)
    finally:
        ring.close()


def test_oversized_utterance_is_truncated_not_crashing() -> None:
    ring = AudioRing.create(NAME, slots=2, slot_seconds=1, sample_rate=16000)
    try:
        ref = ring.publish(np.ones(50000, dtype=np.float32))
        assert ref.frames == 16000
        assert ring.read(ref).size == 16000
    finally:
        ring.close()
