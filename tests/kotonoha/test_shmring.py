from __future__ import annotations

import fcntl
import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

from kotonoha._shmring import (
    _MAXIMUM_ATTACHED_RINGS,
    DESC_FMT,
    AudioRef,
    AudioRing,
    StaleSlotError,
    _attached,
    close_attached,
    read_cached,
)

NAME = "kotonoha_test_ring"


def test_publish_and_read_across_attach() -> None:
    ring = AudioRing.create(NAME, slots=3, slot_seconds=1, sample_rate=16000)
    try:
        pcm = np.linspace(-1, 1, 8000, dtype=np.float32)
        ref = ring.publish(pcm)

        # A service receiving only the reference and attaching to the segment.
        consumer = AudioRing.attach(NAME)
        try:
            output = consumer.read(AudioRef.from_json(ref.to_json()))
        finally:
            consumer.close()
        assert output.shape == pcm.shape
        assert np.allclose(output, pcm)
        assert ref.seconds == pytest.approx(0.5)
    finally:
        ring.close()


def test_publish_rejects_empty_audio() -> None:
    ring = AudioRing.create(NAME, slots=2, slot_seconds=1, sample_rate=16000)
    try:
        with pytest.raises(ValueError, match="empty audio"):
            ring.publish(np.zeros(0, dtype=np.float32))
    finally:
        ring.close()


def test_independent_consumer_does_not_unlink_the_owner_segment() -> None:
    ring = AudioRing.create(NAME, slots=2, slot_seconds=1, sample_rate=16000)
    try:
        environment = os.environ.copy()
        environment["KOTONOHA_TEST_SHARED_MEMORY_NAME"] = NAME
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os; from kotonoha._shmring import AudioRing; "
                "ring = AudioRing.attach(os.environ['KOTONOHA_TEST_SHARED_MEMORY_NAME']); "
                "ring.close()",
            ],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=10,
        )
        assert completed.returncode == 0, completed.stderr

        consumer = AudioRing.attach(NAME)
        consumer.close()
    finally:
        ring.close()


def test_independent_reader_waits_for_the_owner_copy_lock() -> None:
    ring = AudioRing.create(NAME, slots=2, slot_seconds=1, sample_rate=16000)
    process: subprocess.Popen[str] | None = None
    try:
        reference = ring.publish(np.ones(160, dtype=np.float32))
        environment = os.environ.copy()
        environment["KOTONOHA_TEST_SHARED_MEMORY_REFERENCE"] = json.dumps(
            reference.to_json()
        )
        fcntl.flock(ring._lock_descriptor, fcntl.LOCK_EX)
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import json, os; "
                "from kotonoha._shmring import AudioRef, AudioRing; "
                "data = json.loads(os.environ['KOTONOHA_TEST_SHARED_MEMORY_REFERENCE']); "
                "ring = AudioRing.attach(data['name']); print('ready', flush=True); "
                "ring.read(AudioRef.from_json(data)); ring.close()",
            ],
            env=environment,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.1)
        fcntl.flock(ring._lock_descriptor, fcntl.LOCK_UN)
        _stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
    finally:
        fcntl.flock(ring._lock_descriptor, fcntl.LOCK_UN)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
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


def test_read_rejects_a_slot_marked_as_being_overwritten() -> None:
    ring = AudioRing.create(NAME, slots=2, slot_seconds=1, sample_rate=16000)
    try:
        reference = ring.publish(np.zeros(1000, dtype=np.float32))
        struct.pack_into(
            DESC_FMT,
            ring._shared_memory.buf,
            ring._descriptor_offset(reference.slot),
            reference.seq,
            reference.frames,
            1,
        )

        with pytest.raises(StaleSlotError, match="being overwritten"):
            ring.read(reference)
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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("name", ""),
        ("slot", -1),
        ("seq", 0),
        ("frames", -1),
        ("sample_rate", 0),
        ("generation", 0),
    ),
)
def test_audio_reference_rejects_invalid_metadata(
    _positional_only: object | None = None,
    /,
    *,
    field: str,
    value: object,
) -> None:
    data = {
        "name": NAME,
        "slot": 0,
        "seq": 1,
        "frames": 160,
        "sample_rate": 16000,
        "generation": 1,
    }
    data[field] = value

    with pytest.raises(ValueError):
        AudioRef.from_json(data)


def test_ring_rejects_reference_metadata_mismatches() -> None:
    ring = AudioRing.create(NAME, slots=2, slot_seconds=1, sample_rate=16000)
    try:
        reference = ring.publish(np.ones(160, dtype=np.float32))
        with pytest.raises(ValueError, match="name mismatch"):
            ring.read(
                AudioRef(
                    name="another_ring",
                    slot=reference.slot,
                    seq=reference.seq,
                    frames=reference.frames,
                    sample_rate=reference.sample_rate,
                    generation=reference.generation,
                )
            )
        with pytest.raises(ValueError, match="sample rate mismatch"):
            ring.read(
                AudioRef(
                    name=reference.name,
                    slot=reference.slot,
                    seq=reference.seq,
                    frames=reference.frames,
                    sample_rate=48000,
                    generation=reference.generation,
                )
            )
    finally:
        ring.close()


def test_cached_reader_reattaches_after_the_owner_recreates_the_segment() -> None:
    first_ring = AudioRing.create(NAME, slots=2, slot_seconds=1, sample_rate=16000)
    try:
        first_reference = first_ring.publish(np.zeros(160, dtype=np.float32))
        assert np.count_nonzero(read_cached(first_reference, NAME)) == 0
    finally:
        first_ring.close()

    second_ring = AudioRing.create(NAME, slots=2, slot_seconds=1, sample_rate=16000)
    try:
        second_reference = second_ring.publish(np.ones(160, dtype=np.float32))
        assert np.all(read_cached(second_reference, NAME) == 1.0)
    finally:
        close_attached()
        second_ring.close()


def test_cached_reader_rejects_an_unexpected_segment_name() -> None:
    reference = AudioRef(NAME, 0, 1, 160, 16000, 1)

    with pytest.raises(ValueError, match="unexpected shared-memory name"):
        read_cached(reference, "another_ring")


def test_cached_reader_evicts_handles_above_the_fixed_limit() -> None:
    rings: list[AudioRing] = []
    close_attached()
    try:
        for index in range(_MAXIMUM_ATTACHED_RINGS + 1):
            ring = AudioRing.create(
                f"{NAME}_cache_{index}",
                slots=1,
                slot_seconds=1,
                sample_rate=16000,
            )
            rings.append(ring)
            reference = ring.publish(np.ones(160, dtype=np.float32))
            assert read_cached(reference, reference.name).size == 160

        assert len(_attached) == _MAXIMUM_ATTACHED_RINGS
        assert rings[0]._shared_memory.name not in _attached
    finally:
        close_attached()
        for ring in rings:
            ring.close()
