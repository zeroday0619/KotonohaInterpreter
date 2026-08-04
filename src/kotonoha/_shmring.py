"""Shared-memory audio ring buffer (§3).

This exists so audio never rides in an HTTP body. Round-tripping six seconds of
PCM through base64 costs 100-200 ms per turn. The orchestrator writes the
utterance into a slot and services receive only a small reference —
{name, slot, seq, frames} — as JSON.

Layout (little endian, float32 mono):

    [ header 32B ][ slot descriptors 16B * N ][ data: N * slot_frames * 4B ]

    header: magic(4s) version(u32) slots(u32) slot_frames(u32)
            sample_rate(u32) write_seq(u64) pad(4)
    descriptor: seq(u64) nframes(u32) flags(u32)

Single producer (the orchestrator), multiple consumers (ASR and the verifier).
A consumer re-checks the descriptor's seq after reading to detect an overwrite
mid-read. An overwrite raises `StaleSlotError` instead of returning audio from a
different utterance.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, ClassVar

import numpy as np

from kotonoha._typing import override

MAGIC = b"KTNH"
VERSION = 1
HEADER_FMT = "<4sIIIIQI"
HEADER_SIZE = 32
DESC_FMT = "<QII"
DESC_SIZE = 16
DTYPE = np.float32


class StaleSlotError(RuntimeError):
    """The referenced slot has already been overwritten by a later utterance."""
    __slots__: ClassVar[tuple[str, ...]] = ()


@dataclass(frozen=True, slots=True)
class AudioRef:
    """The small reference handed to services. Serialises straight to JSON."""

    name: str
    slot: int
    seq: int
    frames: int
    sample_rate: int

    @property
    def seconds(
        self,
        /,
    ) -> float:
        return self.frames / float(self.sample_rate)

    def to_json(
        self,
        /,
    ) -> dict:
        return {
            "name": self.name,
            "slot": self.slot,
            "seq": self.seq,
            "frames": self.frames,
            "sample_rate": self.sample_rate,
        }

    @classmethod
    def from_json(
        cls,
        /,
        data: dict,
    ) -> AudioRef:
        return cls(
            name=data["name"],
            slot=int(data["slot"]),
            seq=int(data["seq"]),
            frames=int(data["frames"]),
            sample_rate=int(data["sample_rate"]),
        )


class AudioRing:
    __slots__: ClassVar[tuple[str, ...]] = (
        "_data_offset",
        "_owner",
        "_shared_memory",
        "sample_rate",
        "slot_frames",
        "slots",
    )
    slots: int
    slot_frames: int
    sample_rate: int
    _shared_memory: shared_memory.SharedMemory
    _owner: bool
    _data_offset: int

    @override
    def __init__(
        self,
        /,
        memory: shared_memory.SharedMemory,
        owner: bool,
    ) -> None:
        self._shared_memory = memory
        self._owner = owner
        magic, version, slots, slot_frames, sample_rate, _sequence, _padding = (
            struct.unpack_from(HEADER_FMT, memory.buf, 0)
        )
        if magic != MAGIC:
            raise ValueError(f"not a kotonoha audio ring: magic={magic!r}")
        if version != VERSION:
            raise ValueError(f"ring version mismatch: {version} != {VERSION}")
        self.slots = slots
        self.slot_frames = slot_frames
        self.sample_rate = sample_rate
        self._data_offset = HEADER_SIZE + DESC_SIZE * slots

    # -- create / attach -------------------------------------------------
    @staticmethod
    def _size(
        slots: int,
        /,
        slot_frames: int,
    ) -> int:
        return HEADER_SIZE + DESC_SIZE * slots + slots * slot_frames * 4

    @classmethod
    def create(
        cls,
        /,
        name: str,
        slots: int = 8,
        slot_seconds: int = 30,
        sample_rate: int = 16000,
        force: bool = True,
    ) -> AudioRing:
        slot_frames = int(slot_seconds * sample_rate)
        size = cls._size(slots, slot_frames)
        if force:
            try:
                existing = shared_memory.SharedMemory(name=name)
                existing.close()
                existing.unlink()
            except FileNotFoundError:
                pass
        memory = shared_memory.SharedMemory(name=name, create=True, size=size)
        memory.buf[:size] = b"\x00" * size
        struct.pack_into(
            HEADER_FMT,
            memory.buf,
            0,
            MAGIC,
            VERSION,
            slots,
            slot_frames,
            sample_rate,
            0,
            0,
        )
        return cls(memory, owner=True)

    @classmethod
    def attach(
        cls,
        /,
        name: str,
    ) -> AudioRing:
        memory = shared_memory.SharedMemory(name=name)
        return cls(memory, owner=False)

    # -- writing (orchestrator only) -------------------------------------
    def publish(
        self,
        /,
        pcm: np.ndarray,
    ) -> AudioRef:
        if pcm.ndim != 1:
            pcm = pcm.reshape(-1)
        if pcm.dtype != DTYPE:
            pcm = pcm.astype(DTYPE, copy=False)
        frame_count = int(pcm.shape[0])
        if frame_count > self.slot_frames:
            # max_utterance_ms (§4) already caps this upstream; drop the tail
            # defensively rather than corrupting the next slot.
            frame_count = self.slot_frames
            pcm = pcm[:frame_count]

        write_sequence = self._write_sequence() + 1
        slot = (write_sequence - 1) % self.slots

        # Write the data first, then the descriptor, so a consumer never sees half.
        self._slot_view(slot)[:frame_count] = pcm
        struct.pack_into(
            DESC_FMT,
            self._shared_memory.buf,
            self._descriptor_offset(slot),
            write_sequence,
            frame_count,
            0,
        )
        self._set_write_sequence(write_sequence)

        return AudioRef(
            name=self._shared_memory.name,
            slot=slot,
            seq=write_sequence,
            frames=frame_count,
            sample_rate=self.sample_rate,
        )

    # -- reading (services) ----------------------------------------------
    def read(
        self,
        /,
        reference: AudioRef,
    ) -> np.ndarray:
        if not (0 <= reference.slot < self.slots):
            raise ValueError(f"slot out of range: {reference.slot}")
        sequence_before, frame_count, _flags = struct.unpack_from(
            DESC_FMT,
            self._shared_memory.buf,
            self._descriptor_offset(reference.slot),
        )
        if sequence_before != reference.seq:
            raise StaleSlotError(
                f"slot {reference.slot}: have seq {sequence_before}, want {reference.seq}"
            )
        output = np.array(
            self._slot_view(reference.slot)[: min(frame_count, reference.frames)],
            copy=True,
        )
        sequence_after, _, _ = struct.unpack_from(
            DESC_FMT,
            self._shared_memory.buf,
            self._descriptor_offset(reference.slot),
        )
        if sequence_after != reference.seq:
            raise StaleSlotError(f"slot {reference.slot} overwritten during read")
        return output

    # -- internals -------------------------------------------------------
    def _descriptor_offset(
        self,
        /,
        slot: int,
    ) -> int:
        return HEADER_SIZE + DESC_SIZE * slot

    def _slot_view(
        self,
        /,
        slot: int,
    ) -> np.ndarray:
        # np.frombuffer returns a read-only array, so a writable ndarray view is required.
        # ndarray(buffer=) instead.
        start = self._data_offset + slot * self.slot_frames * 4
        return np.ndarray(
            (self.slot_frames,),
            dtype=DTYPE,
            buffer=self._shared_memory.buf,
            offset=start,
        )

    def _write_sequence(
        self,
        /,
    ) -> int:
        return struct.unpack_from("<Q", self._shared_memory.buf, 20)[0]

    def _set_write_sequence(
        self,
        /,
        value: int,
    ) -> None:
        struct.pack_into("<Q", self._shared_memory.buf, 20, value)

    # -- teardown --------------------------------------------------------
    def close(
        self,
        /,
    ) -> None:
        try:
            self._shared_memory.close()
        finally:
            if self._owner:
                try:
                    self._shared_memory.unlink()
                except FileNotFoundError:
                    pass

    def __enter__(
        self,
        /,
    ) -> AudioRing:
        return self

    def __exit__(
        self,
        /,
        *exc: Any,
    ) -> None:
        self.close()


_attached: dict[str, AudioRing] = {}


def attach_cached(
    name: str,
    /,
) -> AudioRing:
    """For service processes — attach once per name and reuse."""
    ring = _attached.get(name)
    if ring is None:
        ring = AudioRing.attach(name)
        _attached[name] = ring
    return ring
