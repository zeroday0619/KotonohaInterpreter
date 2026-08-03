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
mid-read. With slots * 30 s of capacity that is far longer than a consecutive
turn takes, so it should never happen in practice — but if it does, we fail with
StaleSlot instead of quietly transcribing the wrong audio.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

MAGIC = b"KTNH"
VERSION = 1
HEADER_FMT = "<4sIIIIQI"
HEADER_SIZE = 32
DESC_FMT = "<QII"
DESC_SIZE = 16
DTYPE = np.float32


class StaleSlotError(RuntimeError):
    """The referenced slot has already been overwritten by a later utterance."""


@dataclass(frozen=True)
class AudioRef:
    """The small reference handed to services. Serialises straight to JSON."""

    name: str
    slot: int
    seq: int
    frames: int
    sample_rate: int

    @property
    def seconds(self) -> float:
        return self.frames / float(self.sample_rate)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "slot": self.slot,
            "seq": self.seq,
            "frames": self.frames,
            "sample_rate": self.sample_rate,
        }

    @classmethod
    def from_json(cls, d: dict) -> AudioRef:
        return cls(
            name=d["name"],
            slot=int(d["slot"]),
            seq=int(d["seq"]),
            frames=int(d["frames"]),
            sample_rate=int(d["sample_rate"]),
        )


class AudioRing:
    def __init__(self, shm: shared_memory.SharedMemory, owner: bool):
        self._shm = shm
        self._owner = owner
        magic, ver, slots, slot_frames, sr, _wseq, _pad = struct.unpack_from(
            HEADER_FMT, shm.buf, 0
        )
        if magic != MAGIC:
            raise ValueError(f"not a kotonoha audio ring: magic={magic!r}")
        if ver != VERSION:
            raise ValueError(f"ring version mismatch: {ver} != {VERSION}")
        self.slots = slots
        self.slot_frames = slot_frames
        self.sample_rate = sr
        self._data_off = HEADER_SIZE + DESC_SIZE * slots

    # -- create / attach -------------------------------------------------
    @staticmethod
    def _size(slots: int, slot_frames: int) -> int:
        return HEADER_SIZE + DESC_SIZE * slots + slots * slot_frames * 4

    @classmethod
    def create(
        cls,
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
                old = shared_memory.SharedMemory(name=name)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        shm.buf[:size] = b"\x00" * size
        struct.pack_into(
            HEADER_FMT, shm.buf, 0, MAGIC, VERSION, slots, slot_frames, sample_rate, 0, 0
        )
        return cls(shm, owner=True)

    @classmethod
    def attach(cls, name: str) -> AudioRing:
        shm = shared_memory.SharedMemory(name=name)
        return cls(shm, owner=False)

    # -- writing (orchestrator only) -------------------------------------
    def publish(self, pcm: np.ndarray) -> AudioRef:
        if pcm.ndim != 1:
            pcm = pcm.reshape(-1)
        if pcm.dtype != DTYPE:
            pcm = pcm.astype(DTYPE, copy=False)
        n = int(pcm.shape[0])
        if n > self.slot_frames:
            # max_utterance_ms (§4) already caps this upstream; drop the tail
            # defensively rather than corrupting the next slot.
            n = self.slot_frames
            pcm = pcm[:n]

        wseq = self._write_seq() + 1
        slot = (wseq - 1) % self.slots

        # Write the data first, then the descriptor, so a consumer never sees half.
        self._slot_view(slot)[:n] = pcm
        struct.pack_into(DESC_FMT, self._shm.buf, self._desc_off(slot), wseq, n, 0)
        self._set_write_seq(wseq)

        return AudioRef(
            name=self._shm.name,
            slot=slot,
            seq=wseq,
            frames=n,
            sample_rate=self.sample_rate,
        )

    # -- reading (services) ----------------------------------------------
    def read(self, ref: AudioRef) -> np.ndarray:
        if not (0 <= ref.slot < self.slots):
            raise ValueError(f"slot out of range: {ref.slot}")
        seq0, n, _flags = struct.unpack_from(DESC_FMT, self._shm.buf, self._desc_off(ref.slot))
        if seq0 != ref.seq:
            raise StaleSlotError(f"slot {ref.slot}: have seq {seq0}, want {ref.seq}")
        out = np.array(self._slot_view(ref.slot)[: min(n, ref.frames)], copy=True)
        seq1, _, _ = struct.unpack_from(DESC_FMT, self._shm.buf, self._desc_off(ref.slot))
        if seq1 != ref.seq:
            raise StaleSlotError(f"slot {ref.slot} overwritten during read")
        return out

    # -- internals -------------------------------------------------------
    def _desc_off(self, slot: int) -> int:
        return HEADER_SIZE + DESC_SIZE * slot

    def _slot_view(self, slot: int) -> np.ndarray:
        # np.frombuffer hands back a read-only array; we need to write, so use
        # ndarray(buffer=) instead.
        start = self._data_off + slot * self.slot_frames * 4
        return np.ndarray((self.slot_frames,), dtype=DTYPE, buffer=self._shm.buf, offset=start)

    def _write_seq(self) -> int:
        return struct.unpack_from("<Q", self._shm.buf, 20)[0]

    def _set_write_seq(self, v: int) -> None:
        struct.pack_into("<Q", self._shm.buf, 20, v)

    # -- teardown --------------------------------------------------------
    def close(self) -> None:
        try:
            self._shm.close()
        finally:
            if self._owner:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass

    def __enter__(self) -> AudioRing:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_attached: dict[str, AudioRing] = {}


def attach_cached(name: str) -> AudioRing:
    """For service processes — attach once per name and reuse."""
    ring = _attached.get(name)
    if ring is None:
        ring = AudioRing.attach(name)
        _attached[name] = ring
    return ring
