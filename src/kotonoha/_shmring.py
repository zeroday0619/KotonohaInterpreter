"""Shared-memory audio ring buffer (§3).

This exists so audio never rides in an HTTP body. Round-tripping six seconds of
PCM through base64 costs 100-200 ms per turn. The orchestrator writes the
utterance into a slot and services receive only a small reference —
{name, generation, slot, seq, frames} — as JSON.

Layout (little endian, float32 mono):

    [ header 40B ][ slot descriptors 16B * N ][ data: N * slot_frames * 4B ]

    header: magic(4s) version(u32) slots(u32) slot_frames(u32)
            sample_rate(u32) write_seq(u64) generation(u64) pad(4)
    descriptor: seq(u64) nframes(u32) flags(u32)

Single producer (the orchestrator), multiple consumers (ASR and the verifier).
A consumer re-checks the descriptor's seq after reading to detect an overwrite
mid-read. An overwrite raises `StaleSlotError` instead of returning audio from a
different utterance.
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
from secrets import randbits
from typing import Any, ClassVar

import numpy as np

from kotonoha._typing import override

MAGIC = b"KTNH"
VERSION = 2
HEADER_FMT = "<4sIIIIQQI"
HEADER_SIZE = 40
DESC_FMT = "<QII"
DESC_SIZE = 16
DTYPE = np.float32
WRITING_FLAG = 1
_owned_shared_memory_names: set[str] = set()
_ownership_lock = threading.RLock()


def _open_shared_memory(
    name: str,
    /,
    *,
    create: bool = False,
    size: int = 0,
    track: bool = True,
) -> shared_memory.SharedMemory:
    """Open a segment without assigning consumer processes deletion ownership."""
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(
            name=name,
            create=create,
            size=size,
            track=track,
        )
    memory = shared_memory.SharedMemory(name=name, create=create, size=size)
    if not track and os.name == "posix":
        # Python before 3.13 has no public track=False parameter. Independent
        # service processes otherwise unlink the owner's segment when they exit.
        resource_tracker.unregister(memory._name, "shared_memory")
    return memory


def prepare_shared_memory_tracking() -> None:
    """Start the tracker before a terminal interface replaces standard error."""
    if os.name == "posix":
        resource_tracker.ensure_running()


def _open_lock_descriptor(
    name: str,
    /,
) -> int:
    """Open the persistent sidecar used to synchronize independent processes."""
    shared_directory = Path("/dev/shm")  # nosec B108
    base_directory = (
        shared_directory
        if shared_directory.is_dir()
        else Path(tempfile.gettempdir())
    )
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    lock_path = base_directory / f".kotonoha-shm-{digest}.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(lock_path, flags, 0o600)


@contextmanager
def _memory_lock(
    descriptor: int,
    /,
    *,
    exclusive: bool,
) -> Iterator[None]:
    """Serialize cross-process copies and provide a POSIX memory-ordering barrier."""
    if os.name != "posix":
        yield
        return
    import fcntl

    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(descriptor, operation)
    try:
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


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
    generation: int

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
            "generation": self.generation,
        }

    @classmethod
    def from_json(
        cls,
        /,
        data: dict,
    ) -> AudioRef:
        reference = cls(
            name=str(data["name"]),
            slot=int(data["slot"]),
            seq=int(data["seq"]),
            frames=int(data["frames"]),
            sample_rate=int(data["sample_rate"]),
            generation=int(data["generation"]),
        )
        if not reference.name:
            raise ValueError("shared-memory name must not be empty")
        if reference.slot < 0:
            raise ValueError("shared-memory slot must not be negative")
        if reference.seq <= 0:
            raise ValueError("shared-memory sequence must be positive")
        if reference.frames <= 0:
            raise ValueError("audio frame count must be positive")
        if reference.sample_rate <= 0:
            raise ValueError("audio sample rate must be positive")
        if reference.generation <= 0:
            raise ValueError("shared-memory generation must be positive")
        return reference


class AudioRing:
    __slots__: ClassVar[tuple[str, ...]] = (
        "_data_offset",
        "_lock_descriptor",
        "_owner",
        "_shared_memory",
        "generation",
        "sample_rate",
        "slot_frames",
        "slots",
    )
    slots: int
    slot_frames: int
    sample_rate: int
    generation: int
    _shared_memory: shared_memory.SharedMemory
    _owner: bool
    _data_offset: int
    _lock_descriptor: int

    @override
    def __init__(
        self,
        /,
        memory: shared_memory.SharedMemory,
        owner: bool,
    ) -> None:
        self._shared_memory = memory
        self._owner = owner
        if memory.size < HEADER_SIZE:
            raise ValueError(
                f"shared-memory ring header is truncated: {memory.size} < {HEADER_SIZE}"
            )
        (
            magic,
            version,
            slots,
            slot_frames,
            sample_rate,
            _sequence,
            generation,
            _padding,
        ) = struct.unpack_from(HEADER_FMT, memory.buf, 0)
        if magic != MAGIC:
            raise ValueError(f"not a kotonoha audio ring: magic={magic!r}")
        if version != VERSION:
            raise ValueError(f"ring version mismatch: {version} != {VERSION}")
        if slots <= 0 or slot_frames <= 0 or sample_rate <= 0 or generation <= 0:
            raise ValueError("invalid shared-memory ring metadata")
        expected_size = self._size(slots, slot_frames)
        # Darwin rounds POSIX shared-memory segments up to its VM page size.
        # Reject only truncated layouts; trailing page padding is inaccessible
        # to the ring and does not change descriptor or audio offsets.
        if memory.size < expected_size:
            raise ValueError(
                f"ring is truncated: {memory.size} < {expected_size}"
            )
        self.slots = slots
        self.slot_frames = slot_frames
        self.sample_rate = sample_rate
        self.generation = generation
        self._data_offset = HEADER_SIZE + DESC_SIZE * slots
        self._lock_descriptor = _open_lock_descriptor(memory.name)

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
        slot_seconds: int = 31,
        sample_rate: int = 16000,
        force: bool = True,
    ) -> AudioRing:
        if slots <= 0 or slot_seconds <= 0 or sample_rate <= 0:
            raise ValueError("shared-memory dimensions must be positive")
        slot_frames = int(slot_seconds * sample_rate)
        size = cls._size(slots, slot_frames)
        with _ownership_lock:
            if name in _owned_shared_memory_names:
                raise FileExistsError(f"shared-memory ring is already owned: {name}")
            if force:
                try:
                    existing = _open_shared_memory(name)
                    existing.close()
                    existing.unlink()
                except FileNotFoundError:
                    pass
            memory = _open_shared_memory(name, create=True, size=size)
            _owned_shared_memory_names.add(memory.name)
        try:
            metadata_size = HEADER_SIZE + DESC_SIZE * slots
            memory.buf[:metadata_size] = bytes(metadata_size)
            generation = randbits(64) or 1
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
                generation,
                0,
            )
            return cls(memory, owner=True)
        except Exception:
            memory.close()
            memory.unlink()
            with _ownership_lock:
                _owned_shared_memory_names.discard(memory.name)
            raise

    @classmethod
    def attach(
        cls,
        /,
        name: str,
    ) -> AudioRing:
        with _ownership_lock:
            track = name in _owned_shared_memory_names
        memory = _open_shared_memory(name, track=track)
        try:
            return cls(memory, owner=False)
        except Exception:
            memory.close()
            raise

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
        if frame_count == 0:
            raise ValueError("cannot publish an empty audio buffer")
        if frame_count > self.slot_frames:
            # max_utterance_ms (§4) already caps this upstream; drop the tail
            # defensively rather than corrupting the next slot.
            frame_count = self.slot_frames
            pcm = pcm[:frame_count]

        with _memory_lock(self._lock_descriptor, exclusive=True):
            write_sequence = self._write_sequence() + 1
            slot = (write_sequence - 1) % self.slots

            # Invalidate the old descriptor before overwriting its data. Consumers
            # reject this flag and verify the committed descriptor again after copying.
            struct.pack_into(
                DESC_FMT,
                self._shared_memory.buf,
                self._descriptor_offset(slot),
                write_sequence,
                frame_count,
                WRITING_FLAG,
            )
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
            generation=self.generation,
        )

    # -- reading (services) ----------------------------------------------
    def read(
        self,
        /,
        reference: AudioRef,
    ) -> np.ndarray:
        if reference.name != self._shared_memory.name:
            raise ValueError(
                f"shared-memory name mismatch: {reference.name} != {self._shared_memory.name}"
            )
        if not (0 <= reference.slot < self.slots):
            raise ValueError(f"slot out of range: {reference.slot}")
        if reference.seq <= 0:
            raise ValueError("shared-memory sequence must be positive")
        if not (0 < reference.frames <= self.slot_frames):
            raise ValueError(f"audio frame count out of range: {reference.frames}")
        if reference.sample_rate != self.sample_rate:
            raise ValueError(
                f"sample rate mismatch: {reference.sample_rate} != {self.sample_rate}"
            )
        if reference.generation != self.generation:
            raise StaleSlotError(
                f"ring generation mismatch: {reference.generation} != {self.generation}"
            )
        with _memory_lock(self._lock_descriptor, exclusive=False):
            sequence_before, frame_count, flags_before = struct.unpack_from(
                DESC_FMT,
                self._shared_memory.buf,
                self._descriptor_offset(reference.slot),
            )
            if sequence_before != reference.seq:
                raise StaleSlotError(
                    f"slot {reference.slot}: have seq {sequence_before}, want {reference.seq}"
                )
            if flags_before & WRITING_FLAG:
                raise StaleSlotError(f"slot {reference.slot} is being overwritten")
            if frame_count != reference.frames or frame_count > self.slot_frames:
                raise ValueError(
                    f"descriptor frame count mismatch: {frame_count} != {reference.frames}"
                )
            output = np.array(self._slot_view(reference.slot)[:frame_count], copy=True)
            sequence_after, frame_count_after, flags_after = struct.unpack_from(
                DESC_FMT,
                self._shared_memory.buf,
                self._descriptor_offset(reference.slot),
            )
            if (
                sequence_after != reference.seq
                or frame_count_after != frame_count
                or flags_after != 0
            ):
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
        lock_descriptor = self._lock_descriptor
        self._lock_descriptor = -1
        try:
            self._shared_memory.close()
        finally:
            try:
                if self._owner:
                    try:
                        self._shared_memory.unlink()
                    except FileNotFoundError:
                        pass
                    finally:
                        with _ownership_lock:
                            _owned_shared_memory_names.discard(self._shared_memory.name)
            finally:
                if lock_descriptor >= 0:
                    os.close(lock_descriptor)

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


_MAXIMUM_ATTACHED_RINGS = 4
_attached: OrderedDict[str, AudioRing] = OrderedDict()
_attached_lock = threading.RLock()


def attach_cached(
    name: str,
    /,
    expected_name: str | None = None,
) -> AudioRing:
    """Attach to a bounded set of allowed segments and reuse open handles."""
    if expected_name is not None and name != expected_name:
        raise ValueError(f"unexpected shared-memory name: {name}")
    with _attached_lock:
        ring = _attached.get(name)
        if ring is not None:
            _attached.move_to_end(name)
            return ring
        ring = AudioRing.attach(name)
        _attached[name] = ring
        while len(_attached) > _MAXIMUM_ATTACHED_RINGS:
            _old_name, old_ring = _attached.popitem(last=False)
            old_ring.close()
        return ring


def read_cached(
    reference: AudioRef,
    /,
    expected_name: str | None = None,
) -> np.ndarray:
    """Read once more after reattaching when a process recreated the segment."""
    # Closing a stale cached handle must not race another request that is copying
    # from the same mapping. The lock covers only the bounded shared-memory copy.
    with _attached_lock:
        ring = attach_cached(reference.name, expected_name)
        try:
            return ring.read(reference)
        except StaleSlotError:
            close_attached(reference.name)
            return attach_cached(reference.name, expected_name).read(reference)


def close_attached(
    name: str | None = None,
    /,
) -> None:
    """Close cached consumer handles during shutdown or segment replacement."""
    with _attached_lock:
        if name is not None:
            ring = _attached.pop(name, None)
            rings = () if ring is None else (ring,)
        else:
            rings = tuple(_attached.values())
            _attached.clear()
    for ring in rings:
        ring.close()
