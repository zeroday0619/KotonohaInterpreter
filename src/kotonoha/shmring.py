"""공유 메모리 오디오 링버퍼 (§3).

오디오를 HTTP body에 태우지 않기 위한 장치. 6초 PCM을 base64로 왕복시키면
100~200ms가 그냥 사라진다. 오케스트레이터가 발화 PCM을 슬롯에 쓰고,
서비스에는 {name, slot, seq, frames} 같은 작은 참조만 JSON으로 넘긴다.

레이아웃 (little endian, float32 mono):

    [ header 32B ][ slot descriptors 16B * N ][ data: N * slot_frames * 4B ]

    header: magic(4s) version(u32) slots(u32) slot_frames(u32)
            sample_rate(u32) write_seq(u64) pad(4)
    descriptor: seq(u64) nframes(u32) flags(u32)

단일 생산자(오케스트레이터) / 다중 소비자(ASR·검증 ASR). 소비자는 읽은 뒤
descriptor의 seq를 재확인해서 읽는 도중 덮어써졌는지 판별한다. 슬롯 수 * 30초면
순차식 통역의 왕복 시간보다 훨씬 길므로 실사용에서 덮어쓰기는 발생하지 않지만,
발생하면 조용히 틀린 오디오를 쓰는 대신 StaleSlot으로 실패시킨다.
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
    """참조한 슬롯이 이미 다른 발화로 덮어써졌다."""


@dataclass(frozen=True)
class AudioRef:
    """서비스로 넘기는 작은 참조. 그대로 JSON 직렬화된다."""

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

    # ── 생성 / 접속 ─────────────────────────────────────────────────────
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

    # ── 쓰기 (오케스트레이터 전용) ──────────────────────────────────────
    def publish(self, pcm: np.ndarray) -> AudioRef:
        if pcm.ndim != 1:
            pcm = pcm.reshape(-1)
        if pcm.dtype != DTYPE:
            pcm = pcm.astype(DTYPE, copy=False)
        n = int(pcm.shape[0])
        if n > self.slot_frames:
            # §4의 max_utterance_ms 로 상류에서 잘리지만, 방어적으로 뒤를 버린다.
            n = self.slot_frames
            pcm = pcm[:n]

        wseq = self._write_seq() + 1
        slot = (wseq - 1) % self.slots

        # 데이터를 먼저 쓰고, 그 다음 descriptor를 갱신한다(소비자가 반쪽을 보지 않도록).
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

    # ── 읽기 (서비스) ───────────────────────────────────────────────────
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

    # ── 내부 ────────────────────────────────────────────────────────────
    def _desc_off(self, slot: int) -> int:
        return HEADER_SIZE + DESC_SIZE * slot

    def _slot_view(self, slot: int) -> np.ndarray:
        # np.frombuffer 는 읽기 전용 배열을 준다. 쓰기가 필요하므로 ndarray(buffer=)를 쓴다.
        start = self._data_off + slot * self.slot_frames * 4
        return np.ndarray((self.slot_frames,), dtype=DTYPE, buffer=self._shm.buf, offset=start)

    def _write_seq(self) -> int:
        return struct.unpack_from("<Q", self._shm.buf, 20)[0]

    def _set_write_seq(self, v: int) -> None:
        struct.pack_into("<Q", self._shm.buf, 20, v)

    # ── 정리 ────────────────────────────────────────────────────────────
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
    """서비스 프로세스용 — 이름당 한 번만 attach 하고 재사용한다."""
    ring = _attached.get(name)
    if ring is None:
        ring = AudioRing.attach(name)
        _attached[name] = ring
    return ring
