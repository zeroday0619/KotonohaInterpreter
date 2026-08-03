"""How an utterance reaches a service.

Two paths, chosen by where the service runs:

  local   the shared-memory ring (§3). The service attaches to /dev/shm and
          reads the slot. Nothing is copied, nothing is serialised.
  remote  the raw PCM goes over the wire as a multipart upload.

The §3 rule was "do not put audio in an HTTP body", and the reason was the
100-200 ms wasted on base64 for a purely local hop. Across a network there is
no shared memory to use, so the audio has to travel — but it still does not get
base64'd. It goes as a binary part, as s16le by default, which halves the bytes
against float32 and costs nothing at 16 kHz.

    6 s utterance @ 16 kHz
      f32le  384 KB
      s16le  192 KB   ~1.6 ms on a gigabit link, plus RTT

`kotonoha netcheck` measures what the link actually adds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .shmring import AudioRef

Encoding = Literal["s16le", "f32le"]


@dataclass
class AudioPayload:
    """One utterance, carried in whichever form the consumer needs.

    The orchestrator always has the float32 PCM and always publishes it to the
    ring, so both forms are available and the client picks. That keeps the
    local/remote choice out of the orchestrator entirely.
    """

    pcm: np.ndarray
    ref: AudioRef | None = None
    sample_rate: int = 16000

    @property
    def seconds(self) -> float:
        return self.pcm.size / float(self.sample_rate)

    def encoded(self, encoding: Encoding = "s16le") -> bytes:
        return encode_pcm(self.pcm, encoding)


def encode_pcm(pcm: np.ndarray, encoding: Encoding = "s16le") -> bytes:
    x = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if encoding == "f32le":
        return x.astype("<f4").tobytes()
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def decode_pcm(data: bytes, encoding: Encoding = "s16le") -> np.ndarray:
    if encoding == "f32le":
        n = (len(data) // 4) * 4
        return np.frombuffer(data[:n], dtype="<f4").astype(np.float32, copy=True)
    n = (len(data) // 2) * 2
    return np.frombuffer(data[:n], dtype="<i2").astype(np.float32) / 32768.0


def encoded_size(seconds: float, sample_rate: int = 16000, encoding: Encoding = "s16le") -> int:
    return int(seconds * sample_rate) * (2 if encoding == "s16le" else 4)
