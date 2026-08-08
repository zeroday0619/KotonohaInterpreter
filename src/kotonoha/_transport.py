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

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from kotonoha._shmring import AudioRef

Encoding = Literal["s16le", "f32le"]


@dataclass(slots=True)
class AudioPayload:
    """One utterance, carried in whichever form the consumer needs.

    The orchestrator always has the float32 PCM and always publishes it to the
    ring, so both forms are available and the client picks. That keeps the
    local/remote choice out of the orchestrator entirely.
    """

    pcm: np.ndarray
    audio_reference: AudioRef | None = None
    sample_rate: int = 16000
    _encoded_cache: dict[Encoding, bytes] = field(default_factory=dict, init=False)

    @property
    def seconds(
        self,
        /,
    ) -> float:
        return self.pcm.size / float(self.sample_rate)

    def encoded(
        self,
        /,
        encoding: Encoding = "s16le",
    ) -> bytes:
        cached = self._encoded_cache.get(encoding)
        if cached is None:
            cached = encode_pcm(self.pcm, encoding)
            self._encoded_cache[encoding] = cached
        return cached


def encode_pcm(
    pcm: np.ndarray,
    /,
    encoding: Encoding = "s16le",
) -> bytes:
    audio_samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if encoding == "f32le":
        return audio_samples.astype("<f4", copy=False).tobytes()
    if encoding != "s16le":
        raise ValueError(f"unsupported PCM encoding: {encoding}")
    scaled = np.clip(audio_samples, -1.0, 1.0)
    np.multiply(scaled, 32767.0, out=scaled)
    return scaled.astype("<i2").tobytes()


def decode_pcm(
    data: bytes,
    /,
    encoding: Encoding = "s16le",
) -> np.ndarray:
    if encoding == "f32le":
        if len(data) % 4:
            raise ValueError("f32le PCM byte length must be divisible by four")
        return np.frombuffer(data, dtype="<f4").astype(np.float32, copy=True)
    if encoding != "s16le":
        raise ValueError(f"unsupported PCM encoding: {encoding}")
    if len(data) % 2:
        raise ValueError("s16le PCM byte length must be divisible by two")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def encoded_size(
    seconds: float,
    /,
    sample_rate: int = 16000,
    encoding: Encoding = "s16le",
) -> int:
    if encoding not in {"s16le", "f32le"}:
        raise ValueError(f"unsupported PCM encoding: {encoding}")
    return int(seconds * sample_rate) * (2 if encoding == "s16le" else 4)
