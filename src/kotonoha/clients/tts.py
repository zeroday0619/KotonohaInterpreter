"""TTS client. Sends one clause and streams PCM chunks back.

The body is raw PCM. No shared memory here: the audio flows once, from service
to orchestrator, and goes straight to the speaker. At 200 ms chunks the HTTP
chunked overhead is smaller than the playback interval.

When the service is on the A6000 the chunks cross the network, so the client
asks for s16le — half the bytes of float32, and inaudible at 24 kHz.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import numpy as np

from ..config import TtsCfg
from ..transport import Encoding, decode_pcm
from .base import BaseClient, ServiceError, ServiceTimeout

_WIDTH = {"s16le": 2, "f32le": 4}


class TtsClient(BaseClient):
    def __init__(
        self,
        base_url: str,
        cfg: TtsCfg,
        *,
        side: str = "local",
        encoding: Encoding | None = None,
        **transport,
    ):
        super().__init__(base_url, cfg.timeout_s, "tts", side=side, **transport)
        self.cfg = cfg
        self.encoding: Encoding = encoding or ("s16le" if side == "remote" else "f32le")

    async def synthesize(self, text: str, lang: str) -> AsyncIterator[np.ndarray]:
        payload = {
            "text": text,
            "lang": lang,
            "voice": self.cfg.voices.get(lang),
            "speaker": self.cfg.melo_speakers.get(lang),
            "sample_rate": self.cfg.sample_rate,
            "encoding": self.encoding,
        }
        width = _WIDTH[self.encoding]
        carry = b""
        try:
            timeout = httpx.Timeout(self.cfg.timeout_s, connect=2.0)
            async with self._client.stream(
                "POST", "/synthesize", json=payload, timeout=timeout
            ) as r:
                r.raise_for_status()
                # An older service that ignores `encoding` tells us so in the
                # response header; trust the header over our request.
                actual: Encoding = r.headers.get("x-encoding", self.encoding)  # type: ignore[assignment]
                width = _WIDTH.get(actual, width)
                async for chunk in r.aiter_bytes():
                    if not chunk:
                        continue
                    buf = carry + chunk
                    n = (len(buf) // width) * width
                    carry = buf[n:]
                    if n:
                        yield decode_pcm(buf[:n], actual)
        except httpx.TimeoutException as e:
            raise ServiceTimeout(f"{self.label} timeout") from e
        except httpx.HTTPStatusError as e:
            raise ServiceError(f"{self.label} {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise ServiceError(f"{self.label} transport error: {e!r}") from e
