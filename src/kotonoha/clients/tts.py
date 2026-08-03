"""TTS 클라이언트. 절 하나를 보내고 PCM 청크를 스트리밍으로 받는다.

바디는 raw float32 LE mono. 여기서는 오디오가 서비스 → 오케스트레이터 방향으로
한 번만 흐르고 즉시 스피커로 나가므로 공유메모리를 쓰지 않는다. 200ms 청크면
HTTP chunked 오버헤드가 재생 간격보다 작다.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import numpy as np

from ..config import TtsCfg
from .base import BaseClient, ServiceError, ServiceTimeout


class TtsClient(BaseClient):
    def __init__(self, base_url: str, cfg: TtsCfg):
        super().__init__(base_url, cfg.timeout_s, "tts")
        self.cfg = cfg

    async def synthesize(self, text: str, lang: str) -> AsyncIterator[np.ndarray]:
        payload = {
            "text": text,
            "lang": lang,
            "voice": self.cfg.voices.get(lang),
            "speaker": self.cfg.melo_speakers.get(lang),
            "sample_rate": self.cfg.sample_rate,
        }
        carry = b""
        try:
            timeout = httpx.Timeout(self.cfg.timeout_s, connect=2.0)
            async with self._client.stream(
                "POST", "/synthesize", json=payload, timeout=timeout
            ) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    if not chunk:
                        continue
                    buf = carry + chunk
                    n = (len(buf) // 4) * 4
                    carry = buf[n:]
                    if n:
                        yield np.frombuffer(buf[:n], dtype="<f4").copy()
        except httpx.TimeoutException as e:
            raise ServiceTimeout("tts timeout") from e
        except httpx.HTTPStatusError as e:
            raise ServiceError(f"tts {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise ServiceError(f"tts transport error: {e!r}") from e
