"""TTS client. Sends one clause and streams PCM chunks back.

The body is raw PCM. No shared memory here: the audio flows once, from service
to orchestrator, and goes straight to the speaker. At 200 ms chunks the HTTP
chunked overhead is smaller than the playback interval.

When the service is on the A6000 the chunks cross the network, so the client
asks for s16le — half the bytes of float32, and inaudible at 24 kHz.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx
import numpy as np

from kotonoha._config import TextToSpeechConfig
from kotonoha._transport import Encoding, decode_pcm
from kotonoha._typing import override
from kotonoha.clients._base import BaseClient, ServiceError, ServiceTimeout

_WIDTH = {"s16le": 2, "f32le": 4}


class TextToSpeechClient(BaseClient):
    __slots__: ClassVar[tuple[str, ...]] = (
        "config",
        "encoding",
    )
    config: TextToSpeechConfig
    encoding: Encoding

    @override
    def __init__(
        self,
        /,
        base_url: str,
        config: TextToSpeechConfig,
        *,
        side: str = "local",
        encoding: Encoding | None = None,
        **transport_options: Any,
    ) -> None:
        super().__init__(
            base_url,
            config.timeout_s,
            "tts",
            side=side,
            **transport_options,
        )
        self.config = config
        self.encoding: Encoding = encoding or ("s16le" if side == "remote" else "f32le")

    async def synthesize(
        self,
        /,
        text: str,
        language: str,
    ) -> AsyncIterator[np.ndarray]:
        payload = {
            "text": text,
            "lang": language,
            "voice": self.config.voices.get(language),
            "speaker": self.config.melo_speakers.get(language),
            "sample_rate": self.config.sample_rate,
            "encoding": self.encoding,
        }
        width = _WIDTH[self.encoding]
        remainder = b""
        try:
            timeout = httpx.Timeout(self.config.timeout_s, connect=2.0)
            async with self._client.stream(
                "POST", "/synthesize", json=payload, timeout=timeout
            ) as response:
                response.raise_for_status()
                # The response header takes precedence when a service ignores the
                # requested encoding.
                actual_encoding: Encoding = response.headers.get(  # type: ignore[assignment]
                    "x-encoding",
                    self.encoding,
                )
                width = _WIDTH.get(actual_encoding, width)
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    buffer = remainder + chunk
                    complete_byte_count = (len(buffer) // width) * width
                    remainder = buffer[complete_byte_count:]
                    if complete_byte_count:
                        yield decode_pcm(buffer[:complete_byte_count], actual_encoding)
        except httpx.TimeoutException as error:
            raise ServiceTimeout(f"{self.label} timeout") from error
        except httpx.HTTPStatusError as error:
            raise ServiceError(f"{self.label} {error.response.status_code}") from error
        except httpx.HTTPError as error:
            raise ServiceError(f"{self.label} transport error: {error!r}") from error
