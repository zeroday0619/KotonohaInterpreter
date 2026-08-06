"""vLLM-Omni Speech API client for clause-level Qwen3-TTS streaming."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx2
import numpy as np

from kotonoha._config import TextToSpeechConfig
from kotonoha._transport import decode_pcm
from kotonoha._typing import override
from kotonoha.clients._base import BaseClient, ServiceError, ServiceTimeout

_SAMPLE_WIDTH = 2
_LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-TW": "Chinese",
}


class TextToSpeechClient(BaseClient):
    __slots__: ClassVar[tuple[str, ...]] = (
        "config",
    )
    config: TextToSpeechConfig

    @override
    def __init__(
        self,
        /,
        base_url: str,
        config: TextToSpeechConfig,
        *,
        side: str = "local",
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

    @override
    async def health(
        self,
        /,
    ) -> dict:
        """Return the vLLM-Omni server health state."""
        try:
            response = await self._client.get("/health", timeout=2.0)
            return {
                "ok": response.status_code == 200,
                "service": "tts",
                "status": response.status_code,
                "side": self.side,
            }
        except Exception as error:  # noqa: BLE001
            return {
                "ok": False,
                "service": "tts",
                "error": repr(error),
                "side": self.side,
            }

    async def synthesize(
        self,
        /,
        text: str,
        language: str,
    ) -> AsyncIterator[np.ndarray]:
        payload = {
            "input": text,
            "model": self.config.served_model_name,
            "voice": self.config.voices.for_language(language),
            "language": _LANGUAGE_NAMES.get(language, "Auto"),
            "task_type": self.config.task_type,
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
        }
        remainder = b""
        try:
            timeout = httpx2.Timeout(self.config.timeout_s, connect=2.0)
            async with self._client.stream(
                "POST", "/v1/audio/speech", json=payload, timeout=timeout
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    buffer = remainder + chunk
                    complete_byte_count = (len(buffer) // _SAMPLE_WIDTH) * _SAMPLE_WIDTH
                    remainder = buffer[complete_byte_count:]
                    if complete_byte_count:
                        yield decode_pcm(buffer[:complete_byte_count], "s16le")
                if remainder:
                    raise ServiceError(
                        f"{self.label} returned an incomplete 16-bit PCM sample"
                    )
        except httpx2.TimeoutException as error:
            raise ServiceTimeout(f"{self.label} timeout") from error
        except httpx2.HTTPStatusError as error:
            detail = f"{self.label} {error.response.status_code}: {error.response.text[:200]}"
            raise ServiceError(detail) from error
        except httpx2.HTTPError as error:
            raise ServiceError(f"{self.label} transport error: {error!r}") from error
