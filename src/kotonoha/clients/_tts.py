"""vLLM-Omni Speech API client for clause-level Qwen3-TTS streaming."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx2
import numpy as np

from kotonoha._config import TextToSpeechConfig
from kotonoha._logging_setup import get_logger
from kotonoha._transport import decode_pcm
from kotonoha._typing import override
from kotonoha.audio._statistics import signal_statistics
from kotonoha.clients._base import (
    BaseClient,
    ServiceError,
    ServiceTimeout,
    read_json_object_response,
    service_error_from_status,
)

_SAMPLE_WIDTH = 2
_SUPPORTED_MEDIA_TYPES = {"", "application/octet-stream", "audio/pcm"}
_LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-TW": "Chinese",
}

log = get_logger(__name__)


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
            async with self._client.stream("GET", "/health", timeout=2.0) as response:
                payload = await read_json_object_response(response, allow_empty=True)
                status_code = response.status_code
            return {
                **payload,
                "ok": status_code == 200 and payload.get("ok", True) is not False,
                "service": "tts",
                "status": status_code,
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
            "max_new_tokens": self.config.max_new_tokens,
        }
        remainder = b""
        format_checked = False
        pending_prefix = bytearray()
        sample_count = 0
        square_sum = 0.0
        peak = 0.0
        clipped_sample_count = 0
        maximum_sample_count = int(
            self.config.sample_rate * self.config.max_audio_seconds
        )
        try:
            timeout = httpx2.Timeout(self.config.timeout_s, connect=2.0)
            async with self._client.stream(
                "POST", "/v1/audio/speech", json=payload, timeout=timeout
            ) as response:
                response.raise_for_status()
                media_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if media_type not in _SUPPORTED_MEDIA_TYPES:
                    raise ServiceError(
                        f"{self.label} returned {media_type or 'an unknown media type'}; "
                        "expected raw signed 16-bit PCM"
                    )
                audio_format = response.headers.get("x-kotonoha-audio-format")
                if audio_format not in {None, "s16le"}:
                    raise ServiceError(
                        f"{self.label} returned unsupported PCM format {audio_format}"
                    )
                sample_rate_header = response.headers.get("x-kotonoha-sample-rate")
                if sample_rate_header is not None:
                    try:
                        response_sample_rate = int(sample_rate_header)
                    except ValueError as error:
                        raise ServiceError(
                            f"{self.label} returned invalid sample rate {sample_rate_header}"
                        ) from error
                    if response_sample_rate != self.config.sample_rate:
                        raise ServiceError(
                            f"{self.label} returned {response_sample_rate} Hz PCM; "
                            f"expected {self.config.sample_rate} Hz"
                        )
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    if not format_checked:
                        pending_prefix.extend(chunk)
                        if len(pending_prefix) < 12:
                            continue
                        if pending_prefix[:4] == b"RIFF" and pending_prefix[8:12] == b"WAVE":
                            raise ServiceError(
                                f"{self.label} returned a WAV container where raw PCM was requested"
                            )
                        format_checked = True
                        chunk = bytes(pending_prefix)
                        pending_prefix.clear()
                    buffer = remainder + chunk
                    complete_byte_count = (len(buffer) // _SAMPLE_WIDTH) * _SAMPLE_WIDTH
                    remainder = buffer[complete_byte_count:]
                    if complete_byte_count:
                        samples = decode_pcm(buffer[:complete_byte_count], "s16le")
                        sample_count += samples.size
                        if sample_count > maximum_sample_count:
                            raise ServiceError(
                                f"{self.label} exceeded the configured "
                                f"{self.config.max_audio_seconds:.1f}-second audio limit"
                            )
                        statistics = signal_statistics(samples)
                        peak = max(peak, statistics.peak)
                        square_sum += statistics.square_sum
                        clipped_sample_count += statistics.clipped_sample_count
                        yield samples
                if not format_checked and pending_prefix:
                    if pending_prefix[:4] == b"RIFF":
                        raise ServiceError(
                            f"{self.label} returned a truncated WAV container"
                        )
                    buffer = remainder + bytes(pending_prefix)
                    complete_byte_count = (len(buffer) // _SAMPLE_WIDTH) * _SAMPLE_WIDTH
                    remainder = buffer[complete_byte_count:]
                    if complete_byte_count:
                        samples = decode_pcm(buffer[:complete_byte_count], "s16le")
                        sample_count += samples.size
                        if sample_count > maximum_sample_count:
                            raise ServiceError(
                                f"{self.label} exceeded the configured "
                                f"{self.config.max_audio_seconds:.1f}-second audio limit"
                            )
                        statistics = signal_statistics(samples)
                        peak = max(peak, statistics.peak)
                        square_sum += statistics.square_sum
                        clipped_sample_count += statistics.clipped_sample_count
                        yield samples
                if remainder:
                    raise ServiceError(
                        f"{self.label} returned an incomplete 16-bit PCM sample"
                    )
                if sample_count == 0:
                    raise ServiceError(f"{self.label} returned no PCM samples")
                root_mean_square = (square_sum / sample_count) ** 0.5
                log.info(
                    "tts.audio_received",
                    side=self.side,
                    language=language,
                    voice=payload["voice"],
                    samples=sample_count,
                    duration_s=round(sample_count / self.config.sample_rate, 3),
                    peak_dbfs=round(20.0 * np.log10(max(peak, 1e-12)), 1),
                    rms_dbfs=round(20.0 * np.log10(max(root_mean_square, 1e-12)), 1),
                    clipped_fraction=round(clipped_sample_count / sample_count, 6),
                )
        except httpx2.TimeoutException as error:
            raise ServiceTimeout(f"{self.label} timeout") from error
        except httpx2.HTTPStatusError as error:
            raise service_error_from_status(error, self.label) from error
        except httpx2.HTTPError as error:
            raise ServiceError(f"{self.label} transport error: {error!r}") from error
