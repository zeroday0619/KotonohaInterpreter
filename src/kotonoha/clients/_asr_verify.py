"""Cross-verification ASR client (faster-whisper large-v3).

Called conditionally on the Orin, per §5.5, because it costs 0.8 s there. Remote
deployments can select `asr_verify.mode: always` after measuring the added latency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from pydantic import BaseModel, Field, ValidationError

from kotonoha._config import AsrVerificationConfig
from kotonoha._transport import AudioPayload, Encoding
from kotonoha._typing import override
from kotonoha.clients._base import BaseClient, ServiceError

MAXIMUM_VERIFICATION_TEXT_CHARACTERS: Final[int] = 4096


class VerificationResponse(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    text: str = Field("", max_length=MAXIMUM_VERIFICATION_TEXT_CHARACTERS)
    avg_logprob: float = Field(-99.0, allow_inf_nan=False)
    language: str | None = Field(None, max_length=128)
    infer_ms: float = Field(0.0, ge=0.0, le=3_600_000.0, allow_inf_nan=False)


@dataclass(slots=True)
class VerifyResult:
    text: str
    avg_logprob: float
    language: str | None
    infer_ms: float


class AsrVerifyClient(BaseClient):
    __slots__: ClassVar[tuple[str, ...]] = (
        "config",
        "encoding",
    )
    config: AsrVerificationConfig
    encoding: Encoding

    @override
    def __init__(
        self,
        /,
        base_url: str,
        config: AsrVerificationConfig,
        *,
        side: str = "local",
        encoding: Encoding = "s16le",
        **transport: Any,
    ) -> None:
        super().__init__(base_url, config.timeout_s, "asr-verify", side=side, **transport)
        self.config = config
        self.encoding = encoding

    async def transcribe(
        self,
        /,
        payload: AudioPayload,
        language: str | None = None,
    ) -> VerifyResult:
        params = {
            "language": _to_whisper_language(language),
            "beam_size": self.config.beam_size,
        }

        if self.side == "local":
            if payload.audio_reference is None:
                raise ServiceError("local verify client requires a shared-memory reference")
            result = await self._post_json(
                "/transcribe",
                {"audio": payload.audio_reference.to_json(), **params},
            )
        else:
            result = await self._post_multipart(
                "/transcribe/upload",
                files={
                    "audio": ("utt.pcm", payload.encoded(self.encoding), "application/octet-stream")
                },
                data={
                    "params": json.dumps(
                        {
                            **params,
                            "encoding": self.encoding,
                            "sample_rate": payload.sample_rate,
                        }
                    )
                },
            )

        return _parse_verification_response(result)


def _parse_verification_response(
    result: dict[str, Any],
    /,
) -> VerifyResult:
    """Validate bounded verification output before CER computation."""
    try:
        response = VerificationResponse.model_validate(result)
    except ValidationError as error:
        raise ServiceError("asr-verify returned an invalid response") from error
    return VerifyResult(
        text=response.text,
        avg_logprob=response.avg_logprob,
        language=response.language,
        infer_ms=response.infer_ms,
    )


def _to_whisper_language(
    language: str | None,
    /,
) -> str | None:
    """Map application language codes to Whisper language identifiers."""
    if language is None:
        return None
    return {"ko": "ko", "en": "en", "ja": "ja", "zh-TW": "zh"}.get(language)
