"""Primary ASR client for target-specific vLLM models with N-best 5 and LID."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from pydantic import BaseModel, Field, ValidationError

from kotonoha._config import AsrConfig
from kotonoha._transport import AudioPayload, Encoding
from kotonoha._typing import override
from kotonoha.clients._base import BaseClient, ServiceError

MAXIMUM_ASR_TEXT_CHARACTERS: Final[int] = 4096


class HypothesisResponse(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    text: str = Field("", max_length=MAXIMUM_ASR_TEXT_CHARACTERS)
    avg_logprob: float = Field(-99.0, allow_inf_nan=False)


class AsrResponse(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    hypotheses: list[HypothesisResponse] = Field(default_factory=list, max_length=5)
    language: str | None = Field(None, max_length=128)
    language_confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    duration_s: float | None = Field(None, ge=0.0, le=3600.0, allow_inf_nan=False)
    infer_ms: float = Field(0.0, ge=0.0, le=3_600_000.0, allow_inf_nan=False)


@dataclass(slots=True)
class Hypothesis:
    text: str
    avg_logprob: float


@dataclass(slots=True)
class AsrResult:
    hypotheses: list[Hypothesis]
    language: str | None
    language_confidence: float | None
    duration_s: float
    infer_ms: float

    @property
    def best(
        self,
        /,
    ) -> str:
        return self.hypotheses[0].text if self.hypotheses else ""

    @property
    def best_avg_logprob(
        self,
        /,
    ) -> float:
        return self.hypotheses[0].avg_logprob if self.hypotheses else -99.0

    @property
    def texts(
        self,
        /,
    ) -> list[str]:
        return [h.text for h in self.hypotheses]

    @property
    def is_empty(
        self,
        /,
    ) -> bool:
        return not any(h.text.strip() for h in self.hypotheses)


class AsrClient(BaseClient):
    __slots__: ClassVar[tuple[str, ...]] = (
        "config",
        "encoding",
    )
    config: AsrConfig
    encoding: Encoding

    @override
    def __init__(
        self,
        /,
        base_url: str,
        config: AsrConfig,
        *,
        side: str = "local",
        encoding: Encoding = "s16le",
        **transport: Any,
    ) -> None:
        super().__init__(base_url, config.timeout_s, "asr", side=side, **transport)
        self.config = config
        self.encoding = encoding

    async def transcribe(
        self,
        /,
        payload: AudioPayload,
        context: str = "",
        language_hint: str | None = None,
    ) -> AsrResult:
        params = {
            "n_best": self.config.n_best,
            "num_beams": self.config.num_beams,
            "max_new_tokens": self.config.max_new_tokens,
            "context": context,
            "language_hint": language_hint,
        }

        if self.side == "local":
            if payload.audio_reference is None:
                raise ServiceError("local asr client requires a shared-memory reference")
            result = await self._post_json(
                "/transcribe",
                {"audio": payload.audio_reference.to_json(), **params},
            )
        else:
            # Remote: no shared memory to attach to, so the PCM travels as a
            # binary part. Still not base64 (§3).
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

        return _parse_asr_response(result, payload.seconds)


def _parse_asr_response(
    result: dict[str, Any],
    /,
    fallback_duration_seconds: float,
) -> AsrResult:
    """Validate the service boundary before response data reaches quality logic."""
    try:
        response = AsrResponse.model_validate(result)
    except ValidationError as error:
        raise ServiceError("asr returned an invalid response") from error
    return AsrResult(
        hypotheses=[
            Hypothesis(
                text=hypothesis.text,
                avg_logprob=hypothesis.avg_logprob,
            )
            for hypothesis in response.hypotheses
        ],
        language=response.language,
        language_confidence=response.language_confidence,
        duration_s=(
            response.duration_s
            if response.duration_s is not None
            else fallback_duration_seconds
        ),
        infer_ms=response.infer_ms,
    )
