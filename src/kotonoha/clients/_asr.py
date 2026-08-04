"""Primary ASR client (Qwen3-ASR 1.7B, N-best 5 with LID)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar

from kotonoha._config import AsrConfig
from kotonoha._transport import AudioPayload, Encoding
from kotonoha._typing import override
from kotonoha.clients._base import BaseClient, ServiceError


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

        return AsrResult(
            hypotheses=[
                Hypothesis(
                    text=hypothesis["text"],
                    avg_logprob=float(hypothesis.get("avg_logprob", -99.0)),
                )
                for hypothesis in result.get("hypotheses", [])
            ],
            language=result.get("language"),
            language_confidence=(
                float(result["language_confidence"])
                if result.get("language_confidence") is not None
                else None
            ),
            duration_s=float(result.get("duration_s", payload.seconds)),
            infer_ms=float(result.get("infer_ms", 0.0)),
        )
