"""Cross-verification ASR client (faster-whisper large-v3).

Called conditionally on the Orin, per §5.5, because it costs 0.8 s there. Remote
deployments can select `asr_verify.mode: always` after measuring the added latency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from kotonoha.clients.base import BaseClient, ServiceError
from kotonoha.config import AsrVerificationConfig
from kotonoha.transport import AudioPayload, Encoding


@dataclass
class VerifyResult:
    text: str
    avg_logprob: float
    language: str | None
    infer_ms: float


class AsrVerifyClient(BaseClient):
    config: AsrVerificationConfig
    encoding: Encoding

    def __init__(
        self,
        base_url: str,
        config: AsrVerificationConfig,
        *,
        side: str = "local",
        encoding: Encoding = "s16le",
        **transport,
    ):
        super().__init__(base_url, config.timeout_s, "asr-verify", side=side, **transport)
        self.config = config
        self.encoding = encoding

    async def transcribe(self, payload: AudioPayload, language: str | None = None) -> VerifyResult:
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

        return VerifyResult(
            text=result.get("text", ""),
            avg_logprob=float(result.get("avg_logprob", -99.0)),
            language=result.get("language"),
            infer_ms=float(result.get("infer_ms", 0.0)),
        )


def _to_whisper_language(language: str | None) -> str | None:
    """Map application language codes to Whisper language identifiers."""
    if language is None:
        return None
    return {"ko": "ko", "en": "en", "ja": "ja", "zh-TW": "zh"}.get(language)
