"""Cross-verification ASR client (faster-whisper large-v3).

Called conditionally on the Orin, per §5.5, because it costs 0.8 s there.
On the A6000 it is cheap enough to run every turn — see asr_verify.mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..config import AsrVerifyCfg
from ..transport import AudioPayload, Encoding
from .base import BaseClient, ServiceError


@dataclass
class VerifyResult:
    text: str
    avg_logprob: float
    language: str | None
    infer_ms: float


class AsrVerifyClient(BaseClient):
    def __init__(
        self,
        base_url: str,
        cfg: AsrVerifyCfg,
        *,
        side: str = "local",
        encoding: Encoding = "s16le",
        **transport,
    ):
        super().__init__(base_url, cfg.timeout_s, "asr-verify", side=side, **transport)
        self.cfg = cfg
        self.encoding = encoding

    async def transcribe(self, payload: AudioPayload, language: str | None = None) -> VerifyResult:
        params = {
            "language": _to_whisper_lang(language),
            "beam_size": self.cfg.beam_size,
        }

        if self.side == "local":
            if payload.ref is None:
                raise ServiceError("local verify client requires a shared-memory reference")
            d = await self._post_json("/transcribe", {"audio": payload.ref.to_json(), **params})
        else:
            d = await self._post_multipart(
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
            text=d.get("text", ""),
            avg_logprob=float(d.get("avg_logprob", -99.0)),
            language=d.get("language"),
            infer_ms=float(d.get("infer_ms", 0.0)),
        )


def _to_whisper_lang(lang: str | None) -> str | None:
    """Our language code to whisper's. zh-TW is handed to whisper as zh."""
    if lang is None:
        return None
    return {"ko": "ko", "en": "en", "ja": "ja", "zh-TW": "zh"}.get(lang)
