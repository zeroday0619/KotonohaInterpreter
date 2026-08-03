"""Cross-verification ASR client (faster-whisper large-v3).

Called conditionally, per §5.5. Calling it every turn simply adds 0.8 s each time.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AsrVerifyCfg
from ..shmring import AudioRef
from .base import BaseClient


@dataclass
class VerifyResult:
    text: str
    avg_logprob: float
    language: str | None
    infer_ms: float


class AsrVerifyClient(BaseClient):
    def __init__(self, base_url: str, cfg: AsrVerifyCfg):
        super().__init__(base_url, cfg.timeout_s, "asr-verify")
        self.cfg = cfg

    async def transcribe(self, ref: AudioRef, language: str | None = None) -> VerifyResult:
        payload = {
            "audio": ref.to_json(),
            "language": _to_whisper_lang(language),
            "beam_size": self.cfg.beam_size,
        }
        d = await self._post_json("/transcribe", payload)
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
