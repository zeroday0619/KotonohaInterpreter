"""교차 검증 ASR 클라이언트 (faster-whisper large-v3).

§5.5 조건부 호출. 상시 호출하면 매 턴 0.8초가 그냥 붙는다.
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
    """내부 언어코드 → whisper 언어코드. zh-TW 는 whisper 에 zh 로 준다."""
    if lang is None:
        return None
    return {"ko": "ko", "en": "en", "ja": "ja", "zh-TW": "zh"}.get(lang)
