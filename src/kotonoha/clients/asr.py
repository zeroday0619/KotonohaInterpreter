"""Primary ASR client (Qwen3-ASR 1.7B, N-best 5 with LID)."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AsrCfg
from ..shmring import AudioRef
from .base import BaseClient


@dataclass
class Hypothesis:
    text: str
    avg_logprob: float


@dataclass
class AsrResult:
    hypotheses: list[Hypothesis]
    language: str | None
    language_confidence: float | None
    duration_s: float
    infer_ms: float

    @property
    def best(self) -> str:
        return self.hypotheses[0].text if self.hypotheses else ""

    @property
    def best_avg_logprob(self) -> float:
        return self.hypotheses[0].avg_logprob if self.hypotheses else -99.0

    @property
    def texts(self) -> list[str]:
        return [h.text for h in self.hypotheses]

    @property
    def is_empty(self) -> bool:
        return not any(h.text.strip() for h in self.hypotheses)


class AsrClient(BaseClient):
    def __init__(self, base_url: str, cfg: AsrCfg):
        super().__init__(base_url, cfg.timeout_s, "asr")
        self.cfg = cfg

    async def transcribe(
        self,
        ref: AudioRef,
        context: str = "",
        language_hint: str | None = None,
    ) -> AsrResult:
        payload = {
            "audio": ref.to_json(),
            "n_best": self.cfg.n_best,
            "num_beams": self.cfg.num_beams,
            "max_new_tokens": self.cfg.max_new_tokens,
            "context": context,
            "language_hint": language_hint,
        }
        d = await self._post_json("/transcribe", payload)
        return AsrResult(
            hypotheses=[
                Hypothesis(text=h["text"], avg_logprob=float(h.get("avg_logprob", -99.0)))
                for h in d.get("hypotheses", [])
            ],
            language=d.get("language"),
            language_confidence=(
                float(d["language_confidence"])
                if d.get("language_confidence") is not None
                else None
            ),
            duration_s=float(d.get("duration_s", ref.seconds)),
            infer_ms=float(d.get("infer_ms", 0.0)),
        )
