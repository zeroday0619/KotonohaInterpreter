"""Instrumentation (§11) — five timestamps per turn.

    EOU detected -> ASR done -> first clause -> first audio packet -> queue drained

Without these five there is no way to tell where the latency is leaking.
Times come from time.perf_counter(); the log records them as milliseconds
relative to EOU.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import BudgetCfg

MARKS = ("eou", "asr_done", "first_clause", "first_audio", "queue_drained")


@dataclass
class TurnMetrics:
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    wall_start: float = field(default_factory=time.time)

    # The five marks, as monotonic perf_counter values.
    t: dict[str, float] = field(default_factory=dict)

    # Everything else §11 asks to record alongside them.
    lang_detected: str | None = None
    lang_source: str = "lid"  # lid | inherited | forced
    lid_confidence: float | None = None
    asr_avg_logprob: float | None = None
    cross_verify_fired: bool = False
    cross_verify_divergent: bool | None = None
    audio_seconds: float | None = None
    output_tokens: int | None = None
    target_lang: str | None = None
    llm_profile: str | None = None
    tok_per_s: float | None = None
    # High-performance mode: which side ran each role this turn, and whether the
    # link forced a failover. Without these, a turn that quietly ran on the
    # on-board fallback is indistinguishable from one that used the A6000.
    input_mode: str = "voice"  # voice | text
    perf_mode: str | None = None
    placement: dict[str, str] | None = None
    failovers: int = 0
    outcome: str = "ok"  # ok | empty_asr | llm_timeout | tts_failed | oom | aborted
    notes: dict[str, Any] = field(default_factory=dict)

    # -- marking ---------------------------------------------------------
    def mark(self, name: str) -> float:
        now = time.perf_counter()
        self.t.setdefault(name, now)
        return now

    def has(self, name: str) -> bool:
        return name in self.t

    def rel_ms(self, name: str) -> float | None:
        """Milliseconds since EOU."""
        if name not in self.t or "eou" not in self.t:
            return None
        return round((self.t[name] - self.t["eou"]) * 1000, 1)

    def span_ms(self, a: str, b: str) -> float | None:
        if a not in self.t or b not in self.t:
            return None
        return round((self.t[b] - self.t[a]) * 1000, 1)

    # -- budget comparison (§6) ------------------------------------------
    def stage_ms(self) -> dict[str, float | None]:
        return {
            "asr": self.span_ms("eou", "asr_done"),
            "llm_first_clause": self.span_ms("asr_done", "first_clause"),
            "tts_first_packet": self.span_ms("first_clause", "first_audio"),
            "total_to_first_audio": self.rel_ms("first_audio"),
            "playback": self.span_ms("first_audio", "queue_drained"),
        }

    def over_budget(self, budget: BudgetCfg) -> dict[str, float]:
        """Stages that blew the budget, and by how many ms. Empty means we fit."""
        stages = self.stage_ms()
        limits = {
            "asr": budget.asr + budget.verify,
            "llm_first_clause": budget.llm_first_clause,
            "tts_first_packet": budget.tts_first_packet,
            "total_to_first_audio": budget.total - budget.silence,
        }
        out: dict[str, float] = {}
        for k, lim in limits.items():
            v = stages.get(k)
            if v is not None and v > lim:
                out[k] = round(v - lim, 1)
        return out

    # -- serialisation ---------------------------------------------------
    def to_dict(self, budget: BudgetCfg | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {
            "turn_id": self.turn_id,
            "wall_start": self.wall_start,
            "marks_ms": {m: self.rel_ms(m) for m in MARKS if m in self.t},
            "stages_ms": self.stage_ms(),
            "lang_detected": self.lang_detected,
            "lang_source": self.lang_source,
            "lid_confidence": self.lid_confidence,
            "asr_avg_logprob": self.asr_avg_logprob,
            "cross_verify_fired": self.cross_verify_fired,
            "cross_verify_divergent": self.cross_verify_divergent,
            "audio_seconds": self.audio_seconds,
            "output_tokens": self.output_tokens,
            "target_lang": self.target_lang,
            "llm_profile": self.llm_profile,
            "tok_per_s": self.tok_per_s,
            "input_mode": self.input_mode,
            "perf_mode": self.perf_mode,
            "placement": self.placement,
            "failovers": self.failovers,
            "outcome": self.outcome,
        }
        if budget is not None:
            ob = self.over_budget(budget)
            d["over_budget_ms"] = ob
            d["within_budget"] = not ob
        if self.notes:
            d["notes"] = self.notes
        return d


class TurnLog:
    """Appends one JSONL line per turn."""

    def __init__(self, path: Path, budget: BudgetCfg):
        self.path = path
        self.budget = budget
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, m: TurnMetrics) -> dict[str, Any]:
        """Write one JSONL line and hand the record back for reuse.

        The file gets an "event" key; the returned record does not, because it
        would collide with structlog's own key in `log.info("turn", **rec)`.
        """
        rec = m.to_dict(self.budget)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "turn", **rec}, ensure_ascii=False) + "\n")
        return rec
