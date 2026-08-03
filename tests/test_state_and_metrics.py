from __future__ import annotations

import pytest

from kotonoha.config import BudgetCfg
from kotonoha.core.state import IllegalTransition, Machine, State
from kotonoha.metrics import TurnMetrics


def test_happy_path_transitions():
    seen = []
    m = Machine(on_change=lambda a, b, r: seen.append((a, b, r)))
    m.to(State.LISTENING, "vad")
    m.to(State.PROCESSING, "eou")
    m.to(State.SPEAKING, "first_clause")
    m.to(State.IDLE, "drained")
    assert [b for _, b, _ in seen] == [
        State.LISTENING,
        State.PROCESSING,
        State.SPEAKING,
        State.IDLE,
    ]


def test_illegal_transition_is_rejected():
    m = Machine()
    with pytest.raises(IllegalTransition):
        m.to(State.SPEAKING, "skip")


def test_processing_can_bail_to_idle():
    """Empty transcripts and LLM timeouts return to IDLE without SPEAKING (§10)."""
    m = Machine()
    m.to(State.LISTENING)
    m.to(State.PROCESSING)
    m.to(State.IDLE, "empty_asr")
    assert m.state is State.IDLE


def test_force_idle_from_any_state():
    m = Machine()
    m.to(State.LISTENING)
    m.to(State.PROCESSING)
    m.to(State.SPEAKING)
    m.force_idle("crash")
    assert m.state is State.IDLE


def test_metrics_five_marks_and_budget():
    m = TurnMetrics()
    base = 1000.0
    m.t.update(
        {
            "eou": base,
            "asr_done": base + 0.85,
            "first_clause": base + 1.4,
            "first_audio": base + 1.65,
            "queue_drained": base + 4.0,
        }
    )
    s = m.stage_ms()
    assert s["asr"] == pytest.approx(850.0, abs=1)
    assert s["llm_first_clause"] == pytest.approx(550.0, abs=1)
    assert s["tts_first_packet"] == pytest.approx(250.0, abs=1)
    assert s["total_to_first_audio"] == pytest.approx(1650.0, abs=1)

    assert m.over_budget(BudgetCfg()) == {}  # inside the 2.9 s budget


def test_metrics_reports_which_stage_blew_the_budget():
    m = TurnMetrics()
    base = 0.0
    m.t.update({"eou": base, "asr_done": base + 2.0, "first_clause": base + 3.0,
                "first_audio": base + 3.5})
    over = m.over_budget(BudgetCfg())
    assert "asr" in over and "total_to_first_audio" in over
    assert over["asr"] == pytest.approx(1000.0, abs=1)


def test_turn_dict_carries_required_fields():
    m = TurnMetrics()
    m.mark("eou")
    m.lang_detected = "ko"
    m.lang_source = "inherited"
    m.asr_avg_logprob = -0.42
    m.cross_verify_fired = True
    m.audio_seconds = 3.2
    m.output_tokens = 41
    d = m.to_dict(BudgetCfg())
    for k in (
        "lang_detected", "lang_source", "lid_confidence", "asr_avg_logprob",
        "cross_verify_fired", "audio_seconds", "output_tokens",
    ):
        assert k in d
