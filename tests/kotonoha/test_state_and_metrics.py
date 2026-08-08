from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from kotonoha._config import LatencyBudgetConfig, load_settings
from kotonoha._metrics import TurnLog, TurnMetrics
from kotonoha._typing import override
from kotonoha.audio._playback import NullPlayback
from kotonoha.core._events import EventBus
from kotonoha.core._orchestrator import Orchestrator
from kotonoha.core._state import IllegalTransition, Machine, State


class SilentLanguageModel:
    __slots__: ClassVar[tuple[str, ...]] = ()

    async def stream(
        self,
        request_factory: Any,
        /,
    ) -> Any:
        del request_factory
        await asyncio.Event().wait()
        if False:
            yield ""


class EmptyLanguageModel:
    __slots__: ClassVar[tuple[str, ...]] = ()

    async def stream(
        self,
        request_factory: Any,
        /,
    ) -> Any:
        del request_factory
        if False:
            yield ""


class ContinuousLanguageModel:
    __slots__: ClassVar[tuple[str, ...]] = ()

    async def stream(
        self,
        request_factory: Any,
        /,
    ) -> Any:
        del request_factory
        while True:
            yield "Sentence. "
            await asyncio.sleep(0)


class SingleChunkTextToSpeech:
    __slots__: ClassVar[tuple[str, ...]] = ()

    async def stream(
        self,
        stream_factory: Any,
        /,
    ) -> Any:
        del stream_factory
        yield np.ones(4, dtype=np.float32)


class FailingPlayback(NullPlayback):
    __slots__: ClassVar[tuple[str, ...]] = ()

    @override
    def enqueue(
        self,
        /,
        pcm: np.ndarray,
        rate: int | None = None,
    ) -> None:
        del pcm, rate
        raise RuntimeError("playback queue failure")


def test_happy_path_transitions() -> None:
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


def test_illegal_transition_is_rejected() -> None:
    m = Machine()
    with pytest.raises(IllegalTransition):
        m.to(State.SPEAKING, "skip")


def test_processing_can_bail_to_idle() -> None:
    """Empty transcripts and LLM timeouts return to IDLE without SPEAKING (§10)."""
    m = Machine()
    m.to(State.LISTENING)
    m.to(State.PROCESSING)
    m.to(State.IDLE, "empty_asr")
    assert m.state is State.IDLE


def test_force_idle_from_any_state() -> None:
    m = Machine()
    m.to(State.LISTENING)
    m.to(State.PROCESSING)
    m.to(State.SPEAKING)
    m.force_idle("crash")
    assert m.state is State.IDLE


def test_metrics_five_marks_and_budget() -> None:
    m = TurnMetrics()
    base = 1000.0
    m.timestamps.update(
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

    assert m.over_budget(LatencyBudgetConfig()) == {}  # inside the 2.9 s budget


def test_metrics_reports_which_stage_blew_the_budget() -> None:
    m = TurnMetrics()
    base = 0.0
    m.timestamps.update(
        {
            "eou": base,
            "asr_done": base + 2.0,
            "first_clause": base + 3.0,
            "first_audio": base + 3.5,
        }
    )
    over = m.over_budget(LatencyBudgetConfig())
    assert "asr" in over and "total_to_first_audio" in over
    assert over["asr"] == pytest.approx(1000.0, abs=1)


def test_turn_dict_carries_required_fields() -> None:
    m = TurnMetrics()
    m.mark("eou")
    m.lang_detected = "ko"
    m.lang_source = "inherited"
    m.asr_avg_logprob = -0.42
    m.cross_verify_fired = True
    m.audio_seconds = 3.2
    m.output_tokens = 41
    d = m.to_dict(LatencyBudgetConfig())
    for k in (
        "lang_detected", "lang_source", "lid_confidence", "asr_avg_logprob",
        "cross_verify_fired", "audio_seconds", "output_tokens",
    ):
        assert k in d


def test_input_signal_statistics_are_preserved_in_turn_notes() -> None:
    orchestrator = object.__new__(Orchestrator)
    metrics = TurnMetrics(audio_seconds=0.25)

    orchestrator._record_audio_statistics(
        np.array([0.0, 0.25, -0.5, 1.0], dtype=np.float32),
        metrics,
    )

    assert metrics.notes["input_peak_dbfs"] == 0.0
    assert metrics.notes["input_rms_dbfs"] == pytest.approx(-4.8, abs=0.1)
    assert metrics.notes["input_clipped_fraction"] == 0.25


async def test_first_clause_timeout_cancels_the_first_audio_watcher() -> None:
    settings = load_settings()
    settings.llm.timeout_s = 0.01
    orchestrator = object.__new__(Orchestrator)
    orchestrator.settings = settings
    orchestrator.store = SimpleNamespace(glossary_for=lambda *_arguments: [])
    orchestrator.playback = NullPlayback(settings.audio, settings.tts)
    orchestrator.playback.start(asyncio.get_running_loop())
    orchestrator.language_model = SilentLanguageModel()
    orchestrator.event_bus = EventBus()
    orchestrator.machine = Machine()
    orchestrator._turn_children = set()
    metrics = TurnMetrics()

    result = await asyncio.wait_for(
        orchestrator._translate_and_speak(
            metrics,
            ["안녕하세요"],
            None,
            "ko",
            "en",
            [],
        ),
        timeout=0.2,
    )

    assert result is None
    assert metrics.outcome == "llm_timeout"
    assert not orchestrator._turn_children


async def test_empty_translation_finishes_without_waiting_for_the_timeout() -> None:
    settings = load_settings()
    settings.llm.timeout_s = 1.0
    orchestrator = object.__new__(Orchestrator)
    orchestrator.settings = settings
    orchestrator.store = SimpleNamespace(glossary_for=lambda *_arguments: [])
    orchestrator.playback = NullPlayback(settings.audio, settings.tts)
    orchestrator.playback.start(asyncio.get_running_loop())
    orchestrator.language_model = EmptyLanguageModel()
    orchestrator.event_bus = EventBus()
    orchestrator.machine = Machine()
    orchestrator._turn_children = set()
    metrics = TurnMetrics()

    result = await asyncio.wait_for(
        orchestrator._translate_and_speak(
            metrics,
            ["안녕하세요"],
            None,
            "ko",
            "en",
            [],
        ),
        timeout=0.2,
    )

    errors = [
        event.payload["message"]
        for event in orchestrator.event_bus.drain_nowait(32)
        if event.kind == "error"
    ]
    assert result is None
    assert metrics.outcome == "llm_timeout"
    assert errors == ["translation produced no speakable clause"]
    assert not orchestrator._turn_children


async def test_speaker_failure_cancels_a_blocked_language_model() -> None:
    settings = load_settings()
    orchestrator = object.__new__(Orchestrator)
    orchestrator.settings = settings
    orchestrator.store = SimpleNamespace(glossary_for=lambda *_arguments: [])
    orchestrator.playback = FailingPlayback(settings.audio, settings.tts)
    orchestrator.playback.start(asyncio.get_running_loop())
    orchestrator.language_model = ContinuousLanguageModel()
    orchestrator.text_to_speech = SingleChunkTextToSpeech()
    orchestrator.services = SimpleNamespace(placement={})
    orchestrator.event_bus = EventBus()
    orchestrator.machine = Machine()
    orchestrator._turn_children = set()
    metrics = TurnMetrics()

    result = await asyncio.wait_for(
        orchestrator._translate_and_speak(
            metrics,
            ["안녕하세요"],
            None,
            "ko",
            "en",
            [],
        ),
        timeout=0.2,
    )

    assert result
    assert metrics.outcome == "tts_failed"
    assert not orchestrator._turn_children


async def test_cancelled_shutdown_finishes_resource_cleanup() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator._closed = False
    orchestrator._stopping = False
    orchestrator._stop_lock = asyncio.Lock()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def cleanup() -> None:
        cleanup_started.set()
        await release_cleanup.wait()
        cleanup_finished.set()

    orchestrator._stop_components = cleanup
    stop_task = asyncio.create_task(orchestrator.stop())
    await cleanup_started.wait()
    stop_task.cancel()
    await asyncio.sleep(0)

    assert not stop_task.done()
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await stop_task

    assert cleanup_finished.is_set()
    assert orchestrator._closed is True
    assert orchestrator._stopping is False


async def test_turn_log_rejects_a_symbolic_link(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    protected_file = tmp_path / "protected.txt"
    protected_file.write_text("preserve", encoding="utf-8")
    log_path = tmp_path / "turns.jsonl"
    log_path.symlink_to(protected_file)
    turn_log = TurnLog(log_path, LatencyBudgetConfig())

    with pytest.raises(OSError):
        await turn_log.write(TurnMetrics())

    assert protected_file.read_text(encoding="utf-8") == "preserve"


async def test_turn_log_rotates_at_the_configured_size(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    log_path = tmp_path / "turns.jsonl"
    turn_log = TurnLog(
        log_path,
        LatencyBudgetConfig(),
        maximum_bytes=128,
        backup_count=2,
    )

    await turn_log.write(TurnMetrics())
    await turn_log.write(TurnMetrics())

    assert log_path.exists()
    assert (tmp_path / "turns.jsonl.1").exists()


async def test_turn_finish_survives_metrics_and_history_storage_failures() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.settings = load_settings()
    orchestrator.services = SimpleNamespace(placement={}, all=lambda: [])
    orchestrator.session_id = "session"
    events: list[tuple[str, dict[str, Any]]] = []
    orchestrator.event_bus = SimpleNamespace(
        emit=lambda event, **payload: events.append((event, payload))
    )

    async def fail_turn_log(
        _metrics: TurnMetrics,
        /,
    ) -> dict[str, Any]:
        raise OSError("turn log unavailable")

    def fail_history(
        _positional_only: object | None = None,
        /,
        **_values: Any,
    ) -> float:
        raise OSError("history unavailable")

    orchestrator.turn_log = SimpleNamespace(write=fail_turn_log)
    orchestrator.store = SimpleNamespace(add_turn=fail_history)
    metrics = TurnMetrics()

    await orchestrator._finish(metrics, "source", "translation", "en")

    emitted_events = {event for event, _payload in events}
    assert "turn" in emitted_events
    assert "history" in emitted_events
