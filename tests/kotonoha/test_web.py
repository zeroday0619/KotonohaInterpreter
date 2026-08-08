"""Browser interpreter front end: audio adapters, sessions, logs, and routes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kotonoha._config import load_settings
from kotonoha._shmring import _owned_shared_memory_names
from kotonoha.store._db import Store
from kotonoha.tui._config_app import FIELDS
from kotonoha.web._audio import FRAME_SAMPLES, BrowserCapture, BrowserPlayback
from kotonoha.web._logs import LogBroadcaster
from kotonoha.web._server import _changed_model_roles, create_app
from kotonoha.web._session import SessionManager


class RecordingSink:
    """Collects what the playback would have sent to a browser."""

    __slots__: ClassVar[tuple[str, ...]] = ("audio", "control")

    def __init__(
        self,
        /,
    ) -> None:
        self.audio: list[np.ndarray] = []
        self.control: list[dict[str, Any]] = []

    def send_audio(
        self,
        /,
        pcm: np.ndarray,
        rate: int,
    ) -> None:
        del rate
        self.audio.append(pcm)

    def send_control(
        self,
        /,
        message: dict[str, Any],
    ) -> None:
        self.control.append(message)


# -- capture ---------------------------------------------------------------


def test_capture_emits_whole_segmenter_frames() -> None:
    capture = BrowserCapture(work_sample_rate=16000)
    capture.open_gate()

    capture.push(np.zeros(FRAME_SAMPLES + 100, dtype=np.float32))

    assert capture.frames.qsize() == 1
    frame = capture.frames.get_nowait()
    assert frame.pcm.size == FRAME_SAMPLES

    # The remaining 100 samples stay buffered until the frame is complete.
    capture.push(np.zeros(FRAME_SAMPLES - 100, dtype=np.float32))
    assert capture.frames.qsize() == 1


def test_capture_drops_audio_while_the_gate_is_shut() -> None:
    """A client that keeps streaming during SPEAKING must not reach the segmenter.

    Otherwise synthesized speech re-enters the microphone and the turn loops.
    """
    capture = BrowserCapture(work_sample_rate=16000)
    capture.open_gate()
    capture.close_gate()

    capture.push(np.ones(FRAME_SAMPLES * 4, dtype=np.float32))

    assert capture.frames.qsize() == 0


def test_capture_resamples_from_the_rate_the_client_reports() -> None:
    """A browser may refuse the requested context rate; the pitch must survive."""
    capture = BrowserCapture(work_sample_rate=16000)
    capture.set_source_rate(48000)
    capture.open_gate()

    capture.push(np.zeros(48000, dtype=np.float32))

    # One second at 48 kHz is 16000 working samples, so about 31 whole frames.
    produced = capture.frames.qsize()
    assert produced > 0
    assert produced <= 16000 // FRAME_SAMPLES + 1


def test_capture_reports_no_original_rate_audio_for_denoise() -> None:
    capture = BrowserCapture()

    assert capture.tail48(1024).size == 0


# -- playback --------------------------------------------------------------


async def test_playback_drains_only_after_the_client_confirms() -> None:
    """The orchestrator reopens the microphone when playback drains.

    Draining on send rather than on confirmation would reopen it while the
    browser is still speaking, which is the half-duplex failure this guards.
    """
    sink = RecordingSink()
    playback = BrowserPlayback(sink.send_audio, sink.send_control, sample_rate=24000)

    playback.begin_turn()
    playback.enqueue(np.ones(2400, dtype=np.float32), 24000)
    playback.finish_turn()

    assert not playback.drained.is_set()
    assert playback.pending_seconds == 0.1
    assert await playback.wait_drained(timeout=0.05) is False

    playback.acknowledge(2400)

    assert playback.drained.is_set()
    assert playback.pending_seconds == 0.0
    assert await playback.wait_drained(timeout=0.05) is True


async def test_playback_flush_releases_a_waiting_turn() -> None:
    sink = RecordingSink()
    playback = BrowserPlayback(sink.send_audio, sink.send_control, sample_rate=24000)
    playback.begin_turn()
    playback.enqueue(np.ones(24000, dtype=np.float32), 24000)

    playback.flush()

    assert playback.drained.is_set()
    assert await playback.wait_drained(timeout=0.05) is True
    assert {message["type"] for message in sink.control} >= {"playback_begin", "playback_flush"}


async def test_playback_first_packet_marks_the_turn() -> None:
    sink = RecordingSink()
    playback = BrowserPlayback(sink.send_audio, sink.send_control)
    playback.begin_turn()

    assert not playback.first_packet.is_set()
    playback.enqueue(np.ones(240, dtype=np.float32), 24000)
    assert playback.first_packet.is_set()


async def test_playback_bounded_enqueue_waits_for_the_client() -> None:
    sink = RecordingSink()
    playback = BrowserPlayback(sink.send_audio, sink.send_control, sample_rate=24000)
    playback.begin_turn()

    task = asyncio.create_task(
        playback.enqueue_bounded(
            np.ones(24000, dtype=np.float32),
            rate=24000,
            maximum_seconds=0.1,
        )
    )
    await asyncio.sleep(0)
    for acknowledged in range(2400, 24001, 2400):
        playback.acknowledge(acknowledged)
        await asyncio.sleep(0)
    await asyncio.wait_for(task, timeout=2)

    assert sum(chunk.size for chunk in sink.audio) == 24000


# -- sessions --------------------------------------------------------------


async def test_sessions_do_not_share_a_shared_memory_ring() -> None:
    """Two sessions publishing into one ring would hand each other's audio to ASR."""
    manager = SessionManager(load_settings(), maximum_sessions=2)
    first = await manager.create()
    second = await manager.create()
    try:
        assert first.settings.shm.name != second.settings.shm.name
        assert first.orchestrator.session_id != second.orchestrator.session_id
    finally:
        await manager.close_all()

    assert manager.count == 0
    for session in (first, second):
        assert session.settings.shm.name not in _owned_shared_memory_names


async def test_session_limit_is_enforced() -> None:
    manager = SessionManager(load_settings(), maximum_sessions=1)
    await manager.create()
    try:
        try:
            await manager.create()
            raise AssertionError("the session limit was not enforced")
        except RuntimeError as error:
            assert "session limit" in str(error)
    finally:
        await manager.close_all()


# -- logs ------------------------------------------------------------------


def test_every_subscriber_receives_every_record() -> None:
    """Draining the shared buffer per session would split the stream between them."""
    broadcaster = LogBroadcaster()
    first = broadcaster.subscribe()
    second = broadcaster.subscribe()

    broadcaster.publish("[    1.000000] INFO    asr: asr.started")

    assert first.get_nowait() == second.get_nowait()


def test_a_late_subscriber_receives_the_recent_history() -> None:
    broadcaster = LogBroadcaster()
    broadcaster.publish("[    1.000000] INFO    asr: asr.started")

    queue = broadcaster.subscribe()

    assert queue.get_nowait().endswith("asr.started")


# -- routes ----------------------------------------------------------------


def test_web_routes_serve_the_client() -> None:
    client = TestClient(create_app())

    assert client.get("/health").json()["ok"] is True
    assert client.get("/").status_code == 200
    for asset in ("app.js", "app.css", "capture-worklet.js"):
        assert client.get(f"/static/{asset}").status_code == 200, asset


def test_web_exposes_every_control_center_page() -> None:
    client = TestClient(create_app())
    document = client.get("/").text

    for page in (
        "interpreter",
        "monitoring",
        "configuration",
        "history-page",
        "operations",
        "license",
    ):
        assert f'id="{page}"' in document
    assert 'id="theme-select"' in document
    assert 'data-theme="system"' in document
    configuration = client.get("/api/config").json()
    assert {field["path"] for field in configuration["fields"]} == {
        field.path for field in FIELDS
    }


def test_web_client_supports_mobile_layout_and_configuration_navigation() -> None:
    client = TestClient(create_app())

    document = client.get("/").text
    stylesheet = client.get("/static/app.css").text
    script = client.get("/static/app.js").text

    assert "viewport-fit=cover" in document
    assert "@media (max-width: 720px)" in stylesheet
    assert "[hidden]" in stylesheet
    assert ".settings-section[hidden]" in stylesheet
    assert "min-height: 2.75rem" in stylesheet
    assert 'button.setAttribute("aria-controls"' in script
    assert "panel.scrollIntoView" in script
    assert 'window.matchMedia("(prefers-reduced-motion: reduce)")' in script
    assert 'id="monitor-service-grid"' in document
    assert 'id="monitor-memory-chart"' in document
    assert "renderMonitoringCharts" in script


def test_web_always_exposes_unified_metrics_and_monitoring_api() -> None:
    client = TestClient(create_app())

    metrics = client.get("/metrics")
    monitoring = client.get("/api/monitoring?window_seconds=300")

    assert metrics.status_code == 200
    assert "kotonoha_http_requests_total" in metrics.text
    assert monitoring.status_code == 200
    assert monitoring.json()["window_seconds"] == 300
    assert monitoring.json()["summary"]["services_total"] == 4
    assert monitoring.json()["sample_interval_seconds"] == 5.0


def test_web_configuration_save_is_validated_and_applied_without_process_restart(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_path = tmp_path / "web.local.yaml"
    monkeypatch.setenv("KOTONOHA_LOCAL_CONFIG", str(local_path))
    application = create_app()
    client = TestClient(application)

    response = client.put(
        "/api/config",
        json={"changes": {"session.fixed_target": "ja"}},
    )

    assert response.status_code == 200
    assert response.json()["changed"] == ["session.fixed_target"]
    assert response.json()["retired_sessions"] == 0
    assert application.state.kotonoha.settings.session.fixed_target == "ja"
    assert "fixed_target: ja" in local_path.read_text(encoding="utf-8")

    invalid = client.put(
        "/api/config",
        json={"changes": {"frontend.vad.preroll_ms": 10}},
    )
    assert invalid.status_code == 422

    unknown = client.put("/api/config", json={"changes": {"unknown.setting": True}})
    assert unknown.status_code == 422


def test_web_history_and_license_pages_use_the_runtime_store(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    settings = load_settings()
    settings.store.path = tmp_path / "history.db"
    store = Store(settings.store.path)
    store.add_turn("turn", "session", "ko", "ja", "source", "translation")
    store.close()
    client = TestClient(create_app(settings))

    history = client.get("/api/history").json()
    assert history["total"] == 1
    assert history["entries"][0]["source_text"] == "source"
    assert client.get("/api/license").json()["version"]
    assert client.delete("/api/history").json()["removed"] == 1


def test_web_operations_page_reuses_the_validated_tui_catalog() -> None:
    client = TestClient(create_app())

    payload = client.get("/api/operations").json()

    assert {operation["name"] for operation in payload["operations"]} >= {
        "doctor",
        "replay",
        "netcheck",
        "glossary_import",
    }


def test_model_setting_changes_reload_only_the_affected_services() -> None:
    assert _changed_model_roles(["llm.max_model_len"]) == ["llm"]
    assert _changed_model_roles(["session.fixed_target"]) == []
    assert _changed_model_roles(["accelerator.profile"]) == [
        "asr",
        "asr_verify",
        "llm",
        "tts",
    ]


def test_session_socket_announces_the_audio_contract() -> None:
    """The client cannot pick its rates without them, and a wrong rate shifts pitch."""
    client = TestClient(create_app(maximum_sessions=1))

    with client.websocket_connect("/ws") as socket:
        hello = json.loads(socket.receive_text())

    assert hello["type"] == "session"
    assert hello["capture_rate"] == 16000
    assert hello["playback_rate"] == 24000
    assert hello["session"]


def test_socket_refuses_a_session_beyond_the_limit() -> None:
    client = TestClient(create_app(maximum_sessions=1))

    with client.websocket_connect("/ws") as first:
        json.loads(first.receive_text())
        with client.websocket_connect("/ws") as second:
            refusal = json.loads(second.receive_text())

    assert refusal["type"] == "error"
    assert "session limit" in refusal["message"]
