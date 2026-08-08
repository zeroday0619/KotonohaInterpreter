"""HTTP and WebSocket front end for the browser interpreter.

One WebSocket carries a session in both directions. Binary frames are audio and
text frames are control, which keeps microphone blocks off the JSON path where
base64 would cost latency and allocation on every block (§3).

  client -> server   binary  32-bit float PCM at the rate declared in `hello`
                     text    hello, ptt, mode, target, text, played, bye
  server -> client   binary  32-bit float TTS PCM at the playback rate
                     text    session, event, log, playback_begin/end/flush, error
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any, ClassVar, Final

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kotonoha._call_compatibility import keyword_compatible
from kotonoha._config import Settings, load_settings
from kotonoha._logging_setup import get_logger
from kotonoha.web._logs import LogBroadcaster
from kotonoha.web._session import DEFAULT_MAXIMUM_SESSIONS, Session, SessionManager

log = get_logger(__name__)

STATIC_ROOT: Final = Path(__file__).resolve().parent / "static"
# A browser block is 128 samples per channel; anything far above one utterance is
# a client that is not speaking the protocol.
MAXIMUM_AUDIO_BYTES: Final = 4 * 16000 * 60


class WebState:
    """Process-wide objects shared by every connection."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "logs",
        "sessions",
        "settings",
    )

    settings: Settings
    sessions: SessionManager
    logs: LogBroadcaster

    def __init__(
        self,
        /,
        settings: Settings,
        maximum_sessions: int,
    ) -> None:
        self.settings = settings
        self.sessions = SessionManager(settings, maximum_sessions)
        self.logs = LogBroadcaster()


def create_app(
    settings: Settings | None = None,
    /,
    maximum_sessions: int = DEFAULT_MAXIMUM_SESSIONS,
) -> FastAPI:
    state = WebState(settings or load_settings(), maximum_sessions)

    @contextlib.asynccontextmanager
    async def lifespan(
        application: FastAPI,
        /,
    ) -> Any:
        del application
        state.logs.start()
        try:
            yield
        finally:
            await state.logs.stop()
            await state.sessions.close_all()

    application = FastAPI(title="kotonoha-web", lifespan=lifespan)
    application.state.kotonoha = state

    @application.get("/health")
    @keyword_compatible
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "web",
            "sessions": state.sessions.count,
            "maximum_sessions": state.sessions.maximum_sessions,
        }

    @application.get("/api/sessions")
    @keyword_compatible
    async def sessions() -> dict[str, Any]:
        return {
            "sessions": state.sessions.identifiers(),
            "maximum_sessions": state.sessions.maximum_sessions,
        }

    @application.get("/api/logs")
    @keyword_compatible
    async def logs(
        limit: int = 200,
        /,
    ) -> dict[str, Any]:
        """Recent records for a client that wants a snapshot rather than a stream."""
        return {"lines": state.logs.history(max(0, min(limit, 400)))}

    @application.get("/")
    @keyword_compatible
    async def index() -> FileResponse:
        return FileResponse(STATIC_ROOT / "index.html")

    application.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

    @application.websocket("/ws")
    @keyword_compatible
    async def session_socket(
        websocket: WebSocket,
        /,
    ) -> None:
        await websocket.accept()
        try:
            session = await state.sessions.create()
        except RuntimeError as error:
            await websocket.send_text(json.dumps({"type": "error", "message": str(error)}))
            await websocket.close(code=1013)  # try again later
            return

        await websocket.send_text(
            json.dumps(
                {
                    "type": "session",
                    "session": session.identifier,
                    "capture_rate": session.settings.audio.work_sample_rate,
                    "playback_rate": session.playback.sample_rate,
                    "mode": session.settings.session.mode,
                    "languages": list(session.settings.session.languages),
                    "target": session.settings.session.fixed_target,
                }
            )
        )

        log_queue = state.logs.subscribe()
        pumps = [
            asyncio.create_task(_pump_outbound(websocket, session), name="web-outbound"),
            asyncio.create_task(_pump_events(websocket, session), name="web-events"),
            asyncio.create_task(_pump_logs(websocket, log_queue), name="web-logs"),
        ]
        try:
            await _receive_loop(websocket, session)
        except WebSocketDisconnect:
            pass
        except Exception as error:  # noqa: BLE001 - one client must not stop the server
            log.exception("web.session_failed", session=session.identifier, error=repr(error))
        finally:
            for pump in pumps:
                pump.cancel()
            for pump in pumps:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pump
            state.logs.unsubscribe(log_queue)
            await state.sessions.close(session.identifier)

    return application


async def _receive_loop(
    websocket: WebSocket,
    session: Session,
    /,
) -> None:
    orchestrator = session.orchestrator

    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return

        payload = message.get("bytes")
        if payload is not None:
            if len(payload) > MAXIMUM_AUDIO_BYTES or len(payload) % 4:
                continue
            session.capture.push(np.frombuffer(payload, dtype="<f4"))
            continue

        text = message.get("text")
        if text is None:
            continue
        try:
            command = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(command, dict):
            continue
        await _apply_command(websocket, session, orchestrator, command)


async def _apply_command(
    websocket: WebSocket,
    session: Session,
    orchestrator: Any,
    command: dict[str, Any],
    /,
) -> None:
    kind = command.get("type")
    if kind == "hello":
        rate = command.get("capture_rate")
        if isinstance(rate, int) and rate > 0:
            session.capture.set_source_rate(rate)
    elif kind == "ptt":
        if command.get("down"):
            orchestrator.ptt_down()
        else:
            orchestrator.ptt_up()
    elif kind == "mode":
        mode = command.get("mode")
        if mode in ("push_to_talk", "auto", "text"):
            orchestrator.settings.session.mode = mode
    elif kind == "target":
        language = command.get("language")
        if language in orchestrator.settings.session.languages:
            orchestrator.set_target_language(language)
    elif kind == "text":
        utterance = str(command.get("text") or "").strip()
        if utterance:
            await orchestrator.submit_text(utterance)
    elif kind == "played":
        samples = command.get("samples")
        if isinstance(samples, int):
            session.playback.acknowledge(samples)
    elif kind == "bye":
        await websocket.close()


async def _pump_outbound(
    websocket: WebSocket,
    session: Session,
    /,
) -> None:
    while True:
        item = await session.outbound.get()
        if isinstance(item, bytes):
            await websocket.send_bytes(item)
        else:
            await websocket.send_text(json.dumps(item))


async def _pump_events(
    websocket: WebSocket,
    session: Session,
    /,
) -> None:
    bus = session.orchestrator.event_bus
    while True:
        event = await bus.get()
        await websocket.send_text(
            json.dumps({"type": "event", "kind": event.kind, "payload": event.payload}, default=str)
        )


async def _pump_logs(
    websocket: WebSocket,
    queue: asyncio.Queue[str],
    /,
) -> None:
    while True:
        line = await queue.get()
        await websocket.send_text(json.dumps({"type": "log", "line": line}))
