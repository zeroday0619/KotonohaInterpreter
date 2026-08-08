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

import httpx2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kotonoha._call_compatibility import keyword_compatible
from kotonoha._config import Settings, load_settings, local_config_path, read_yaml
from kotonoha._config_store import apply_changes, get_path
from kotonoha._logging_setup import get_logger, setup_logging
from kotonoha.store._db import Store
from kotonoha.tui._config_app import FIELDS, SECTION_LABELS, SECTIONS, effective_value
from kotonoha.tui._history_app import OUTCOMES
from kotonoha.tui._license_app import (
    installed_direct_dependencies,
    project_license_text,
    project_version,
)
from kotonoha.tui._tools_app import OPERATION_DESCRIPTIONS, OPERATION_FIELDS, OPERATIONS
from kotonoha.web._logs import LogBroadcaster
from kotonoha.web._operations import OperationManager
from kotonoha.web._session import DEFAULT_MAXIMUM_SESSIONS, Session, SessionManager

log = get_logger(__name__)

STATIC_ROOT: Final = Path(__file__).resolve().parent / "static"
# A browser block is 128 samples per channel; anything far above one utterance is
# a client that is not speaking the protocol.
MAXIMUM_AUDIO_BYTES: Final = 4 * 16000 * 60
MODEL_ROLE_PREFIXES: Final[dict[str, tuple[str, ...]]] = {
    "asr": ("accelerator.", "asr."),
    "asr_verify": ("accelerator.", "asr_verify."),
    "llm": ("accelerator.", "llm."),
    "tts": ("accelerator.", "tts."),
}


class ConfigurationUpdate(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    changes: dict[str, Any] = Field(default_factory=dict)


class OperationRequest(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    operation: str
    values: dict[str, str] = Field(default_factory=dict)


class WebState:
    """Process-wide objects shared by every connection."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "configuration_lock",
        "logs",
        "operations",
        "reload_tasks",
        "sessions",
        "settings",
        "config_path",
        "local_path",
    )

    settings: Settings
    sessions: SessionManager
    logs: LogBroadcaster
    operations: OperationManager
    config_path: Path | None
    local_path: Path
    configuration_lock: asyncio.Lock
    reload_tasks: set[asyncio.Task[None]]

    def __init__(
        self,
        /,
        settings: Settings,
        maximum_sessions: int,
        config_path: Path | None,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.local_path = local_config_path()
        self.configuration_lock = asyncio.Lock()
        self.reload_tasks = set()
        self.sessions = SessionManager(settings, maximum_sessions)
        self.logs = LogBroadcaster()
        self.operations = OperationManager(config_path)


def create_app(
    settings: Settings | None = None,
    /,
    maximum_sessions: int = DEFAULT_MAXIMUM_SESSIONS,
    config_path: Path | None = None,
) -> FastAPI:
    state = WebState(settings or load_settings(config_path), maximum_sessions, config_path)

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
            for task in state.reload_tasks:
                task.cancel()
            if state.reload_tasks:
                await asyncio.gather(*state.reload_tasks, return_exceptions=True)
            await state.logs.stop()
            await state.operations.close()
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

    @application.get("/api/config")
    @keyword_compatible
    async def configuration() -> dict[str, Any]:
        return _configuration_snapshot(state)

    @application.put("/api/config")
    @keyword_compatible
    async def update_configuration(
        update: ConfigurationUpdate,
        /,
    ) -> dict[str, Any]:
        async with state.configuration_lock:
            return await _apply_configuration_update(state, update)

    @application.get("/api/history")
    @keyword_compatible
    async def history(
        query: str | None = None,
        source_language: str | None = None,
        outcome: str | None = None,
        offset: int = 0,
        limit: int = 200,
        /,
    ) -> dict[str, Any]:
        if outcome is not None and outcome not in OUTCOMES:
            raise HTTPException(422, f"unsupported history outcome: {outcome}")
        return await asyncio.to_thread(
            _history_snapshot,
            state.settings,
            query,
            source_language,
            outcome,
            max(0, offset),
            max(1, min(limit, 200)),
        )

    @application.delete("/api/history")
    @keyword_compatible
    async def clear_history(
        session: str | None = None,
        /,
    ) -> dict[str, Any]:
        removed = await asyncio.to_thread(_clear_history, state.settings, session)
        return {"removed": removed}

    @application.get("/api/license")
    @keyword_compatible
    async def license_information() -> dict[str, Any]:
        version, license_text, dependencies = await asyncio.gather(
            asyncio.to_thread(project_version),
            asyncio.to_thread(project_license_text),
            asyncio.to_thread(installed_direct_dependencies),
        )
        return {
            "version": version,
            "license": license_text,
            "dependencies": [
                {
                    "name": dependency.name,
                    "version": dependency.version,
                    "license": dependency.license_name,
                }
                for dependency in dependencies
            ],
        }

    @application.get("/api/operations")
    @keyword_compatible
    async def operation_status() -> dict[str, Any]:
        return {
            "operations": [
                {
                    "name": operation,
                    "fields": list(OPERATION_FIELDS[operation]),
                    "description": OPERATION_DESCRIPTIONS[operation],
                }
                for operation in OPERATIONS
            ],
            "job": state.operations.snapshot(),
        }

    @application.post("/api/operations")
    @keyword_compatible
    async def start_operation(
        request: OperationRequest,
        /,
    ) -> dict[str, Any]:
        try:
            return await state.operations.start(request.operation, request.values)
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @application.delete("/api/operations")
    @keyword_compatible
    async def stop_operation() -> dict[str, Any]:
        return await state.operations.stop()

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


def _configuration_snapshot(
    state: WebState,
    /,
) -> dict[str, Any]:
    overrides = read_yaml(state.local_path) if state.local_path.exists() else {}
    fields = []
    for specification in FIELDS:
        value = effective_value(state.settings, specification.path)
        secret = specification.path == "remote.token"
        fields.append(
            {
                "path": specification.path,
                "section": specification.section,
                "kind": specification.kind,
                "choices": list(specification.choices),
                "optional": specification.optional,
                "value_kind": specification.value_kind,
                "value": "" if secret else _json_value(value),
                "modified": get_path(overrides, specification.path) is not None,
                "secret": secret,
            }
        )
    return {
        "path": str(state.local_path),
        "sections": [
            {"name": section, "label": SECTION_LABELS[section]} for section in SECTIONS
        ],
        "fields": fields,
        "reloading_roles": sorted(
            task.get_name().removeprefix("reload-") for task in state.reload_tasks
        ),
    }


async def _apply_configuration_update(
    state: WebState,
    update: ConfigurationUpdate,
    /,
) -> dict[str, Any]:
    editable_paths = {specification.path for specification in FIELDS}
    rejected = sorted(set(update.changes) - editable_paths)
    if rejected:
        raise HTTPException(
            422,
            f"unknown or non-editable settings: {', '.join(rejected)}",
        )
    result = await asyncio.to_thread(
        apply_changes,
        update.changes,
        state.config_path,
        state.local_path,
    )
    if not result.written:
        raise HTTPException(422, result.error or "invalid configuration")
    settings = await asyncio.to_thread(load_settings, state.config_path)
    state.settings = settings
    if any(path.startswith("logging.") for path in result.changed):
        _apply_runtime_logging(settings)
    retired_sessions = await state.sessions.replace_settings(settings)
    reloading_roles = _changed_model_roles(result.changed)
    for role in reloading_roles:
        task = asyncio.create_task(
            _reload_model_role(settings, role),
            name=f"reload-{role}",
        )
        state.reload_tasks.add(task)
        task.add_done_callback(state.reload_tasks.discard)
    snapshot = _configuration_snapshot(state)
    snapshot["changed"] = result.changed
    snapshot["retired_sessions"] = retired_sessions
    snapshot["reloading_roles"] = reloading_roles
    return snapshot


def _json_value(
    value: Any,
    /,
) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    return value


def _apply_runtime_logging(
    settings: Settings,
    /,
) -> None:
    logging = settings.logging
    setup_logging(
        logging.effective_level(),
        settings.resolve(logging.log_path),
        logging.console,
        "web",
        terminal_interface=True,
        maximum_bytes=logging.max_bytes,
        backup_count=logging.backup_count,
        console_format=logging.console_format,
    )


def _history_snapshot(
    settings: Settings,
    query: str | None,
    source_language: str | None,
    outcome: str | None,
    offset: int,
    limit: int,
    /,
) -> dict[str, Any]:
    if outcome is not None and outcome not in OUTCOMES:
        raise ValueError(f"unsupported history outcome: {outcome}")
    store = Store(settings.resolve(settings.store.path))
    try:
        parameters = {
            "query": query or None,
            "src_lang": source_language or None,
            "outcome": outcome or None,
        }
        return {
            "total": store.count_turns(**parameters),
            "entries": [
                entry.as_dict()
                for entry in store.search_turns(
                    **parameters,
                    limit=limit,
                    offset=offset,
                )
            ],
            "languages": store.history_languages(),
            "outcomes": list(OUTCOMES),
        }
    finally:
        store.close()


def _clear_history(
    settings: Settings,
    session: str | None,
    /,
) -> int:
    store = Store(settings.resolve(settings.store.path))
    try:
        return store.clear_history(session)
    finally:
        store.close()


def _changed_model_roles(
    changed_paths: list[str],
    /,
) -> list[str]:
    return [
        role
        for role, prefixes in MODEL_ROLE_PREFIXES.items()
        if any(path.startswith(prefixes) for path in changed_paths)
    ]


async def _reload_model_role(
    settings: Settings,
    role: str,
    /,
) -> None:
    side = settings.resolved_placement().get(role, "local")
    url = settings.url_for(role, side).rstrip("/")
    token = (settings.remote.token or "") if side == "remote" else ""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with httpx2.AsyncClient(headers=headers, timeout=600.0) as client:
            response = await client.post(f"{url}/admin/reload")
            response.raise_for_status()
        log.info("web.model_reloaded", role=role, side=side)
    except Exception as error:  # noqa: BLE001
        log.error("web.model_reload_failed", role=role, side=side, error=repr(error))


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
