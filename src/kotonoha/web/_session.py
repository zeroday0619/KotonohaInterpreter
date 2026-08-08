"""One interpreter session per browser connection.

Every session owns an orchestrator, a capture fed from its WebSocket and a
playback that writes back to it. Two things must not be shared between sessions:

  · the shared-memory audio ring, whose name is a process-wide POSIX identifier.
    Two sessions publishing into one ring would hand each other's audio to ASR.
  · the session identifier, which keys conversation history and the turn log.

The resident model services are shared on purpose. They hold the weights, and a
second copy would not fit on the device. Sessions therefore queue against each
other at the services rather than running truly in parallel, which is why the
manager caps how many can exist at once.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, ClassVar, Final

import numpy as np

from kotonoha._config import Settings
from kotonoha._logging_setup import get_logger
from kotonoha.audio._vad import UtteranceSegmenter, build_vad
from kotonoha.core._orchestrator import Orchestrator
from kotonoha.web._audio import BrowserCapture, BrowserPlayback

log = get_logger(__name__)

DEFAULT_MAXIMUM_SESSIONS: Final = 4


class Session:
    """An orchestrator plus the two adapters that bind it to one browser."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "capture",
        "identifier",
        "orchestrator",
        "outbound",
        "playback",
        "settings",
    )

    identifier: str
    settings: Settings
    capture: BrowserCapture
    playback: BrowserPlayback
    orchestrator: Orchestrator
    outbound: asyncio.Queue[Any]

    def __init__(
        self,
        /,
        identifier: str,
        settings: Settings,
    ) -> None:
        self.identifier = identifier
        # A deep copy keeps the per-session shared-memory name off the shared
        # settings object; mutating the original would rename every other ring.
        self.settings = settings.model_copy(deep=True)
        self.settings.shm.name = f"{settings.shm.name}_{identifier}"
        self.outbound = asyncio.Queue(maxsize=256)

        self.capture = BrowserCapture(work_sample_rate=self.settings.audio.work_sample_rate)
        self.playback = BrowserPlayback(
            self._send_audio,
            self._send_control,
            sample_rate=self.settings.tts.sample_rate,
        )

        vad_config = self.settings.frontend.vad
        segmenter = UtteranceSegmenter(
            vad=build_vad(
                vad_config.backend,
                self.settings.resolve(vad_config.model_path),
                self.settings.audio.work_sample_rate,
            ),
            sample_rate=self.settings.audio.work_sample_rate,
            threshold=vad_config.threshold,
            neg_threshold=vad_config.neg_threshold,
            preroll_ms=vad_config.preroll_ms,
            min_speech_ms=vad_config.min_speech_ms,
            silence_ms=vad_config.silence_ms,
            max_utterance_ms=vad_config.max_utterance_ms,
        )
        self.orchestrator = Orchestrator(
            self.settings,
            self.capture,
            segmenter,
            self.playback,
            None,  # browsers already apply their own noise suppression
            session_id=identifier,
        )

    def _send_audio(
        self,
        /,
        pcm: np.ndarray,
        rate: int,
    ) -> None:
        del rate
        self._offer(np.asarray(pcm, dtype=np.float32).tobytes())

    def _send_control(
        self,
        /,
        message: dict[str, Any],
    ) -> None:
        self._offer(message)

    def _offer(
        self,
        /,
        item: Any,
    ) -> None:
        try:
            self.outbound.put_nowait(item)
        except asyncio.QueueFull:
            # Dropping the oldest keeps a stalled browser from blocking the
            # orchestrator, which runs on the same event loop.
            try:
                self.outbound.get_nowait()
                self.outbound.put_nowait(item)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                log.warning("web.outbound_dropped", session=self.identifier)

    async def start(
        self,
        /,
    ) -> None:
        await self.orchestrator.start()

    async def stop(
        self,
        /,
    ) -> None:
        await self.orchestrator.stop()


class SessionManager:
    """Create, track and retire browser sessions."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "_lock",
        "_sessions",
        "maximum_sessions",
        "settings",
    )

    settings: Settings
    maximum_sessions: int
    _sessions: dict[str, Session]
    _lock: asyncio.Lock

    def __init__(
        self,
        /,
        settings: Settings,
        maximum_sessions: int = DEFAULT_MAXIMUM_SESSIONS,
    ) -> None:
        self.settings = settings
        self.maximum_sessions = max(1, maximum_sessions)
        self._sessions = {}
        self._lock = asyncio.Lock()

    @property
    def count(
        self,
        /,
    ) -> int:
        return len(self._sessions)

    def identifiers(
        self,
        /,
    ) -> list[str]:
        return sorted(self._sessions)

    async def create(
        self,
        /,
    ) -> Session:
        async with self._lock:
            if len(self._sessions) >= self.maximum_sessions:
                raise RuntimeError(
                    f"session limit reached ({self.maximum_sessions});"
                    " the model services are shared by every session"
                )
            identifier = uuid.uuid4().hex[:12]
            session = Session(identifier, self.settings)
            self._sessions[identifier] = session
        try:
            await session.start()
        except BaseException:
            async with self._lock:
                self._sessions.pop(identifier, None)
            raise
        log.info("web.session_started", session=identifier, sessions=len(self._sessions))
        return session

    async def close(
        self,
        /,
        identifier: str,
    ) -> None:
        async with self._lock:
            session = self._sessions.pop(identifier, None)
        if session is None:
            return
        try:
            await session.stop()
        finally:
            log.info("web.session_closed", session=identifier, sessions=len(self._sessions))

    async def close_all(
        self,
        /,
    ) -> None:
        for identifier in list(self._sessions):
            await self.close(identifier)
