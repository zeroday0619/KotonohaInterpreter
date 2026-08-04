"""Routing a role to the A6000 or to the on-board service, with failover.

Every remote role keeps its on-board counterpart loaded. Transport failures
activate the on-board fallback so the current turn can complete.

The policy:

  · A transport failure (timeout, connection refused, 5xx) counts. Application
    errors do not — a 400 indicates an invalid request and retrying elsewhere
    would fail identically.
  · The failing call itself is retried once on the fallback, so the current turn
    survives rather than being lost to the failover.
  · After `failover_after` consecutive failures the role is marked degraded and
    goes to the on-board service. A background probe brings it back only after
    the remote has been healthy for `recover_after_s`, so a flapping link does
    not flip the placement every turn.
  · Streaming roles (LLM, TTS) fail over only before the first chunk. Once audio
    or text has started flowing there is no clean way to rewind, so a mid-stream
    failure is reported as one.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, ClassVar

from kotonoha._config import RemoteConfig
from kotonoha._logging_setup import get_logger
from kotonoha._typing import override
from kotonoha.clients._base import BaseClient, ServiceError, ServiceTimeout

log = get_logger(__name__)


class AllEndpointsFailed(ServiceError):
    __slots__: ClassVar[tuple[str, ...]] = ()
    pass


class FailoverClient:
    """One role: a preferred client and, when the preferred is remote, a local fallback."""
    __slots__: ClassVar[tuple[str, ...]] = (
        "_degraded",
        "_failure_count",
        "_healthy_since",
        "_on_change",
        "_probe_task",
        "config",
        "failover_count",
        "fallback",
        "preferred",
        "role",
    )

    role: str
    preferred: BaseClient
    fallback: BaseClient | None
    config: RemoteConfig
    failover_count: int
    _on_change: Callable[[str, str, str], None] | None
    _failure_count: int
    _degraded: bool
    _healthy_since: float | None
    _probe_task: asyncio.Task[None] | None

    @override
    def __init__(
        self,
        /,
        role: str,
        preferred: BaseClient,
        fallback: BaseClient | None,
        config: RemoteConfig,
        on_change: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self.role = role
        self.preferred = preferred
        self.fallback = fallback
        self.config = config
        self._on_change = on_change

        self._failure_count = 0
        self._degraded = False
        self._healthy_since: float | None = None
        self._probe_task = None
        self.failover_count = 0

    # -- state -------------------------------------------------------------
    @property
    def active(
        self,
        /,
    ) -> BaseClient:
        if self._degraded and self.fallback is not None:
            return self.fallback
        return self.preferred

    @property
    def side(
        self,
        /,
    ) -> str:
        return self.active.side

    @property
    def degraded(
        self,
        /,
    ) -> bool:
        return self._degraded

    @property
    def name(
        self,
        /,
    ) -> str:
        return self.preferred.name

    def status(
        self,
        /,
    ) -> dict[str, Any]:
        return {
            "role": self.role,
            "side": self.side,
            "preferred": self.preferred.side,
            "degraded": self._degraded,
            "failovers": self.failover_count,
        }

    # -- invocation --------------------------------------------------------
    async def run(
        self,
        /,
        request_factory: Callable[[BaseClient], Awaitable[Any]],
    ) -> Any:
        """Await a one-shot call, retrying once on the fallback."""
        client = self.active
        try:
            output = await request_factory(client)
            self._note_success(client)
            return output
        except (ServiceTimeout, ServiceError) as error:
            self._note_failure(client, error)
            alternate = self._other(client)
            if alternate is None:
                raise
            log.warning(
                "route.retry",
                role=self.role,
                from_side=client.side,
                to_side=alternate.side,
            )
            try:
                output = await request_factory(alternate)
            except (ServiceTimeout, ServiceError) as alternate_error:
                raise AllEndpointsFailed(
                    f"{self.role}: {error!r} then {alternate_error!r}"
                ) from alternate_error
            self.failover_count += 1
            return output

    async def stream(
        self,
        /,
        stream_factory: Callable[[BaseClient], AsyncIterator[Any]],
    ) -> AsyncIterator[Any]:
        """Iterate a stream, failing over only before the first chunk arrives."""
        client = self.active
        started = False
        try:
            async for item in stream_factory(client):
                started = True
                yield item
            self._note_success(client)
            return
        except (ServiceTimeout, ServiceError) as error:
            self._note_failure(client, error)
            alternate = self._other(client)
            if started or alternate is None:
                raise
            log.warning(
                "route.retry_stream",
                role=self.role,
                from_side=client.side,
                to_side=alternate.side,
            )
            async for item in stream_factory(alternate):
                yield item
            self.failover_count += 1

    def _other(
        self,
        /,
        used: BaseClient,
    ) -> BaseClient | None:
        if self.fallback is None or used is self.fallback:
            return None
        return self.fallback

    # -- health ------------------------------------------------------------
    def _note_success(
        self,
        /,
        client: BaseClient,
    ) -> None:
        if client is self.preferred:
            self._failure_count = 0

    def _note_failure(
        self,
        /,
        client: BaseClient,
        exc: Exception,
    ) -> None:
        if client is not self.preferred:
            return
        self._failure_count += 1
        log.warning(
            "route.failure",
            role=self.role,
            side=client.side,
            failure_count=self._failure_count,
            error=repr(exc),
        )
        if not self._degraded and self._failure_count >= self.config.failover_after:
            self._set_degraded(True, f"{self._failure_count} consecutive failures")

    def _set_degraded(
        self,
        /,
        value: bool,
        reason: str,
    ) -> None:
        if self._degraded == value:
            return
        self._degraded = value
        self._healthy_since = None
        log.warning("route.placement", role=self.role, side=self.side, reason=reason)
        if self._on_change:
            self._on_change(self.role, self.side, reason)

    async def _probe_loop(
        self,
        /,
    ) -> None:
        """Watch a degraded remote and restore it once it stays healthy."""
        while True:
            await asyncio.sleep(self.config.health_interval_s)
            if not self._degraded:
                self._healthy_since = None
                continue
            health = await self.preferred.health()
            if not health.get("ok"):
                self._healthy_since = None
                continue
            now = time.monotonic()
            if self._healthy_since is None:
                self._healthy_since = now
            elif now - self._healthy_since >= self.config.recover_after_s:
                self._failure_count = 0
                self._set_degraded(
                    False,
                    f"healthy for {self.config.recover_after_s:.0f}s",
                )

    def start_probe(
        self,
        /,
    ) -> None:
        if self.fallback is not None and self._probe_task is None:
            self._probe_task = asyncio.create_task(
                self._probe_loop(),
                name=f"probe-{self.role}",
            )

    async def aclose(
        self,
        /,
    ) -> None:
        if self._probe_task is not None:
            self._probe_task.cancel()
            try:
                await self._probe_task
            except asyncio.CancelledError:
                pass
            self._probe_task = None
        await self.preferred.aclose()
        if self.fallback is not None:
            await self.fallback.aclose()
