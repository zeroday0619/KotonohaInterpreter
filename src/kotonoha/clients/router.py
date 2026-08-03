"""Routing a role to the A6000 or to the on-board service, with failover.

§10 says an interpreter that stops is worse than one that is wrong. Adding a
network hop adds a whole new way to stop, so every remote role keeps its
on-board counterpart loaded and falls back to it.

The policy:

  · A transport failure (timeout, connection refused, 5xx) counts. Application
    errors do not — a 400 means we sent something wrong and retrying elsewhere
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
from typing import Any

from ..config import RemoteCfg
from ..logging_setup import get_logger
from .base import BaseClient, ServiceError, ServiceTimeout

log = get_logger(__name__)


class AllEndpointsFailed(ServiceError):
    pass


class FailoverClient:
    """One role: a preferred client and, when the preferred is remote, a local fallback."""

    def __init__(
        self,
        role: str,
        preferred: BaseClient,
        fallback: BaseClient | None,
        cfg: RemoteCfg,
        on_change: Callable[[str, str, str], None] | None = None,
    ):
        self.role = role
        self.preferred = preferred
        self.fallback = fallback
        self.cfg = cfg
        self._on_change = on_change

        self._fails = 0
        self._degraded = False
        self._healthy_since: float | None = None
        self._probe: asyncio.Task | None = None
        self.failover_count = 0

    # -- state -------------------------------------------------------------
    @property
    def active(self) -> BaseClient:
        if self._degraded and self.fallback is not None:
            return self.fallback
        return self.preferred

    @property
    def side(self) -> str:
        return self.active.side

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def name(self) -> str:
        return self.preferred.name

    def status(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "side": self.side,
            "preferred": self.preferred.side,
            "degraded": self._degraded,
            "failovers": self.failover_count,
        }

    # -- invocation --------------------------------------------------------
    async def run(self, make_coro: Callable[[BaseClient], Awaitable[Any]]) -> Any:
        """Await a one-shot call, retrying once on the fallback."""
        client = self.active
        try:
            out = await make_coro(client)
            self._note_success(client)
            return out
        except (ServiceTimeout, ServiceError) as e:
            self._note_failure(client, e)
            other = self._other(client)
            if other is None:
                raise
            log.warning("route.retry", role=self.role, frm=client.side, to=other.side)
            try:
                out = await make_coro(other)
            except (ServiceTimeout, ServiceError) as e2:
                raise AllEndpointsFailed(f"{self.role}: {e!r} then {e2!r}") from e2
            self.failover_count += 1
            return out

    async def stream(
        self, make_agen: Callable[[BaseClient], AsyncIterator[Any]]
    ) -> AsyncIterator[Any]:
        """Iterate a stream, failing over only before the first chunk arrives."""
        client = self.active
        started = False
        try:
            async for item in make_agen(client):
                started = True
                yield item
            self._note_success(client)
            return
        except (ServiceTimeout, ServiceError) as e:
            self._note_failure(client, e)
            other = self._other(client)
            if started or other is None:
                raise
            log.warning("route.retry_stream", role=self.role, frm=client.side, to=other.side)
            async for item in make_agen(other):
                yield item
            self.failover_count += 1

    def _other(self, used: BaseClient) -> BaseClient | None:
        if self.fallback is None or used is self.fallback:
            return None
        return self.fallback

    # -- health ------------------------------------------------------------
    def _note_success(self, client: BaseClient) -> None:
        if client is self.preferred:
            self._fails = 0

    def _note_failure(self, client: BaseClient, exc: Exception) -> None:
        if client is not self.preferred:
            return
        self._fails += 1
        log.warning(
            "route.failure", role=self.role, side=client.side, n=self._fails, error=repr(exc)
        )
        if not self._degraded and self._fails >= self.cfg.failover_after:
            self._set_degraded(True, f"{self._fails} consecutive failures")

    def _set_degraded(self, value: bool, reason: str) -> None:
        if self._degraded == value:
            return
        self._degraded = value
        self._healthy_since = None
        log.warning("route.placement", role=self.role, side=self.side, reason=reason)
        if self._on_change:
            self._on_change(self.role, self.side, reason)

    async def _probe_loop(self) -> None:
        """Watch a degraded remote and restore it once it stays healthy."""
        while True:
            await asyncio.sleep(self.cfg.health_interval_s)
            if not self._degraded:
                self._healthy_since = None
                continue
            h = await self.preferred.health()
            if not h.get("ok"):
                self._healthy_since = None
                continue
            now = time.monotonic()
            if self._healthy_since is None:
                self._healthy_since = now
            elif now - self._healthy_since >= self.cfg.recover_after_s:
                self._fails = 0
                self._set_degraded(False, f"healthy for {self.cfg.recover_after_s:.0f}s")

    def start_probe(self) -> None:
        if self.fallback is not None and self._probe is None:
            self._probe = asyncio.create_task(self._probe_loop(), name=f"probe-{self.role}")

    async def aclose(self) -> None:
        if self._probe is not None:
            self._probe.cancel()
            try:
                await self._probe
            except asyncio.CancelledError:
                pass
            self._probe = None
        await self.preferred.aclose()
        if self.fallback is not None:
            await self.fallback.aclose()
