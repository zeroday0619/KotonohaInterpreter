"""Uniform invocation wrapper for one hardware-profile service endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, ClassVar

from kotonoha.clients._base import BaseClient


class ManagedServiceClient:
    """Expose one resident client through the orchestrator invocation contract."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "client",
        "role",
    )

    role: str
    client: BaseClient

    def __init__(
        self,
        role: str,
        client: BaseClient,
        /,
    ) -> None:
        self.role = role
        self.client = client

    @property
    def active(
        self,
        /,
    ) -> BaseClient:
        return self.client

    @property
    def degraded(
        self,
        /,
    ) -> bool:
        return False

    @property
    def side(
        self,
        /,
    ) -> str:
        return self.client.side

    def status(
        self,
        /,
    ) -> dict[str, Any]:
        return {
            "role": self.role,
            "endpoint": self.client.base_url,
            "degraded": False,
        }

    async def run(
        self,
        request_factory: Callable[[BaseClient], Awaitable[Any]],
        /,
    ) -> Any:
        return await request_factory(self.client)

    async def stream(
        self,
        stream_factory: Callable[[BaseClient], AsyncIterator[Any]],
        /,
    ) -> AsyncIterator[Any]:
        async for item in stream_factory(self.client):
            yield item

    def start_probe(
        self,
        /,
    ) -> None:
        return

    async def aclose(
        self,
        /,
    ) -> None:
        await self.client.aclose()
