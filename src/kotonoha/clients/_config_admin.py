"""Authenticated client for the remote configuration management API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import httpx

from kotonoha._config import RemoteConfig
from kotonoha._typing import override
from kotonoha.clients._base import ServiceError, ServiceTimeout, remote_transport_kwargs


@dataclass(frozen=True, slots=True)
class RemoteConfigSnapshot:
    config: dict[str, Any]
    editable_paths: list[str]
    overrides: dict[str, Any]
    path: str
    restart_required: bool


class RemoteConfigClient:
    __slots__: ClassVar[tuple[str, ...]] = (
        "_client",
    )
    _client: httpx.AsyncClient

    @override
    def __init__(
        self,
        /,
        base_url: str,
        remote: RemoteConfig,
    ) -> None:
        transport = remote_transport_kwargs(remote)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=transport["headers"],
            verify=transport["verify"],
            timeout=httpx.Timeout(10.0, connect=transport["connect_timeout"]),
        )

    async def aclose(
        self,
        /,
    ) -> None:
        await self._client.aclose()

    async def read(
        self,
        /,
    ) -> RemoteConfigSnapshot:
        result = await self._request("GET", "/admin/config")
        return RemoteConfigSnapshot(**result)

    async def update(
        self,
        /,
        changes: dict[str, Any],
    ) -> RemoteConfigSnapshot:
        result = await self._request("PUT", "/admin/config", json={"changes": changes})
        return RemoteConfigSnapshot(**result)

    async def _request(
        self,
        /,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict:
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as error:
            raise ServiceTimeout(f"remote config timeout on {path}") from error
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:300]
            raise ServiceError(f"remote config {error.response.status_code}: {detail}") from error
        except httpx.HTTPError as error:
            raise ServiceError(f"remote config transport error: {error!r}") from error
