"""Authenticated client for the remote configuration management API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import httpx2

from kotonoha._config import RemoteConfig
from kotonoha._typing import override
from kotonoha.clients._base import (
    ServiceError,
    ServiceTimeout,
    read_json_object_response,
    remote_transport_kwargs,
    service_error_from_status,
)


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
    _client: httpx2.AsyncClient

    @override
    def __init__(
        self,
        /,
        base_url: str,
        remote: RemoteConfig,
    ) -> None:
        transport = remote_transport_kwargs(remote)
        self._client = httpx2.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=transport["headers"],
            verify=transport["verify"],
            timeout=httpx2.Timeout(10.0, connect=transport["connect_timeout"]),
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
            async with self._client.stream(method, path, **kwargs) as response:
                response.raise_for_status()
                return await read_json_object_response(response)
        except httpx2.TimeoutException as error:
            raise ServiceTimeout(f"remote config timeout on {path}") from error
        except httpx2.HTTPStatusError as error:
            raise service_error_from_status(error, "remote config") from error
        except httpx2.HTTPError as error:
            raise ServiceError(f"remote config transport error: {error!r}") from error
