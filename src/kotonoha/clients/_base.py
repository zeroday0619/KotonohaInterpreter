"""Shared plumbing for the service clients."""

from __future__ import annotations

import asyncio
import ssl
from typing import Any, ClassVar

import httpx

from kotonoha._config import RemoteConfig
from kotonoha._logging_setup import get_logger
from kotonoha._typing import override

log = get_logger(__name__)


class ServiceError(RuntimeError):
    __slots__: ClassVar[tuple[str, ...]] = ()
    pass


class ServiceTimeout(ServiceError):
    __slots__: ClassVar[tuple[str, ...]] = ()
    pass


def remote_transport_kwargs(
    config: RemoteConfig,
    /,
) -> dict[str, object]:
    """Build HTTP transport options for the external server."""
    headers = {"connection": "keep-alive"}
    if config.token:
        headers["authorization"] = f"Bearer {config.token}"

    verify: bool | ssl.SSLContext = config.verify_tls
    if config.ca_bundle is not None:
        verify = ssl.create_default_context(cafile=str(config.ca_bundle))

    return {
        "headers": headers,
        "verify": verify,
        "connect_timeout": config.connect_timeout_s,
    }


class BaseClient:
    __slots__: ClassVar[tuple[str, ...]] = (
        "_client",
        "_connect_timeout",
        "_headers",
        "_timeout",
        "_verify",
        "base_url",
        "name",
        "side",
    )
    base_url: str
    name: str
    side: str
    _timeout: float
    _connect_timeout: float
    _headers: dict[str, str]
    _verify: bool | ssl.SSLContext
    _client: httpx.AsyncClient

    @override
    def __init__(
        self,
        /,
        base_url: str,
        timeout: float,
        name: str,
        *,
        side: str = "local",
        headers: dict | None = None,
        verify: bool | ssl.SSLContext = True,
        connect_timeout: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.side = side
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._headers = {"connection": "keep-alive", **(headers or {})}
        self._verify = verify
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            headers=self._headers,
            verify=verify,
        )

    @property
    def label(
        self,
        /,
    ) -> str:
        return f"{self.name}@{self.side}"

    async def aclose(
        self,
        /,
    ) -> None:
        await self._client.aclose()

    async def health(
        self,
        /,
    ) -> dict:
        try:
            response = await self._client.get("/health", timeout=2.0)
            response.raise_for_status()
            return {**response.json(), "side": self.side}
        except Exception as error:  # noqa: BLE001
            return {
                "ok": False,
                "error": repr(error),
                "service": self.name,
                "side": self.side,
            }

    async def wait_ready(
        self,
        /,
        timeout: float = 300.0,
        interval: float = 2.0,
    ) -> bool:
        """Block until the service has finished loading its model.

        Models are resident (§3); nothing is loaded per turn.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            health = await self.health()
            if health.get("ok"):
                log.info("service.ready", service=self.label, detail=health)
                return True
            await asyncio.sleep(interval)
        log.error("service.not_ready", service=self.label, timeout=timeout)
        return False

    # -- request helpers ---------------------------------------------------
    async def _post_json(
        self,
        /,
        path: str,
        payload: dict,
        timeout: float | None = None,
    ) -> dict:
        return await self._request("POST", path, timeout, json=payload)

    async def _post_multipart(
        self,
        /,
        path: str,
        files: dict,
        data: dict | None = None,
        timeout: float | None = None,
    ) -> dict:
        return await self._request("POST", path, timeout, files=files, data=data)

    async def _request(
        self,
        /,
        method: str,
        path: str,
        timeout: float | None,
        **request_options: Any,
    ) -> dict:
        try:
            response = await self._client.request(
                method,
                path,
                timeout=timeout or self._timeout,
                **request_options,
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as error:
            raise ServiceTimeout(f"{self.label} timeout on {path}") from error
        except httpx.HTTPStatusError as error:
            detail = (
                f"{self.label} {error.response.status_code}: {error.response.text[:200]}"
            )
            raise ServiceError(detail) from error
        except httpx.HTTPError as error:
            raise ServiceError(f"{self.label} transport error: {error!r}") from error
