"""Shared plumbing for the service clients."""

from __future__ import annotations

import asyncio
import json
import ssl
from typing import Any, ClassVar, Final

import httpx2

from kotonoha._config import RemoteConfig
from kotonoha._logging_setup import get_logger
from kotonoha._typing import override

log = get_logger(__name__)
MAXIMUM_SERVICE_RESPONSE_BYTES: Final[int] = 2 * 1024 * 1024


class ServiceError(RuntimeError):
    __slots__: ClassVar[tuple[str, ...]] = ()
    pass


class ServiceApplicationError(ServiceError):
    """A request defect that must not be retried against another endpoint."""
    __slots__: ClassVar[tuple[str, ...]] = ()
    pass


class ServiceTimeout(ServiceError):
    __slots__: ClassVar[tuple[str, ...]] = ()
    pass


def service_error_from_status(
    error: httpx2.HTTPStatusError,
    /,
    label: str,
) -> ServiceError:
    """Classify HTTP failures so failover only handles retryable conditions."""
    status_code = error.response.status_code
    # Validation responses can echo transcript or translation input. Keep
    # operator logs useful without copying user content from the response body.
    detail = f"{label} request failed with HTTP {status_code}"
    if 400 <= status_code < 500 and status_code not in {408, 425, 429}:
        return ServiceApplicationError(detail)
    return ServiceError(detail)


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


async def read_json_object_response(
    response: httpx2.Response,
    /,
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Read a bounded service response and require a JSON object."""
    payload = bytearray()
    async for chunk in response.aiter_bytes():
        if len(payload) + len(chunk) > MAXIMUM_SERVICE_RESPONSE_BYTES:
            raise ServiceError(
                f"service response exceeded {MAXIMUM_SERVICE_RESPONSE_BYTES} bytes"
            )
        payload.extend(chunk)
    if not payload and allow_empty:
        return {}
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ServiceError("service returned invalid JSON") from error
    if not isinstance(decoded, dict):
        raise ServiceError("service returned a non-object JSON response")
    return decoded


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
    _client: httpx2.AsyncClient

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
        self._client = httpx2.AsyncClient(
            base_url=self.base_url,
            timeout=httpx2.Timeout(timeout, connect=connect_timeout),
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
            async with self._client.stream("GET", "/health", timeout=2.0) as response:
                response.raise_for_status()
                payload = await read_json_object_response(response)
            return {**payload, "side": self.side}
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
            async with self._client.stream(
                method,
                path,
                timeout=timeout or self._timeout,
                **request_options,
            ) as response:
                response.raise_for_status()
                return await read_json_object_response(response)
        except httpx2.TimeoutException as error:
            raise ServiceTimeout(f"{self.label} timeout on {path}") from error
        except httpx2.HTTPStatusError as error:
            raise service_error_from_status(error, self.label) from error
        except httpx2.HTTPError as error:
            raise ServiceError(f"{self.label} transport error: {error!r}") from error
