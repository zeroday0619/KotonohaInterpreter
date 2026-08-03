"""Shared plumbing for the service clients."""

from __future__ import annotations

import asyncio
import ssl

import httpx

from ..config import RemoteCfg
from ..logging_setup import get_logger

log = get_logger(__name__)


class ServiceError(RuntimeError):
    pass


class ServiceTimeout(ServiceError):
    pass


def remote_transport_kwargs(cfg: RemoteCfg) -> dict:
    """httpx settings for talking to the external box."""
    headers = {"connection": "keep-alive"}
    if cfg.token:
        headers["authorization"] = f"Bearer {cfg.token}"

    verify: bool | ssl.SSLContext = cfg.verify_tls
    if cfg.ca_bundle is not None:
        verify = ssl.create_default_context(cafile=str(cfg.ca_bundle))

    return {"headers": headers, "verify": verify, "connect_timeout": cfg.connect_timeout_s}


class BaseClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        name: str,
        *,
        side: str = "local",
        headers: dict | None = None,
        verify: bool | ssl.SSLContext = True,
        connect_timeout: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.side = side  # local | remote, for logs and turn metrics
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            headers={"connection": "keep-alive", **(headers or {})},
            verify=verify,
        )

    @property
    def label(self) -> str:
        return f"{self.name}@{self.side}"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        try:
            r = await self._client.get("/health", timeout=2.0)
            r.raise_for_status()
            return {**r.json(), "side": self.side}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": repr(e), "service": self.name, "side": self.side}

    async def wait_ready(self, timeout: float = 300.0, interval: float = 2.0) -> bool:
        """Block until the service has finished loading its model.

        Models are resident (§3); nothing is loaded per turn.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            h = await self.health()
            if h.get("ok"):
                log.info("service.ready", service=self.label, detail=h)
                return True
            await asyncio.sleep(interval)
        log.error("service.not_ready", service=self.label, timeout=timeout)
        return False

    # -- request helpers ---------------------------------------------------
    async def _post_json(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        return await self._request("POST", path, timeout, json=payload)

    async def _post_multipart(
        self, path: str, files: dict, data: dict | None = None, timeout: float | None = None
    ) -> dict:
        return await self._request("POST", path, timeout, files=files, data=data)

    async def _request(self, method: str, path: str, timeout: float | None, **kw) -> dict:
        try:
            r = await self._client.request(method, path, timeout=timeout or self._timeout, **kw)
            r.raise_for_status()
            return r.json()
        except httpx.TimeoutException as e:
            raise ServiceTimeout(f"{self.label} timeout on {path}") from e
        except httpx.HTTPStatusError as e:
            detail = f"{self.label} {e.response.status_code}: {e.response.text[:200]}"
            raise ServiceError(detail) from e
        except httpx.HTTPError as e:
            raise ServiceError(f"{self.label} transport error: {e!r}") from e
