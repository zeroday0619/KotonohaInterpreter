"""서비스 클라이언트 공통."""

from __future__ import annotations

import asyncio

import httpx

from ..logging_setup import get_logger

log = get_logger(__name__)


class ServiceError(RuntimeError):
    pass


class ServiceTimeout(ServiceError):
    pass


class BaseClient:
    def __init__(self, base_url: str, timeout: float, name: str):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=2.0),
            headers={"connection": "keep-alive"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        try:
            r = await self._client.get("/health", timeout=2.0)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": repr(e), "service": self.name}

    async def wait_ready(self, timeout: float = 300.0, interval: float = 2.0) -> bool:
        """서비스가 모델을 다 올릴 때까지 기다린다(§3: 매 턴 로드 금지)."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            h = await self.health()
            if h.get("ok"):
                log.info("service.ready", service=self.name, detail=h)
                return True
            await asyncio.sleep(interval)
        log.error("service.not_ready", service=self.name, timeout=timeout)
        return False

    async def _post_json(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        try:
            r = await self._client.post(path, json=payload, timeout=timeout or self._timeout)
            r.raise_for_status()
            return r.json()
        except httpx.TimeoutException as e:
            raise ServiceTimeout(f"{self.name} timeout on {path}") from e
        except httpx.HTTPStatusError as e:
            detail = f"{self.name} {e.response.status_code}: {e.response.text[:200]}"
            raise ServiceError(detail) from e
        except httpx.HTTPError as e:
            raise ServiceError(f"{self.name} transport error: {e!r}") from e
