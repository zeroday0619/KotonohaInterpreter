"""Prometheus receiver for the resident RTX A6000 model services."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Final

from fastapi import FastAPI
from fastapi.responses import Response

from kotonoha._config import ROLES, load_settings
from kotonoha._logging_setup import setup_logging
from kotonoha._prometheus import (
    MetricsAggregator,
    create_unified_registry,
    metrics_response,
)
from kotonoha.services._auth import install_auth

log = setup_logging(service="metrics", console=True)
REFRESH_SECONDS: Final[float] = 10.0
SOURCE: Final[str] = "a6000"
SETTINGS: Final = load_settings()
PLACEMENT: Final[dict[str, str]] = dict.fromkeys(ROLES, SOURCE)
ENDPOINTS: Final[dict[str, str]] = {
    role: getattr(SETTINGS.services, role)
    for role in ROLES
}
TOKEN: Final[str] = os.environ.get("KOTONOHA_SERVICE_TOKEN", "").strip()
REQUEST_HEADERS: Final[dict[str, str]] = (
    {"authorization": f"Bearer {TOKEN}"} if TOKEN else {}
)
AGGREGATOR = MetricsAggregator(
    SETTINGS,
    PLACEMENT,
    endpoint_urls=ENDPOINTS,
    headers=REQUEST_HEADERS,
)
REGISTRY = create_unified_registry(AGGREGATOR)


async def _refresh_loop() -> None:
    while True:
        await asyncio.sleep(REFRESH_SECONDS)
        try:
            await AGGREGATOR.refresh(PLACEMENT)
        except Exception as error:
            log.error("metrics.refresh_failed", error=repr(error))


@asynccontextmanager
async def lifespan(
    application: FastAPI,
    /,
) -> AsyncIterator[None]:
    del application
    await AGGREGATOR.refresh(PLACEMENT)
    refresh_task = asyncio.create_task(_refresh_loop(), name="metrics-refresh")
    try:
        yield
    finally:
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="kotonoha-metrics", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "metrics",
        "backend": "prometheus_aggregator",
        "port": SETTINGS.logging.prometheus_port or 9091,
    }


install_auth(app, "metrics")


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return metrics_response(REGISTRY)


def main() -> None:
    import uvicorn

    port = SETTINGS.logging.prometheus_port
    if port is None:
        raise RuntimeError("logging.prometheus_port must be set for the metrics receiver")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        loop="uvloop",
    )


if __name__ == "__main__":
    main()
