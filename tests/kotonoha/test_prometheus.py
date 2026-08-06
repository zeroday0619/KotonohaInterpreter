"""Prometheus exposition and instrumentation contracts."""

from __future__ import annotations

from typing import ClassVar

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from kotonoha._config import load_settings
from kotonoha._metrics import TurnMetrics
from kotonoha._prometheus import (
    MetricsAggregator,
    create_unified_registry,
    install_metrics,
    metrics_response,
    observe_service_health,
    observe_turn,
)


class MetricsResponse:
    __slots__: ClassVar[tuple[str, ...]] = ("text",)

    text: str

    def __init__(
        self,
        text: str,
        /,
    ) -> None:
        self.text = text

    def raise_for_status(
        self,
        /,
    ) -> None:
        return None


class MetricsClient:
    __slots__: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        /,
        **_: object,
    ) -> None:
        pass

    async def __aenter__(
        self,
        /,
    ) -> MetricsClient:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
        /,
    ) -> None:
        return None

    async def get(
        self,
        url: str,
        /,
        **_: object,
    ) -> MetricsResponse:
        return MetricsResponse(
            "# HELP kotonoha_remote_test A remote test metric.\n"
            "# TYPE kotonoha_remote_test gauge\n"
            "kotonoha_remote_test 1\n"
        )


def test_metrics_endpoint_exports_http_samples() -> None:
    application = FastAPI()
    install_metrics(application, "test-http")

    @application.get("/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    with TestClient(application) as client:
        assert client.get("/ping").json() == {"ok": True}
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert (
        'kotonoha_http_requests_total{method="GET",path="/ping",service="test-http"'
        in response.text
    )


def test_service_health_exports_accelerator_and_memory_samples() -> None:
    observe_service_health(
        "test-resources",
        True,
        {
            "gpu_memory_utilization": 0.35,
            "max_num_seqs": 4,
            "prefix_caching": False,
            "allocated_mib": 128.0,
            "system": {
                "accelerator": {
                    "backend": "cpu",
                    "device_count": 0,
                    "memory_architecture": "unknown",
                },
                "memory": {"total_mib": 1024.0, "available_mib": 512.0},
            },
        },
    )
    output = metrics_response().body.decode("utf-8")

    assert 'kotonoha_service_up{service="test-resources"} 1.0' in output
    assert 'backend="cpu"' in output
    assert 'state="allocated"' in output
    assert 'state="total"' in output


def test_completed_turn_exports_latency_and_quality_samples() -> None:
    metrics = TurnMetrics()
    metrics.input_mode = "text"
    metrics.perf_mode = "onboard"
    metrics.output_tokens = 12
    metrics.tok_per_s = 24.0
    metrics.asr_avg_logprob = -0.4
    metrics.mark("eou")
    metrics.mark("asr_done")
    metrics.mark("first_clause")
    metrics.mark("first_audio")
    metrics.mark("queue_drained")

    observe_turn(metrics, load_settings().budget_ms)
    output = metrics_response().body.decode("utf-8")

    assert 'input_mode="text"' in output
    assert 'stage="total_to_first_audio"' in output
    assert "kotonoha_turn_output_tokens_count" in output
    assert "kotonoha_cross_verification_turns_total" in output


@pytest.mark.asyncio
async def test_remote_service_metrics_are_merged_into_one_registry(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    settings.perf_mode = "remote"
    settings.remote.enabled = True
    aggregator = MetricsAggregator(settings, settings.resolved_placement())
    monkeypatch.setattr(httpx, "AsyncClient", MetricsClient)

    await aggregator.refresh(settings.resolved_placement())

    output = generate_latest(create_unified_registry(aggregator)).decode("utf-8")
    assert 'kotonoha_remote_test{role="asr",source="remote"} 1.0' in output
    assert 'kotonoha_remote_test{role="tts",source="remote"} 1.0' in output

    local_placement = dict.fromkeys(settings.resolved_placement(), "local")
    await aggregator.refresh(local_placement)
    output = generate_latest(create_unified_registry(aggregator)).decode("utf-8")
    assert 'kotonoha_remote_test{role="asr",source="local"} 1.0' in output
    assert 'kotonoha_remote_test{role="asr",source="remote"}' not in output
