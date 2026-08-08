"""Prometheus exposition and instrumentation contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import ClassVar

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY, generate_latest
from prometheus_client.core import Metric
from prometheus_client.parser import text_string_to_metric_families

from kotonoha._config import load_settings
from kotonoha._metrics import TurnMetrics
from kotonoha._prometheus import (
    MAXIMUM_METRICS_PAYLOAD_BYTES,
    MetricsAggregator,
    create_unified_registry,
    install_metrics,
    metrics_response,
    observe_service_health,
    observe_turn,
)
from kotonoha._typing import override
from kotonoha.web._monitoring import _monitoring_point


class MetricsResponse:
    __slots__: ClassVar[tuple[str, ...]] = ("_payload", "text")

    text: str
    _payload: bytes

    def __init__(
        self,
        text: str,
        /,
    ) -> None:
        self.text = text
        self._payload = text.encode("utf-8")

    def raise_for_status(
        self,
        /,
    ) -> None:
        return None

    async def aiter_bytes(
        self,
        /,
    ) -> AsyncIterator[bytes]:
        yield self._payload


class MetricsClient:
    __slots__: ClassVar[tuple[str, ...]] = ()
    requested_urls: ClassVar[list[str]] = []

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

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        /,
        **_: object,
    ) -> AsyncIterator[MetricsResponse]:
        del method
        self.requested_urls.append(url)
        yield MetricsResponse(
            "# HELP kotonoha_remote_test A remote test metric.\n"
            "# TYPE kotonoha_remote_test gauge\n"
            "kotonoha_remote_test 1\n"
        )

    async def aclose(
        self,
        /,
    ) -> None:
        return None


class OversizedMetricsClient(MetricsClient):
    __slots__: ClassVar[tuple[str, ...]] = ()

    @override
    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        /,
        **_: object,
    ) -> AsyncIterator[MetricsResponse]:
        del method, url
        yield MetricsResponse("x" * (MAXIMUM_METRICS_PAYLOAD_BYTES + 1))


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


def test_unmatched_paths_share_one_bounded_metric_label() -> None:
    application = FastAPI()
    install_metrics(application, "test-unmatched")

    with TestClient(application) as client:
        for sequence in range(20):
            assert client.get(f"/random-{sequence}").status_code == 404
        response = client.get("/metrics")

    service_lines = [
        line
        for line in response.text.splitlines()
        if 'service="test-unmatched"' in line and "http_requests_total" in line
    ]
    assert len(service_lines) == 1
    assert 'path="unmatched"' in service_lines[0]
    assert service_lines[0].endswith("20.0")


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
                "os": {"name": "Test OS"},
                "kernel": {"release": "test-kernel", "machine": "test-machine"},
                "cpu": {"load_1m_ratio": 0.25},
                "disk": {"total_mib": 4096.0, "used_mib": 1024.0, "free_mib": 3072.0},
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
    assert 'kernel="test-kernel"' in output
    assert 'kotonoha_service_system_cpu_load_ratio{service="test-resources"} 0.25' in output
    assert 'kotonoha_service_disk_bytes{service="test-resources",state="used"}' in output
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
    monkeypatch.setattr(httpx2, "AsyncClient", MetricsClient)

    await aggregator.refresh(settings.resolved_placement())

    output = generate_latest(create_unified_registry(aggregator)).decode("utf-8")
    assert 'kotonoha_remote_test{role="asr",source="remote"} 1.0' in output
    assert 'kotonoha_remote_test{role="tts",source="remote"} 1.0' in output

    local_placement = dict.fromkeys(settings.resolved_placement(), "local")
    await aggregator.refresh(local_placement)
    output = generate_latest(create_unified_registry(aggregator)).decode("utf-8")
    assert 'kotonoha_remote_test{role="asr",source="local"} 1.0' in output
    assert 'kotonoha_remote_test{role="asr",source="remote"}' not in output


def test_monitoring_targets_include_onboard_fallbacks_for_remote_roles() -> None:
    settings = load_settings()
    settings.perf_mode = "remote"
    settings.remote.enabled = True
    placement = settings.resolved_placement()
    aggregator = MetricsAggregator(settings, placement, include_fallbacks=True)

    targets = aggregator.targets(placement)

    assert len(targets) == 8
    for role in placement:
        assert (role, "remote") in {(target_role, source) for target_role, source, _ in targets}
        assert (role, "local") in {(target_role, source) for target_role, source, _ in targets}


def test_monitoring_targets_remain_bounded_for_onboard_mode() -> None:
    settings = load_settings()
    settings.perf_mode = "onboard"
    placement = settings.resolved_placement()
    aggregator = MetricsAggregator(settings, placement, include_fallbacks=True)

    targets = aggregator.targets(placement)

    assert len(targets) == 4
    assert {source for _role, source, _url in targets} == {"local"}


def test_monitoring_point_combines_host_service_and_turn_metrics() -> None:
    scrape = Metric("kotonoha_remote_metrics_scrape_up", "scrape", "gauge")
    scrape.add_sample(
        "kotonoha_remote_metrics_scrape_up",
        {"service": "asr", "source": "local"},
        1,
    )
    service = Metric("kotonoha_service_up", "service", "gauge")
    service.add_sample(
        "kotonoha_service_up",
        {"service": "asr", "role": "asr", "source": "local"},
        1,
    )
    accelerator = Metric("kotonoha_service_accelerator_info", "accelerator", "gauge")
    accelerator.add_sample(
        "kotonoha_service_accelerator_info",
        {
            "service": "asr",
            "role": "asr",
            "source": "local",
            "backend": "cuda",
            "memory_architecture": "unified",
        },
        1,
    )
    system = Metric("kotonoha_service_system_info", "system", "gauge")
    system.add_sample(
        "kotonoha_service_system_info",
        {
            "service": "asr",
            "role": "asr",
            "source": "local",
            "os": "Linux",
            "kernel": "6.8.12-tegra",
            "machine": "aarch64",
        },
        1,
    )
    cpu = Metric("kotonoha_service_system_cpu_load_ratio", "cpu", "gauge")
    cpu.add_sample(
        "kotonoha_service_system_cpu_load_ratio",
        {"service": "asr", "role": "asr", "source": "local"},
        0.5,
    )
    memory = Metric("kotonoha_service_memory_bytes", "memory", "gauge")
    for state, value in (("total", 64 * 1024**3), ("available", 16 * 1024**3)):
        memory.add_sample(
            "kotonoha_service_memory_bytes",
            {
                "service": "asr",
                "role": "asr",
                "source": "local",
                "scope": "system",
                "state": state,
            },
            value,
        )
    for state, value in (("total", 64 * 1024**3), ("reserved", 8 * 1024**3)):
        memory.add_sample(
            "kotonoha_service_memory_bytes",
            {
                "service": "asr",
                "role": "asr",
                "source": "local",
                "scope": "accelerator",
                "state": state,
            },
            value,
        )
    turns = Metric("kotonoha_turns", "turns", "counter")
    turns.add_sample(
        "kotonoha_turns_total",
        {"outcome": "ok", "input_mode": "voice", "perf_mode": "onboard"},
        7,
    )

    point = _monitoring_point(
        (scrape, service, accelerator, system, cpu, memory, turns),
        (("asr", "local"),),
        load_settings(),
        100.0,
    )

    assert point["summary"]["services_ready"] == 1
    assert point["summary"]["turns_total"] == 7
    assert point["services"][0]["accelerator_backend"] == "cuda"
    assert point["services"][0]["kernel"] == "6.8.12-tegra"
    assert point["services"][0]["cpu_load_ratio"] == 0.5
    assert point["services"][0]["memory_percent"] == 75.0


@pytest.mark.asyncio
async def test_remote_metrics_payload_is_bounded() -> None:
    settings = load_settings()
    aggregator = MetricsAggregator(settings, settings.resolved_placement())

    role, source, payload = await aggregator._fetch(
        OversizedMetricsClient(),
        "asr",
        "remote",
        "http://remote.test:8001",
        {},
    )

    assert (role, source, payload) == ("asr", "remote", None)


@pytest.mark.asyncio
async def test_monitoring_refreshes_service_health_before_scraping_metrics() -> None:
    settings = load_settings()
    aggregator = MetricsAggregator(settings, settings.resolved_placement())
    client = MetricsClient()
    MetricsClient.requested_urls.clear()

    role, source, payload = await aggregator._fetch(
        client,
        "asr",
        "local",
        "http://service.test:8001",
        {},
        probe_health=True,
    )

    assert (role, source) == ("asr", "local")
    assert payload is not None
    assert MetricsClient.requested_urls == [
        "http://service.test:8001/health",
        "http://service.test:8001/metrics",
    ]


def test_unified_registry_emits_metric_metadata_once() -> None:
    settings = load_settings()
    aggregator = MetricsAggregator(settings, settings.resolved_placement())
    payload = generate_latest(REGISTRY).decode("utf-8")
    aggregator._payloads[("asr", "local")] = tuple(
        family
        for family in text_string_to_metric_families(payload)
        if family.name.startswith("kotonoha_")
    )

    output = generate_latest(create_unified_registry(aggregator)).decode("utf-8")

    assert output.count("# HELP kotonoha_http_requests_total") == 1
    assert output.count("# TYPE kotonoha_http_requests_total") == 1
