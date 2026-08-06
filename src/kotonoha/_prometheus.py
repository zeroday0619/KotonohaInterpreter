"""Prometheus metrics adapter for services and completed interpretation turns."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any, ClassVar

import httpx2
from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)
from prometheus_client.core import Metric
from prometheus_client.parser import text_string_to_metric_families

from kotonoha._config import ROLES, LatencyBudgetConfig, Settings
from kotonoha._metrics import TurnMetrics
from kotonoha.clients._base import remote_transport_kwargs

HTTP_REQUESTS = Counter(
    "kotonoha_http_requests",
    "HTTP requests handled by a Kotonoha service.",
    ("service", "method", "path", "status"),
)
HTTP_DURATION = Histogram(
    "kotonoha_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("service", "method", "path", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
SERVICE_UP = Gauge(
    "kotonoha_service_up",
    "Whether the resident model service is ready.",
    ("service",),
)
SERVICE_ACCELERATOR_INFO = Gauge(
    "kotonoha_service_accelerator_info",
    "Accelerator metadata for a resident model service.",
    ("service", "backend", "memory_architecture"),
)
SERVICE_ACCELERATOR_DEVICES = Gauge(
    "kotonoha_service_accelerator_devices",
    "Number of detected accelerator devices.",
    ("service",),
)
SERVICE_MEMORY = Gauge(
    "kotonoha_service_memory_bytes",
    "Resident service memory counters in bytes.",
    ("service", "scope", "state"),
)
SERVICE_GPU_MEMORY_UTILIZATION = Gauge(
    "kotonoha_service_configured_gpu_memory_utilization_ratio",
    "Configured fraction of accelerator memory available to the engine.",
    ("service",),
)
SERVICE_MAX_NUM_SEQS = Gauge(
    "kotonoha_service_configured_max_num_sequences",
    "Configured maximum number of concurrent sequences.",
    ("service",),
)
SERVICE_PREFIX_CACHING = Gauge(
    "kotonoha_service_configured_prefix_caching",
    "Whether automatic prefix caching is enabled.",
    ("service",),
)
TURNS = Counter(
    "kotonoha_turns",
    "Completed interpretation turns.",
    ("outcome", "input_mode", "perf_mode"),
)
TURN_STAGE_DURATION = Histogram(
    "kotonoha_turn_stage_duration_seconds",
    "Interpretation stage duration in seconds from the turn record.",
    ("stage", "outcome"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
TURN_OUTPUT_TOKENS = Histogram(
    "kotonoha_turn_output_tokens",
    "Generated translation tokens per completed turn.",
    buckets=(1, 5, 10, 20, 40, 80, 160, 320, 640),
)
TURN_TOKENS_PER_SECOND = Histogram(
    "kotonoha_turn_tokens_per_second",
    "Translation generation rate per completed turn.",
    buckets=(1, 2, 5, 10, 20, 40, 80, 160),
)
TURN_ASR_LOGPROB = Histogram(
    "kotonoha_turn_asr_average_logprob",
    "Average primary ASR log probability per completed turn.",
    buckets=(-10, -5, -2, -1, -0.5, -0.25, 0),
)
CROSS_VERIFICATION = Counter(
    "kotonoha_cross_verification_turns",
    "Turns classified by conditional ASR cross-verification state.",
    ("fired", "divergent"),
)
FAILOVERS = Counter(
    "kotonoha_failovers",
    "Role failover events recorded on completed turns.",
)
OVER_BUDGET_TURNS = Counter(
    "kotonoha_over_budget_turns",
    "Completed turns that exceeded at least one latency budget.",
)
REMOTE_SCRAPE_UP = Gauge(
    "kotonoha_remote_metrics_scrape_up",
    "Whether the metrics receiver received the latest service metrics payload.",
    ("service", "source"),
)


class MetricsAggregator:
    """Collect active service metrics for a unified exporter."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "_endpoint_urls",
        "_headers",
        "_payloads",
        "placement",
        "settings",
    )

    settings: Settings
    placement: dict[str, str]
    _endpoint_urls: dict[str, str] | None
    _headers: dict[str, str] | None
    _payloads: dict[tuple[str, str], tuple[Metric, ...]]

    def __init__(
        self,
        settings: Settings,
        /,
        placement: Mapping[str, str],
        *,
        endpoint_urls: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.settings = settings
        self.placement = dict(placement)
        self._endpoint_urls = dict(endpoint_urls) if endpoint_urls is not None else None
        self._headers = dict(headers) if headers is not None else None
        self._payloads = {}

    async def refresh(
        self,
        placement: Mapping[str, str],
        /,
    ) -> None:
        """Fetch metrics from every currently active role endpoint."""
        self.placement = dict(placement)
        transport = remote_transport_kwargs(self.settings.remote)
        timeout = httpx2.Timeout(
            3.0,
            connect=float(transport["connect_timeout"]),
        )
        endpoints = tuple(
            (
                role,
                self.placement[role],
                (
                    self._endpoint_urls[role]
                    if self._endpoint_urls is not None
                    else self.settings.url_for(role, self.placement[role])
                ),
            )
            for role in ROLES
        )
        active_keys = {(role, source) for role, source, _url in endpoints}
        for key in tuple(self._payloads):
            if key not in active_keys:
                self._payloads.pop(key, None)
                role, source = key
                REMOTE_SCRAPE_UP.labels(service=role, source=source).set(0)
        async with httpx2.AsyncClient(
            verify=transport["verify"],
            timeout=timeout,
        ) as client:
            results = await asyncio.gather(
                *(
                    self._fetch(
                        client,
                        role,
                        source,
                        url,
                        (
                            self._headers
                            if self._headers is not None
                            else transport["headers"] if source == "remote" else {}
                        ),
                    )
                    for role, source, url in endpoints
                )
            )

        for role, source, payload in results:
            key = (role, source)
            if payload is None:
                self._payloads.pop(key, None)
                REMOTE_SCRAPE_UP.labels(service=role, source=source).set(0)
                continue
            try:
                families = tuple(
                    family
                    for family in text_string_to_metric_families(payload)
                    if family.name.startswith("kotonoha_")
                )
            except (TypeError, ValueError):
                self._payloads.pop(key, None)
                REMOTE_SCRAPE_UP.labels(service=role, source=source).set(0)
                continue
            self._payloads[key] = families
            REMOTE_SCRAPE_UP.labels(service=role, source=source).set(1)

    async def _fetch(
        self,
        client: httpx2.AsyncClient,
        role: str,
        source: str,
        url: str,
        headers: object,
        /,
    ) -> tuple[str, str, str | None]:
        try:
            response = await client.get(
                f"{url.rstrip('/')}/metrics",
                headers=headers,
            )
            response.raise_for_status()
        except httpx2.HTTPError:
            return role, source, None
        return role, source, response.text

    def collect(
        self,
        /,
    ) -> Iterable[Metric]:
        """Merge cached service families and add bounded placement labels."""
        merged: dict[str, Metric] = {}
        for (role, source), families in self._payloads.items():
            for family in families:
                metric = merged.get(family.name)
                if metric is None:
                    metric = Metric(
                        family.name,
                        family.documentation,
                        family.type,
                        getattr(family, "unit", ""),
                    )
                    merged[family.name] = metric
                for sample in family.samples:
                    labels = dict(sample.labels)
                    labels["role"] = role
                    labels["source"] = source
                    metric.add_sample(
                        sample.name,
                        labels,
                        sample.value,
                        sample.timestamp,
                        sample.exemplar,
                        sample.native_histogram,
                    )
        yield from merged.values()


class UnifiedCollector:
    """Expose the local registry and aggregated service families together."""

    __slots__: ClassVar[tuple[str, ...]] = ("aggregator",)

    def __init__(
        self,
        aggregator: MetricsAggregator,
        /,
    ) -> None:
        self.aggregator = aggregator

    def collect(
        self,
        /,
    ) -> Iterable[Metric]:
        yield from REGISTRY.collect()
        yield from self.aggregator.collect()

    def describe(
        self,
        /,
    ) -> tuple[Metric, ...]:
        return ()


def create_unified_registry(
    aggregator: MetricsAggregator,
    /,
) -> CollectorRegistry:
    """Create a registry that combines receiver-local and service metrics."""
    registry = CollectorRegistry(auto_describe=True)
    registry.register(UnifiedCollector(aggregator))
    return registry


def install_metrics(
    app: FastAPI,
    service: str,
    /,
    *,
    registry: CollectorRegistry | None = None,
) -> None:
    """Install HTTP instrumentation and a protected `/metrics` endpoint."""

    @app.middleware("http")
    async def _observe_request(
        request: Request,
        /,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start_time = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            labels = {
                "service": service,
                "method": request.method,
                "path": request.url.path,
                "status": status,
            }
            HTTP_REQUESTS.labels(**labels).inc()
            HTTP_DURATION.labels(**labels).observe(time.perf_counter() - start_time)

    @app.get("/metrics", include_in_schema=False)
    def _metrics_endpoint() -> Response:
        return metrics_response(registry)


def metrics_response(
    registry: CollectorRegistry | None = None,
    /,
) -> Response:
    """Return the current process metrics in Prometheus exposition format."""
    return Response(
        content=generate_latest(registry if registry is not None else REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


def start_metrics_server(
    port: int,
    /,
    *,
    registry: CollectorRegistry | None = None,
) -> tuple[Any, Any]:
    """Start the optional localhost exporter for the selected registry."""
    return start_http_server(
        port,
        addr="127.0.0.1",
        registry=registry if registry is not None else REGISTRY,
    )


def stop_metrics_server(
    server: tuple[Any, Any],
    /,
) -> None:
    """Stop an exporter created by :func:`start_metrics_server`."""
    http_server, _thread = server
    http_server.shutdown()
    http_server.server_close()


def observe_service_health(
    service: str,
    ok: bool,
    resources: dict[str, Any],
    /,
) -> None:
    """Update service readiness, accelerator metadata, and memory gauges."""
    SERVICE_UP.labels(service=service).set(1 if ok else 0)
    accelerator = resources.get("system", {}).get("accelerator", {})
    backend = str(accelerator.get("backend", "unknown"))
    memory_architecture = str(accelerator.get("memory_architecture", "unknown"))
    SERVICE_ACCELERATOR_INFO.labels(
        service=service,
        backend=backend,
        memory_architecture=memory_architecture,
    ).set(1)
    SERVICE_ACCELERATOR_DEVICES.labels(service=service).set(
        float(accelerator.get("device_count", 0))
    )

    memory = resources.get("system", {}).get("memory", {})
    _set_memory_gauge(service, "system", "total", memory.get("total_mib"))
    _set_memory_gauge(service, "system", "available", memory.get("available_mib"))
    accelerator_memory = {
        "allocated": resources.get("allocated_mib"),
        "reserved": resources.get("reserved_mib"),
        "max_reserved": resources.get("max_reserved_mib"),
        "free": resources.get("free_mib"),
        "total": resources.get("total_mib"),
    }
    for state, value in accelerator_memory.items():
        _set_memory_gauge(service, "accelerator", state, value)

    utilization = resources.get("gpu_memory_utilization")
    if utilization is not None:
        SERVICE_GPU_MEMORY_UTILIZATION.labels(service=service).set(float(utilization))
    max_num_sequences = resources.get("max_num_seqs")
    if max_num_sequences is not None:
        SERVICE_MAX_NUM_SEQS.labels(service=service).set(float(max_num_sequences))
    prefix_caching = resources.get("prefix_caching")
    if prefix_caching is not None:
        SERVICE_PREFIX_CACHING.labels(service=service).set(1 if prefix_caching else 0)


def observe_turn(
    metrics: TurnMetrics,
    budget: LatencyBudgetConfig,
    /,
) -> None:
    """Export one completed turn without exporting user text or identifiers."""
    record = metrics.to_dict(budget)
    TURNS.labels(
        outcome=metrics.outcome,
        input_mode=metrics.input_mode,
        perf_mode=metrics.perf_mode or "unknown",
    ).inc()
    for stage, duration_ms in record["stages_ms"].items():
        if duration_ms is not None:
            TURN_STAGE_DURATION.labels(stage=stage, outcome=metrics.outcome).observe(
                float(duration_ms) / 1000
            )
    if metrics.output_tokens is not None:
        TURN_OUTPUT_TOKENS.observe(metrics.output_tokens)
    if metrics.tok_per_s is not None:
        TURN_TOKENS_PER_SECOND.observe(metrics.tok_per_s)
    if metrics.asr_avg_logprob is not None:
        TURN_ASR_LOGPROB.observe(metrics.asr_avg_logprob)
    CROSS_VERIFICATION.labels(
        fired=str(metrics.cross_verify_fired).lower(),
        divergent=(
            str(metrics.cross_verify_divergent).lower()
            if metrics.cross_verify_divergent is not None
            else "unknown"
        ),
    ).inc()
    if metrics.failovers:
        FAILOVERS.inc(metrics.failovers)
    if record.get("over_budget_ms"):
        OVER_BUDGET_TURNS.inc()


def _set_memory_gauge(
    service: str,
    scope: str,
    state: str,
    mib: Any,
    /,
) -> None:
    if mib is not None:
        SERVICE_MEMORY.labels(service=service, scope=scope, state=state).set(
            float(mib) * 1024**2
        )
