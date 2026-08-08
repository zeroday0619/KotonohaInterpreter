"""Always-on unified metrics collection for the Web control center."""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Final

from prometheus_client import CollectorRegistry
from prometheus_client.core import Metric, Sample

from kotonoha._config import ROLES, Settings
from kotonoha._logging_setup import get_logger
from kotonoha._prometheus import MetricsAggregator, create_unified_registry

log = get_logger(__name__)

MONITORING_REFRESH_SECONDS: Final[float] = 5.0
HEALTH_REFRESH_SECONDS: Final[float] = 10.0
MONITORING_HISTORY_SAMPLES: Final[int] = 720
DEFAULT_MONITORING_WINDOW_SECONDS: Final[int] = 15 * 60


class MonitoringService:
    """Collect service metrics continuously and retain a bounded chart history."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "_aggregator",
        "_history",
        "_last_health_probe",
        "_last_error",
        "_lock",
        "_task",
        "registry",
        "settings",
    )

    settings: Settings
    registry: CollectorRegistry
    _aggregator: MetricsAggregator
    _history: deque[dict[str, Any]]
    _last_health_probe: float
    _last_error: str | None
    _lock: asyncio.Lock
    _task: asyncio.Task[None] | None

    def __init__(
        self,
        settings: Settings,
        /,
    ) -> None:
        self.settings = settings
        self._aggregator = MetricsAggregator(
            settings,
            settings.resolved_placement(),
            include_fallbacks=True,
        )
        self.registry = create_unified_registry(self._aggregator)
        self._history = deque(maxlen=MONITORING_HISTORY_SAMPLES)
        self._last_health_probe = 0.0
        self._last_error = None
        self._lock = asyncio.Lock()
        self._task = None

    def start(
        self,
        /,
    ) -> None:
        if self._task is None:
            self._task = asyncio.create_task(
                self._sample_loop(),
                name="web-monitoring",
            )

    async def stop(
        self,
        /,
    ) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._aggregator.aclose()

    async def reconfigure(
        self,
        settings: Settings,
        /,
    ) -> None:
        """Apply endpoint, placement, token, and TLS changes without restarting Web."""
        async with self._lock:
            await self._aggregator.aclose()
            self.settings = settings
            self._aggregator.settings = settings
            self._aggregator.placement = settings.resolved_placement()

    async def sample_once(
        self,
        /,
    ) -> dict[str, Any]:
        async with self._lock:
            placement = self.settings.resolved_placement()
            monotonic_now = time.monotonic()
            probe_health = monotonic_now - self._last_health_probe >= HEALTH_REFRESH_SECONDS
            await self._aggregator.refresh(placement, probe_health=probe_health)
            if probe_health:
                self._last_health_probe = monotonic_now
            targets = tuple(
                (role, source)
                for role, source, _url in self._aggregator.targets(placement)
            )
            point = _monitoring_point(
                self.registry.collect(),
                targets,
                self.settings,
                time.time(),
            )
            self._history.append(point)
            self._last_error = None
            return point

    def snapshot(
        self,
        window_seconds: int = DEFAULT_MONITORING_WINDOW_SECONDS,
        /,
    ) -> dict[str, Any]:
        now = time.time()
        cutoff = now - window_seconds
        history = [point for point in self._history if point["timestamp"] >= cutoff]
        if history:
            latest = history[-1]
        else:
            placement = self.settings.resolved_placement()
            targets = tuple(
                (role, source)
                for role, source, _url in self._aggregator.targets(placement)
            )
            latest = _monitoring_point(
                self.registry.collect(),
                targets,
                self.settings,
                now,
            )
        return {
            "generated_at": now,
            "sample_interval_seconds": MONITORING_REFRESH_SECONDS,
            "window_seconds": window_seconds,
            "last_error": self._last_error,
            "summary": latest["summary"],
            "services": latest["services"],
            "series": [_series_point(point) for point in history],
        }

    async def _sample_loop(
        self,
        /,
    ) -> None:
        while True:
            started_at = time.monotonic()
            try:
                await self.sample_once()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - monitoring must remain available
                self._last_error = repr(error)
                log.error("monitoring.sample_failed", error=repr(error))
            elapsed = time.monotonic() - started_at
            await asyncio.sleep(max(0.1, MONITORING_REFRESH_SECONDS - elapsed))


def _monitoring_point(
    metric_families: Iterable[Metric],
    targets: Iterable[tuple[str, str]],
    settings: Settings,
    timestamp: float,
    /,
) -> dict[str, Any]:
    target_keys = tuple(dict.fromkeys(targets))
    services: dict[tuple[str, str], dict[str, Any]] = {
        key: _empty_service(*key) for key in target_keys
    }
    turns_total = 0.0
    over_budget_total = 0.0
    failovers_total = 0.0
    requests_total = 0.0
    request_totals: defaultdict[str, float] = defaultdict(float)
    first_audio_buckets: defaultdict[float, float] = defaultdict(float)
    token_rate_buckets: defaultdict[float, float] = defaultdict(float)

    for family in metric_families:
        for sample in family.samples:
            labels = sample.labels
            target = _sample_target(labels)
            if target in services:
                _apply_service_sample(services[target], sample)
            if sample.name == "kotonoha_http_requests_total":
                requests_total += float(sample.value)
                request_totals[_request_key(labels)] += float(sample.value)
            if "role" in labels:
                continue
            if sample.name == "kotonoha_turns_total":
                turns_total += float(sample.value)
            elif sample.name == "kotonoha_over_budget_turns_total":
                over_budget_total += float(sample.value)
            elif sample.name == "kotonoha_failovers_total":
                failovers_total += float(sample.value)
            elif (
                sample.name == "kotonoha_turn_stage_duration_seconds_bucket"
                and labels.get("stage") == "total_to_first_audio"
            ):
                first_audio_buckets[float(labels["le"])] += float(sample.value)
            elif sample.name == "kotonoha_turn_tokens_per_second_bucket":
                token_rate_buckets[float(labels["le"])] += float(sample.value)

    service_values = []
    memory_percent: dict[str, float | None] = {}
    ready_services = 0
    scraped_services = 0
    for key in target_keys:
        service = services[key]
        service["memory_percent"] = _memory_percent(service)
        service_key = f"{service['role']}@{service['source']}"
        memory_percent[service_key] = service["memory_percent"]
        ready_services += int(service["ready"] is True)
        scraped_services += int(service["scrape_up"] is True)
        service_values.append(service)

    first_audio_p95_seconds = _histogram_quantile(first_audio_buckets, 0.95)
    token_rate_p50 = _histogram_quantile(token_rate_buckets, 0.50)
    return {
        "timestamp": timestamp,
        "summary": {
            "services_ready": ready_services,
            "services_scraped": scraped_services,
            "services_total": len(service_values),
            "turns_total": int(turns_total),
            "over_budget_turns_total": int(over_budget_total),
            "failovers_total": int(failovers_total),
            "requests_total": int(requests_total),
            "first_audio_p95_ms": (
                round(first_audio_p95_seconds * 1000, 1)
                if first_audio_p95_seconds is not None
                else None
            ),
            "first_audio_budget_ms": settings.budget_ms.total,
            "tokens_per_second_p50": (
                round(token_rate_p50, 2) if token_rate_p50 is not None else None
            ),
        },
        "services": service_values,
        "memory_percent": memory_percent,
        "request_totals": dict(request_totals),
    }


def _empty_service(
    role: str,
    source: str,
    /,
) -> dict[str, Any]:
    return {
        "role": role,
        "source": source,
        "scrape_up": False,
        "ready": None,
        "accelerator_backend": "unknown",
        "memory_architecture": "unknown",
        "operating_system": "unknown",
        "kernel": "unknown",
        "machine": "unknown",
        "cpu_load_ratio": None,
        "system_memory_total_bytes": None,
        "system_memory_available_bytes": None,
        "accelerator_memory_total_bytes": None,
        "accelerator_memory_free_bytes": None,
        "accelerator_memory_reserved_bytes": None,
        "disk_total_bytes": None,
        "disk_used_bytes": None,
        "memory_percent": None,
    }


def _sample_target(
    labels: Mapping[str, str],
    /,
) -> tuple[str, str] | None:
    role = labels.get("role") or labels.get("service")
    source = labels.get("source")
    if role not in ROLES or source is None:
        return None
    return role, source


def _apply_service_sample(
    service: dict[str, Any],
    sample: Sample,
    /,
) -> None:
    labels = sample.labels
    value = float(sample.value)
    if sample.name == "kotonoha_remote_metrics_scrape_up":
        service["scrape_up"] = bool(value)
    elif sample.name == "kotonoha_service_up":
        service["ready"] = bool(value)
    elif sample.name == "kotonoha_service_accelerator_info":
        service["accelerator_backend"] = labels.get("backend", "unknown")
        service["memory_architecture"] = labels.get("memory_architecture", "unknown")
    elif sample.name == "kotonoha_service_system_info":
        service["operating_system"] = labels.get("os", "unknown")
        service["kernel"] = labels.get("kernel", "unknown")
        service["machine"] = labels.get("machine", "unknown")
    elif sample.name == "kotonoha_service_system_cpu_load_ratio":
        service["cpu_load_ratio"] = value
    elif sample.name == "kotonoha_service_memory_bytes":
        memory_key = f"{labels.get('scope')}_memory_{labels.get('state')}_bytes"
        if memory_key in service:
            service[memory_key] = value
    elif sample.name == "kotonoha_service_disk_bytes":
        disk_key = f"disk_{labels.get('state')}_bytes"
        if disk_key in service:
            service[disk_key] = value


def _memory_percent(
    service: Mapping[str, Any],
    /,
) -> float | None:
    system_total = service.get("system_memory_total_bytes")
    system_available = service.get("system_memory_available_bytes")
    if service.get("memory_architecture") == "unified":
        if isinstance(system_total, (int, float)) and isinstance(
            system_available,
            (int, float),
        ):
            if system_total > 0:
                return round(100 * (system_total - system_available) / system_total, 2)
    reserved = service.get("accelerator_memory_reserved_bytes")
    accelerator_total = service.get("accelerator_memory_total_bytes")
    if isinstance(reserved, (int, float)) and isinstance(accelerator_total, (int, float)):
        if accelerator_total > 0:
            return round(100 * reserved / accelerator_total, 2)
    if isinstance(system_total, (int, float)) and isinstance(system_available, (int, float)):
        if system_total > 0:
            return round(100 * (system_total - system_available) / system_total, 2)
    return None


def _request_key(
    labels: Mapping[str, str],
    /,
) -> str:
    role = labels.get("role") or labels.get("service", "unknown")
    source = labels.get("source", "local")
    return f"{role}@{source}"


def _histogram_quantile(
    buckets: Mapping[float, float],
    quantile: float,
    /,
) -> float | None:
    if not buckets:
        return None
    ordered = sorted(buckets.items())
    total = ordered[-1][1]
    if total <= 0:
        return None
    rank = total * quantile
    previous_upper = 0.0
    previous_count = 0.0
    for upper, count in ordered:
        if count < rank:
            previous_upper = upper
            previous_count = count
            continue
        if math.isinf(upper):
            return previous_upper if previous_count else None
        bucket_count = count - previous_count
        if bucket_count <= 0:
            return upper
        fraction = (rank - previous_count) / bucket_count
        return previous_upper + (upper - previous_upper) * fraction
    return None


def _series_point(
    point: Mapping[str, Any],
    /,
) -> dict[str, Any]:
    summary = point["summary"]
    return {
        "timestamp": point["timestamp"],
        "services_ready": summary["services_ready"],
        "services_total": summary["services_total"],
        "turns_total": summary["turns_total"],
        "over_budget_turns_total": summary["over_budget_turns_total"],
        "failovers_total": summary["failovers_total"],
        "requests_total": summary["requests_total"],
        "first_audio_p95_ms": summary["first_audio_p95_ms"],
        "memory_percent": point["memory_percent"],
        "request_totals": point["request_totals"],
    }
