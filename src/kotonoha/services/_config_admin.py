"""SQLite configuration management API installed on the ASR service.

The ASR service is the control-plane endpoint because it already owns the authenticated
port at :8001. The API persists validated overrides only. Resident models are never
reloaded in the request path; the response states that a service restart is required.
"""

from __future__ import annotations

from typing import Any, ClassVar

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from kotonoha._call_compatibility import keyword_compatible
from kotonoha._config import (
    Settings,
    configuration_database_path,
    configuration_snapshot,
    load_settings,
)
from kotonoha._config_store import apply_changes, reset_changes

router = APIRouter(prefix="/admin", tags=["administration"])

# The service editor exposes settings consumed by resident model processes. Credentials,
# transport, browser policy, and local storage remain owned by the Web application.
SERVICE_EDITABLE_PATHS = frozenset(
    {
        "accelerator.profile",
        "asr.backend",
        "asr.dtype",
        "asr.model_id",
        "asr.model_revision",
        "asr.vllm_model_id",
        "asr.vllm_realtime_architecture",
        "asr.vllm_served_model_name",
        "asr.vllm_gpu_memory_utilization",
        "asr.vllm_max_model_len",
        "asr.vllm_enforce_eager",
        "asr_verify.backend",
        "asr_verify.compute_type",
        "asr_verify.device",
        "asr_verify.model_id",
        "llm.compilation_mode",
        "llm.compilation_cudagraph_capture_sizes",
        "llm.compilation_cache_dir",
        "llm.enable_prefix_caching",
        "llm.enforce_eager",
        "llm.gpu_memory_utilization",
        "llm.max_num_batched_tokens",
        "llm.max_model_len",
        "llm.max_num_seqs",
        "llm.models_dir",
        "llm.profile",
        "llm.profiles",
        "llm.served_model_name",
        "logging.prometheus_port",
        "tts.backend",
        "tts.deploy_config",
        "tts.enforce_eager",
        "tts.gpu_memory_utilization",
        "tts.max_audio_seconds",
        "tts.max_new_tokens",
        "tts.model_path",
        "tts.sample_rate",
        "tts.served_model_name",
        "tts.startup_timeout_s",
        "tts.task_type",
        "tts.timeout_s",
    }
)


class ConfigUpdate(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    changes: dict[str, Any]


class ConfigReset(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    paths: list[str] | None = None


def _serializable(
    settings: Settings,
    /,
) -> dict[str, Any]:
    return settings.model_dump(
        mode="json",
        exclude={"root": True, "transport": {"token"}},
    )


def _redacted_overrides(
    overrides: dict[str, Any],
    /,
) -> dict[str, Any]:
    overrides.pop("root", None)
    transport = overrides.get("transport")
    if isinstance(transport, dict):
        transport.pop("token", None)
    return overrides


def snapshot() -> dict[str, Any]:
    persisted = configuration_snapshot()
    settings = load_settings()
    return {
        "config": _serializable(settings),
        "defaults": _redacted_overrides(persisted.defaults),
        "editable_paths": sorted(SERVICE_EDITABLE_PATHS),
        "overrides": _redacted_overrides(persisted.overrides),
        "hardware_profile": persisted.profile,
        "hardware_profiles": list(persisted.profiles),
        "path": str(configuration_database_path()),
        "restart_required": True,
    }


@router.get("/config")
@keyword_compatible
async def get_config() -> dict[str, Any]:
    return snapshot()


@router.put("/config")
@keyword_compatible
async def put_config(
    update: ConfigUpdate,
    /,
) -> dict[str, Any]:
    return _put_config(update)


def _put_config(
    update: ConfigUpdate,
    /,
) -> dict[str, Any]:
    rejected = sorted(set(update.changes) - SERVICE_EDITABLE_PATHS)
    if rejected:
        paths = ", ".join(rejected)
        raise HTTPException(422, f"settings are not editable on the model service: {paths}")
    result = apply_changes(update.changes)
    if not result.written:
        raise HTTPException(422, result.error or "invalid configuration")
    return snapshot()


@router.delete("/config")
@keyword_compatible
async def delete_config(
    reset: ConfigReset,
    /,
) -> dict[str, Any]:
    rejected = sorted(set(reset.paths or ()) - SERVICE_EDITABLE_PATHS)
    if rejected:
        paths = ", ".join(rejected)
        raise HTTPException(422, f"settings are not editable on the model service: {paths}")
    paths = list(reset.paths) if reset.paths is not None else sorted(SERVICE_EDITABLE_PATHS)
    result = reset_changes(paths)
    if not result.written:
        raise HTTPException(422, result.error or "invalid configuration reset")
    return snapshot()
