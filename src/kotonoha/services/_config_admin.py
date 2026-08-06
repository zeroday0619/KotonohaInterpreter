"""Configuration management API installed on the remote ASR service.

The ASR service is the control-plane endpoint because it already owns the authenticated
port at :8001. The API persists validated overrides only. Resident models are never
reloaded in the request path; the response states that a service restart is required.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from kotonoha._call_compatibility import keyword_compatible
from kotonoha._config import Settings, load_settings, local_config_path, read_yaml
from kotonoha._config_store import apply_changes

router = APIRouter(prefix="/admin", tags=["administration"])

# The remote editor exposes settings consumed by resident model processes. Client policy,
# credentials, transport, and local storage remain owned by the Orin configuration.
REMOTE_EDITABLE_PATHS = frozenset(
    {
        "accelerator.profile",
        "asr.backend",
        "asr.dtype",
        "asr.model_id",
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
    }
)


class ConfigUpdate(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    changes: dict[str, Any]


def _base_config_path() -> Path | None:
    raw = os.environ.get("KOTONOHA_CONFIG")
    return Path(raw) if raw else None


def _serializable(
    settings: Settings,
    /,
) -> dict[str, Any]:
    return settings.model_dump(mode="json", exclude={"root"})


def snapshot() -> dict[str, Any]:
    target = local_config_path()
    settings = load_settings(_base_config_path())
    return {
        "config": _serializable(settings),
        "editable_paths": sorted(REMOTE_EDITABLE_PATHS),
        "overrides": read_yaml(target) if target.exists() else {},
        "path": str(target),
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
    rejected = sorted(set(update.changes) - REMOTE_EDITABLE_PATHS)
    if rejected:
        paths = ", ".join(rejected)
        raise HTTPException(422, f"settings are not editable on the remote server: {paths}")
    target = local_config_path()
    result = apply_changes(update.changes, _base_config_path(), target)
    if not result.written:
        raise HTTPException(422, result.error or "invalid configuration")
    return snapshot()
