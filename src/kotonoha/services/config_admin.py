"""Configuration management API installed on the remote ASR service.

The ASR service is the control-plane endpoint because it already owns the authenticated
port at :8001. The API persists validated overrides only. Resident models are never
reloaded in the request path; the response states that a service restart is required.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import Settings, load_settings, local_config_path, read_yaml
from ..config_store import apply_changes

router = APIRouter(prefix="/admin", tags=["administration"])

# The remote editor exposes settings consumed by resident model processes. Client policy,
# credentials, transport, and local storage remain owned by the Orin configuration.
REMOTE_EDITABLE_PATHS = frozenset(
    {
        "asr.backend",
        "asr.dtype",
        "asr.model_id",
        "asr.vllm_model_id",
        "asr_verify.backend",
        "asr_verify.compute_type",
        "asr_verify.device",
        "asr_verify.model_id",
        "llm.models_dir",
        "llm.n_batch",
        "llm.n_ctx",
        "llm.profile",
        "llm.profiles",
        "tts.backend",
        "tts.chunk_ms",
        "tts.fallback",
        "tts.model_id",
    }
)


class ConfigUpdate(BaseModel):
    changes: dict[str, Any]


def _base_config_path() -> Path | None:
    raw = os.environ.get("KOTONOHA_CONFIG")
    return Path(raw) if raw else None


def _serializable(settings: Settings) -> dict[str, Any]:
    return settings.model_dump(mode="json", exclude={"root"})


def _write_llm_environment(settings: Settings) -> None:
    """Write values consumed by run_llm.sh after the remote stack restarts."""
    target_raw = os.environ.get("KOTONOHA_LLM_CONFIG_ENV")
    if not target_raw:
        return
    target = Path(target_raw)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "LLM_BATCH": str(settings.llm.n_batch),
        "LLM_PROFILE": settings.llm.profile,
        "LLM_MODEL": str(settings.llm.gguf_path),
        "MODELS_DIR": str(settings.llm.models_dir),
        "LLM_CTX": str(settings.llm.n_ctx),
        "LLM_NGL": str(settings.llm.active.n_gpu_layers),
    }
    content = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in values.items())
    target.write_text(content + "\n", encoding="utf-8")


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
def get_config() -> dict[str, Any]:
    return snapshot()


@router.put("/config")
def put_config(update: ConfigUpdate) -> dict[str, Any]:
    rejected = sorted(set(update.changes) - REMOTE_EDITABLE_PATHS)
    if rejected:
        paths = ", ".join(rejected)
        raise HTTPException(422, f"settings are not editable on the remote server: {paths}")
    target = local_config_path()
    result = apply_changes(update.changes, _base_config_path(), target)
    if not result.written:
        raise HTTPException(422, result.error or "invalid configuration")
    settings = load_settings(_base_config_path())
    _write_llm_environment(settings)
    return snapshot()


__all__ = ["REMOTE_EDITABLE_PATHS", "router", "snapshot"]
