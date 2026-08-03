"""Validation and persistence shared by the local and remote configuration editors."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .config import Settings, config_layers, deep_merge, local_config_path, read_yaml


def get_path(data: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data


def set_path(data: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        data = data.setdefault(part, {})
    data[parts[-1]] = value


def validate_candidate(config_path: Path | None, local: dict) -> str | None:
    """Return None when the candidate loads, otherwise a one-line reason."""
    merged: dict = {}
    for layer in config_layers(config_path):
        merged = deep_merge(merged, read_yaml(layer))
    merged = deep_merge(merged, local)
    try:
        Settings(**merged)
    except ValidationError as error:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        return f"{location}: {first['msg']}"
    except Exception as error:  # noqa: BLE001
        return repr(error)
    return None


def write_local(path: Path, local: dict) -> None:
    """Atomically replace an override file after its candidate has validated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Written by `kotonoha config`. Host-specific overrides.\n"
        "# This layer overrides config/default.yaml and the selected overlay.\n\n"
    )
    content = header + yaml.safe_dump(local, allow_unicode=True, sort_keys=False)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


@dataclass
class ApplyResult:
    written: bool
    error: str | None = None
    changed: list[str] = field(default_factory=list)


def apply_changes(
    changes: dict[str, Any],
    config_path: Path | None = None,
    local_path: Path | None = None,
) -> ApplyResult:
    """Validate and persist dotted-path changes without starting the interface."""
    local_path = local_path or local_config_path()
    candidate = read_yaml(local_path) if local_path.exists() else {}
    for path, value in changes.items():
        set_path(candidate, path, value)

    problem = validate_candidate(config_path, candidate)
    if problem:
        return ApplyResult(written=False, error=problem)

    write_local(local_path, candidate)
    return ApplyResult(written=True, changed=sorted(changes))


__all__ = [
    "ApplyResult",
    "apply_changes",
    "get_path",
    "set_path",
    "validate_candidate",
    "write_local",
]
