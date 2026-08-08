"""Validation and persistence shared by the local and remote configuration editors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from kotonoha._config import Settings, config_layers, deep_merge, local_config_path, read_yaml
from kotonoha._secure_files import atomic_write_text


def get_path(
    data: Any,
    /,
    path: str,
) -> Any:
    for part in path.split("."):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data


def set_path(
    data: dict,
    /,
    path: str,
    value: Any,
) -> None:
    parts = path.split(".")
    if not path or any(not part for part in parts):
        raise ValueError("configuration path must contain non-empty components")
    current = data
    for index, part in enumerate(parts[:-1], start=1):
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        elif not isinstance(child, dict):
            parent = ".".join(parts[:index])
            raise ValueError(f"configuration path crosses non-mapping value: {parent}")
        current = child
    current[parts[-1]] = value


def validate_candidate(
    config_path: Path | None,
    /,
    local: dict,
) -> str | None:
    """Return None when the candidate loads, otherwise a one-line reason."""
    merged: dict = {}
    for layer in config_layers(config_path, local_override=local):
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


def write_local(
    path: Path,
    /,
    local: dict,
) -> None:
    """Atomically replace an override file after its candidate has validated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Written by the Web configuration editor. Host-specific overrides.\n"
        "# This layer overrides config/default.yaml and the selected overlay.\n\n"
    )
    content = header + yaml.safe_dump(local, allow_unicode=True, sort_keys=False)
    atomic_write_text(path, content)


@dataclass(slots=True)
class ApplyResult:
    written: bool
    error: str | None = None
    changed: list[str] = field(default_factory=list)


def apply_changes(
    changes: dict[str, Any],
    /,
    config_path: Path | None = None,
    local_path: Path | None = None,
) -> ApplyResult:
    """Validate and persist dotted-path changes without starting the interface."""
    local_path = local_path or local_config_path()
    candidate = read_yaml(local_path) if local_path.exists() else {}
    try:
        for path, value in changes.items():
            set_path(candidate, path, value)
    except ValueError as error:
        return ApplyResult(written=False, error=str(error))

    problem = validate_candidate(config_path, candidate)
    if problem:
        return ApplyResult(written=False, error=problem)

    write_local(local_path, candidate)
    return ApplyResult(written=True, changed=sorted(changes))
