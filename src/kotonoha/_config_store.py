"""Validation, SQLite persistence, and reset for Web-managed settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kotonoha._config import (
    REPO_ROOT,
    Settings,
    configuration_database_path,
    configuration_snapshot,
    configuration_snapshot_for_profile,
    deep_merge,
)
from kotonoha._settings_database import write_overrides


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


def delete_path(
    data: dict[str, Any],
    path: str,
    /,
) -> bool:
    """Delete one dotted override and prune empty parent mappings."""
    parts = path.split(".")
    current: dict[str, Any] = data
    parents: list[tuple[dict[str, Any], str]] = []
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            return False
        parents.append((current, part))
        current = child
    if not parts or parts[-1] not in current:
        return False
    del current[parts[-1]]
    for parent, key in reversed(parents):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            del parent[key]
        else:
            break
    return True


def _validation_error(
    candidate: dict[str, Any],
    /,
) -> str | None:
    try:
        Settings(**candidate)
    except ValidationError as error:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        return f"{location}: {first['msg']}"
    except Exception as error:  # noqa: BLE001
        return repr(error)
    return None


def validate_candidate(
    config_path: Path | None,
    /,
    local: dict,
) -> str | None:
    """Return None when SQLite overrides form a valid effective configuration."""
    snapshot = configuration_snapshot(config_path)
    profile_value = get_path(local, "accelerator.profile")
    profile = profile_value if isinstance(profile_value, str) else snapshot.profile
    defaults = configuration_snapshot_for_profile(profile, config_path).defaults
    overrides = deep_merge({}, local)
    delete_path(overrides, "accelerator.profile")
    candidate = deep_merge(defaults, overrides)
    candidate["root"] = str(REPO_ROOT)
    return _validation_error(candidate)


@dataclass(slots=True)
class ApplyResult:
    written: bool
    error: str | None = None
    changed: list[str] = field(default_factory=list)


def apply_changes(
    changes: dict[str, Any],
    /,
    config_path: Path | None = None,
    database_path: Path | None = None,
) -> ApplyResult:
    """Validate and persist dotted-path changes in the SQLite override layer."""
    snapshot = configuration_snapshot(config_path)
    candidate = deep_merge({}, snapshot.overrides)
    profile = snapshot.profile
    try:
        for path, value in changes.items():
            if path == "accelerator.profile":
                if not isinstance(value, str):
                    raise ValueError("accelerator.profile must be a hardware profile name")
                profile = value
            else:
                set_path(candidate, path, value)
    except ValueError as error:
        return ApplyResult(written=False, error=str(error))

    profile_snapshot = configuration_snapshot_for_profile(profile, config_path)
    effective = deep_merge(profile_snapshot.defaults, candidate)
    effective["root"] = str(REPO_ROOT)
    problem = _validation_error(effective)
    if problem:
        return ApplyResult(written=False, error=problem)

    try:
        write_overrides(
            database_path or configuration_database_path(),
            candidate,
            profile,
        )
    except (OSError, ValueError) as error:
        return ApplyResult(written=False, error=str(error))
    return ApplyResult(written=True, changed=sorted(changes))


def reset_changes(
    paths: list[str] | None = None,
    /,
    config_path: Path | None = None,
    database_path: Path | None = None,
) -> ApplyResult:
    """Delete selected overrides, or every override, without changing the profile."""
    snapshot = configuration_snapshot(config_path)
    candidate = deep_merge({}, snapshot.overrides)
    changed: list[str]
    if paths is None:
        changed = sorted(_leaf_paths(candidate))
        candidate = {}
    else:
        changed = sorted(path for path in paths if delete_path(candidate, path))
    effective = deep_merge(
        configuration_snapshot_for_profile(snapshot.profile, config_path).defaults,
        candidate,
    )
    effective["root"] = str(REPO_ROOT)
    problem = _validation_error(effective)
    if problem:
        return ApplyResult(written=False, error=problem)
    try:
        write_overrides(
            database_path or configuration_database_path(),
            candidate,
            snapshot.profile,
        )
    except (OSError, ValueError) as error:
        return ApplyResult(written=False, error=str(error))
    return ApplyResult(written=True, changed=changed)


def _leaf_paths(
    values: dict[str, Any],
    /,
    prefix: str = "",
) -> list[str]:
    paths: list[str] = []
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            paths.extend(_leaf_paths(value, path))
        else:
            paths.append(path)
    return paths
