"""SQLite persistence for configuration defaults and operator overrides."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from kotonoha._secure_files import reject_symbolic_link

BASE_PROFILE: Final = "__base__"
DATABASE_ENVIRONMENT_VARIABLE: Final = "KOTONOHA_SETTINGS_DATABASE"

SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS configuration_defaults (
    profile     TEXT PRIMARY KEY,
    values_json TEXT NOT NULL,
    updated_at  REAL NOT NULL DEFAULT (unixepoch('subsec'))
);
CREATE TABLE IF NOT EXISTS configuration_state (
    key         TEXT PRIMARY KEY,
    value_json  TEXT NOT NULL,
    updated_at  REAL NOT NULL DEFAULT (unixepoch('subsec'))
);
"""


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshot:
    """Represent the selected profile, its defaults, and persisted overrides."""

    profile: str
    profiles: tuple[str, ...]
    defaults: dict[str, Any]
    overrides: dict[str, Any]
    effective: dict[str, Any]


def settings_database_path(
    root: Path,
    /,
) -> Path:
    """Return the configuration database shared with history and glossary data."""
    configured = os.environ.get(DATABASE_ENVIRONMENT_VARIABLE)
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else (root / path).resolve()
    return (root / "data" / "kotonoha.db").resolve()


def _connect(
    path: Path,
    /,
) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symbolic_link(path)
    connection = sqlite3.connect(str(path), timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.executescript(SCHEMA)
    connection.commit()
    for suffix in ("", "-wal", "-shm"):
        candidate = path.with_name(f"{path.name}{suffix}")
        try:
            candidate.chmod(0o600)
        except FileNotFoundError:
            continue
    return connection


def _encode(
    value: Any,
    /,
) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _decode_object(
    value: str,
    /,
) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("stored configuration value must be a JSON object")
    return decoded


def _state_value(
    connection: sqlite3.Connection,
    key: str,
    /,
) -> Any:
    row = connection.execute(
        "SELECT value_json FROM configuration_state WHERE key = ?",
        (key,),
    ).fetchone()
    return None if row is None else json.loads(row["value_json"])


def _write_state(
    connection: sqlite3.Connection,
    key: str,
    value: Any,
    /,
) -> None:
    connection.execute(
        """
        INSERT INTO configuration_state (key, value_json, updated_at)
        VALUES (?, ?, unixepoch('subsec'))
        ON CONFLICT(key) DO UPDATE SET
            value_json = excluded.value_json,
            updated_at = excluded.updated_at
        """,
        (key, _encode(value)),
    )


def seed_defaults(
    path: Path,
    base: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    default_profile: str,
    /,
) -> None:
    """Refresh built-in defaults while preserving operator state."""
    if default_profile not in profiles:
        raise ValueError(f"default hardware profile is not registered: {default_profile}")
    connection = _connect(path)
    try:
        with connection:
            documents = {BASE_PROFILE: base, **profiles}
            for profile, values in documents.items():
                connection.execute(
                    """
                    INSERT INTO configuration_defaults (profile, values_json, updated_at)
                    VALUES (?, ?, unixepoch('subsec'))
                    ON CONFLICT(profile) DO UPDATE SET
                        values_json = excluded.values_json,
                        updated_at = excluded.updated_at
                    """,
                    (profile, _encode(values)),
                )
            connection.execute(
                "DELETE FROM configuration_defaults WHERE profile NOT IN "
                f"({','.join('?' for _profile in documents)})",
                tuple(documents),
            )
            if _state_value(connection, "hardware_profile") is None:
                _write_state(connection, "hardware_profile", default_profile)
            if _state_value(connection, "overrides") is None:
                _write_state(connection, "overrides", {})
    finally:
        connection.close()


def read_snapshot(
    path: Path,
    default_profile: str,
    /,
    *,
    forced_profile: str | None = None,
    include_overrides: bool = True,
) -> ConfigurationSnapshot:
    connection = _connect(path)
    try:
        profile_rows = connection.execute(
            "SELECT profile, values_json FROM configuration_defaults ORDER BY profile"
        ).fetchall()
        documents = {
            row["profile"]: _decode_object(row["values_json"])
            for row in profile_rows
        }
        profiles = tuple(profile for profile in documents if profile != BASE_PROFILE)
        selected = forced_profile or _state_value(connection, "hardware_profile") or default_profile
        if selected not in profiles:
            raise ValueError(f"hardware profile is not registered: {selected}")
        overrides = _state_value(connection, "overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError("stored configuration overrides must be a JSON object")
        defaults = _deep_merge(documents.get(BASE_PROFILE, {}), documents[selected])
        effective = _deep_merge(defaults, overrides if include_overrides else {})
        return ConfigurationSnapshot(
            profile=selected,
            profiles=profiles,
            defaults=defaults,
            overrides=overrides if include_overrides else {},
            effective=effective,
        )
    finally:
        connection.close()


def write_overrides(
    path: Path,
    overrides: dict[str, Any],
    profile: str,
    /,
) -> None:
    connection = _connect(path)
    try:
        registered = connection.execute(
            "SELECT 1 FROM configuration_defaults WHERE profile = ?",
            (profile,),
        ).fetchone()
        if registered is None:
            raise ValueError(f"hardware profile is not registered: {profile}")
        with connection:
            _write_state(connection, "hardware_profile", profile)
            _write_state(connection, "overrides", overrides)
    finally:
        connection.close()


def migration_completed(
    path: Path,
    migration: str,
    /,
) -> bool:
    connection = _connect(path)
    try:
        return bool(_state_value(connection, f"migration:{migration}"))
    finally:
        connection.close()


def mark_migration_completed(
    path: Path,
    migration: str,
    /,
) -> None:
    connection = _connect(path)
    try:
        with connection:
            _write_state(connection, f"migration:{migration}", True)
    finally:
        connection.close()


def _deep_merge(
    base: dict[str, Any],
    overlay: dict[str, Any],
    /,
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
