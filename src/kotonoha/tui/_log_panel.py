"""Human-readable rendering for structured JSON application logs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from rich.text import Text

_RESERVED_FIELDS = {"event", "level", "logger", "service", "timestamp"}
_LEVEL_STYLES = {
    "critical": "bold white on red",
    "error": "bold red",
    "warning": "yellow",
    "info": "green",
    "debug": "dim cyan",
}


def _timestamp(
    value: Any,
    /,
) -> str:
    if not isinstance(value, str):
        return "--:--:--"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%H:%M:%S")
    except ValueError:
        return value[:8]


def _field_value(
    value: Any,
    /,
) -> str:
    if isinstance(value, str):
        return value.replace("\n", "\\n")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def format_json_log(
    raw_message: str,
    /,
) -> Text:
    """Parse one structlog JSON line into a compact operator-facing record."""
    try:
        record = json.loads(raw_message)
    except (json.JSONDecodeError, TypeError):
        return Text(raw_message, style="dim")
    if not isinstance(record, dict):
        return Text(_field_value(record), style="dim")

    level = str(record.get("level") or "info").lower()
    service = str(record.get("service") or record.get("logger") or "application")
    event = str(record.get("event") or "log")

    output = Text()
    output.append(_timestamp(record.get("timestamp")) + " ", style="dim")
    output.append(f"{level.upper():<8}", style=_LEVEL_STYLES.get(level, "white"))
    output.append(f"{service:<12}", style="magenta")
    output.append(event, style="bold")

    fields = [
        f"{key}={_field_value(value)}"
        for key, value in record.items()
        if key not in _RESERVED_FIELDS
    ]
    if fields:
        output.append("  " + "  ".join(fields), style="dim")
    return output
