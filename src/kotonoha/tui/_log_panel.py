"""Kernel ring buffer rendering for structured JSON application logs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from rich.text import Text

from kotonoha._logging_setup import dmesg_parts

_RESERVED_FIELDS = {"event", "level", "logger", "service", "timestamp", "uptime"}
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

    stamp, level, service, event, fields = dmesg_parts(record)

    output = Text()
    output.append(stamp + " ", style="dim")
    output.append(f"{level:<7} ", style=_LEVEL_STYLES.get(level.lower(), "white"))
    output.append(f"{service}: ", style="magenta")
    output.append(event, style="bold")
    if fields:
        output.append(" " + fields, style="dim")
    return output
