"""structlog setup for files, service consoles, and the Web log stream."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, ClassVar, Final, TextIO

import structlog

from kotonoha._secure_files import open_append_text
from kotonoha._typing import override

_WEB_LOG_CAPACITY = 500
_web_log_messages: deque[str] = deque(maxlen=_WEB_LOG_CAPACITY)
_web_log_lock = threading.Lock()

# Uptime is measured from import so that every record in one process shares a
# monotonic origin, the way the kernel ring buffer counts from boot.
_PROCESS_START: Final = time.monotonic()

# Fields the dmesg layout renders in its own columns instead of as key=value pairs.
DMESG_RESERVED_FIELDS: Final = frozenset(
    {"event", "level", "logger", "service", "timestamp", "uptime"}
)


def add_uptime(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
    /,
) -> dict[str, Any]:
    """Stamp seconds since process start onto every record."""
    del logger, method_name
    event_dict["uptime"] = round(time.monotonic() - _PROCESS_START, 6)
    return event_dict


def _field_value(
    value: Any,
    /,
) -> str:
    if isinstance(value, str):
        return value.replace("\n", "\\n")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def dmesg_parts(
    record: dict[str, Any],
    /,
) -> tuple[str, str, str, str, str]:
    """Split one structured record into its dmesg columns.

    Returning parts keeps the formatting reusable across browser and console output.
    """
    uptime = record.get("uptime")
    stamp = (
        f"[{float(uptime):12.6f}]"
        if isinstance(uptime, (int, float)) and not isinstance(uptime, bool)
        else "[" + " " * 12 + "]"
    )
    level = str(record.get("level") or "info").upper()
    service = str(record.get("service") or record.get("logger") or "application")
    event = str(record.get("event") or "log")
    fields = " ".join(
        f"{key}={_field_value(value)}"
        for key, value in record.items()
        if key not in DMESG_RESERVED_FIELDS
    )
    return stamp, level, service, event, fields


def render_dmesg(
    raw_message: str,
    /,
) -> str:
    """Render one structlog JSON line in the kernel ring buffer layout."""
    try:
        record = json.loads(raw_message)
    except (json.JSONDecodeError, TypeError):
        return raw_message
    if not isinstance(record, dict):
        return raw_message

    stamp, level, service, event, fields = dmesg_parts(record)
    line = f"{stamp} {level:<7} {service}: {event}"
    return f"{line} {fields}" if fields else line


class DmesgFormatter(logging.Formatter):
    """Print structured records as kernel-style console lines."""

    __slots__: ClassVar[tuple[str, ...]] = ()

    @override
    def format(
        self,
        /,
        record: logging.LogRecord,
    ) -> str:
        return render_dmesg(record.getMessage())


class WebLogHandler(logging.Handler):
    """Retain JSON log lines until the Web broadcaster consumes them."""
    __slots__: ClassVar[tuple[str, ...]] = ()

    @override
    def emit(
        self,
        /,
        record: logging.LogRecord,
    ) -> None:
        try:
            message = self.format(record)
        except Exception:  # noqa: BLE001
            self.handleError(record)
            return
        with _web_log_lock:
            _web_log_messages.append(message)


class SecureFileHandler(RotatingFileHandler):
    """Append owner-only logs without following symbolic links."""

    __slots__: ClassVar[tuple[str, ...]] = ()

    @override
    def _open(
        self,
        /,
    ) -> TextIO:
        return open_append_text(Path(self.baseFilename))


def reset_web_logs() -> None:
    """Discard records from a previous Web application session."""
    with _web_log_lock:
        _web_log_messages.clear()


def drain_web_logs(
    maximum: int | None = None,
    /,
) -> list[str]:
    """Remove a bounded group of buffered JSON lines in arrival order."""
    with _web_log_lock:
        if maximum is None or maximum >= len(_web_log_messages):
            messages = list(_web_log_messages)
            _web_log_messages.clear()
        else:
            messages = [_web_log_messages.popleft() for _message_index in range(maximum)]
    return messages


def setup_logging(
    level: str = "INFO",
    /,
    json_path: Path | None = None,
    console: bool = False,
    service: str = "orchestrator",
    web_interface: bool = False,
    maximum_bytes: int = 64 * 1024 * 1024,
    backup_count: int = 5,
    console_format: str = "dmesg",
) -> structlog.stdlib.BoundLogger:
    handlers: list[logging.Handler] = []

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = SecureFileHandler(
            json_path,
            maxBytes=maximum_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(file_handler)

    if console and web_interface:
        reset_web_logs()
        web_handler = WebLogHandler()
        web_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(web_handler)
    elif console:
        stream_handler = logging.StreamHandler(sys.stderr)
        # The file handler above keeps the structured form. Only what a person
        # reads on a console is reformatted.
        stream_handler.setFormatter(
            DmesgFormatter() if console_format == "dmesg" else logging.Formatter("%(message)s")
        )
        handlers.append(stream_handler)

    if not handlers:
        handlers.append(logging.NullHandler())

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            add_uptime,
            structlog.processors.TimeStamper(fmt="iso", utc=False),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger().bind(service=service)


def get_logger(
    name: str = "kotonoha",
    /,
) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
