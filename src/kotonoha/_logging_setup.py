"""structlog setup for files, service consoles, and the terminal interface.

The terminal interface replaces raw console JSON with an in-process bounded buffer.
The Textual application parses that JSON and renders human-readable records without
interfering with terminal control sequences.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import ClassVar, TextIO

import structlog

from kotonoha._secure_files import open_append_text
from kotonoha._typing import override

_TERMINAL_LOG_CAPACITY = 500
_terminal_log_messages: deque[str] = deque(maxlen=_TERMINAL_LOG_CAPACITY)
_terminal_log_lock = threading.Lock()


class TerminalInterfaceLogHandler(logging.Handler):
    """Retain JSON log lines until the Textual event loop consumes them."""
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
        with _terminal_log_lock:
            _terminal_log_messages.append(message)


class SecureFileHandler(RotatingFileHandler):
    """Append owner-only logs without following symbolic links."""

    __slots__: ClassVar[tuple[str, ...]] = ()

    @override
    def _open(
        self,
        /,
    ) -> TextIO:
        return open_append_text(Path(self.baseFilename))


def reset_terminal_interface_logs() -> None:
    """Discard records from a previous terminal-interface session."""
    with _terminal_log_lock:
        _terminal_log_messages.clear()


def drain_terminal_interface_logs(
    maximum: int | None = None,
    /,
) -> list[str]:
    """Remove a bounded group of buffered JSON lines in arrival order."""
    with _terminal_log_lock:
        if maximum is None or maximum >= len(_terminal_log_messages):
            messages = list(_terminal_log_messages)
            _terminal_log_messages.clear()
        else:
            messages = [_terminal_log_messages.popleft() for _message_index in range(maximum)]
    return messages


def setup_logging(
    level: str = "INFO",
    /,
    json_path: Path | None = None,
    console: bool = False,
    service: str = "orchestrator",
    terminal_interface: bool = False,
    maximum_bytes: int = 64 * 1024 * 1024,
    backup_count: int = 5,
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

    if console and terminal_interface:
        reset_terminal_interface_logs()
        terminal_handler = TerminalInterfaceLogHandler()
        terminal_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(terminal_handler)
    elif console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
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
