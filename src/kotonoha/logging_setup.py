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
from pathlib import Path

import structlog

_TERMINAL_LOG_CAPACITY = 500
_terminal_log_messages: deque[str] = deque(maxlen=_TERMINAL_LOG_CAPACITY)
_terminal_log_lock = threading.Lock()


class TerminalInterfaceLogHandler(logging.Handler):
    """Retain JSON log lines until the Textual event loop consumes them."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # noqa: BLE001
            self.handleError(record)
            return
        with _terminal_log_lock:
            _terminal_log_messages.append(message)


def reset_terminal_interface_logs() -> None:
    """Discard records from a previous terminal-interface session."""
    with _terminal_log_lock:
        _terminal_log_messages.clear()


def drain_terminal_interface_logs() -> list[str]:
    """Return buffered JSON lines and atomically clear the shared buffer."""
    with _terminal_log_lock:
        messages = list(_terminal_log_messages)
        _terminal_log_messages.clear()
    return messages


def setup_logging(
    level: str = "INFO",
    json_path: Path | None = None,
    console: bool = False,
    service: str = "orchestrator",
    terminal_interface: bool = False,
) -> structlog.stdlib.BoundLogger:
    handlers: list[logging.Handler] = []

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(json_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(fh)

    if console and terminal_interface:
        reset_terminal_interface_logs()
        terminal_handler = TerminalInterfaceLogHandler()
        terminal_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(terminal_handler)
    elif console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(sh)

    if not handlers:
        handlers.append(logging.NullHandler())

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
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


def get_logger(name: str = "kotonoha") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
