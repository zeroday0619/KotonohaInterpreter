"""structlog setup — JSON to a file.

The TUI owns the terminal, so console output is off by default. Service
processes turn it back on.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(
    level: str = "INFO",
    json_path: Path | None = None,
    console: bool = False,
    service: str = "orchestrator",
) -> structlog.stdlib.BoundLogger:
    handlers: list[logging.Handler] = []

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(json_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(fh)

    if console:
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
