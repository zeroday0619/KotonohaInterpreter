"""Shared uvloop entry point for synchronous command handlers."""

from __future__ import annotations

from collections.abc import Coroutine
from typing import Any, TypeVar

import uvloop

Result = TypeVar("Result")


def run(coroutine: Coroutine[Any, Any, Result]) -> Result:
    """Run one top-level coroutine on uvloop.

    A required dependency is used instead of a silent asyncio fallback so a
    deployment cannot report uvloop while running the standard event loop.
    """
    return uvloop.run(coroutine)
