"""Shared asynchronous lifecycle helpers with an aiotools integration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Collection, Coroutine
from typing import Any

try:
    import aiotools
except ImportError:  # pragma: no cover - exercised by the Python 3.10 compatibility path.
    aiotools = None


async def cancel_and_wait(
    tasks: asyncio.Task[Any] | Collection[asyncio.Task[Any]],
    /,
) -> None:
    """Cancel tasks and wait for their completion without hiding parent cancellation."""
    if aiotools is not None:
        await aiotools.cancel_and_wait(tasks)
        return

    task_collection = (tasks,) if isinstance(tasks, asyncio.Task) else tuple(tasks)
    for task in task_collection:
        if not task.done():
            task.cancel()
    if not task_collection:
        return
    try:
        await asyncio.gather(*task_collection, return_exceptions=True)
    except asyncio.CancelledError:
        for task in task_collection:
            if not task.done():
                task.cancel()
        await asyncio.gather(*task_collection, return_exceptions=True)
        raise


def create_timer(
    callback: Callable[[float], Coroutine[Any, Any, None]],
    interval: float,
    /,
) -> asyncio.Task[None]:
    """Create a non-overlapping periodic task with an aiotools implementation when available."""
    if aiotools is not None:
        return aiotools.create_timer(
            callback,
            interval,
            aiotools.TimerDelayPolicy.CANCEL,
        )
    return asyncio.create_task(_fallback_timer(callback, interval))


async def _fallback_timer(
    callback: Callable[[float], Coroutine[Any, Any, None]],
    interval: float,
    /,
) -> None:
    while True:
        await asyncio.sleep(interval)
        await callback(interval)
