"""Fan the process-wide log buffer out to every connected browser session.

`drain_terminal_interface_logs` removes what it returns, which suits one terminal
interface and breaks with several readers: two sessions draining concurrently
would each receive a different half of the stream. One task owns the drain here
and every session subscribes to the result, so all of them see the same records.
Late joiners receive the recent history first, because a log panel that starts
empty tells an operator nothing about the failure they opened it to inspect.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, ClassVar, Final

from kotonoha._logging_setup import drain_terminal_interface_logs, render_dmesg

POLL_SECONDS: Final = 0.25
HISTORY_LIMIT: Final = 400
SUBSCRIBER_BACKLOG: Final = 200


class LogBroadcaster:
    """Single reader of the shared buffer, many per-session queues."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "_history",
        "_subscribers",
        "_task",
    )

    _history: deque[str]
    _subscribers: set[asyncio.Queue[str]]
    _task: asyncio.Task[None] | None

    def __init__(
        self,
        /,
    ) -> None:
        self._history = deque(maxlen=HISTORY_LIMIT)
        self._subscribers = set()
        self._task = None

    def start(
        self,
        /,
    ) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._pump(), name="web-log-pump")

    async def stop(
        self,
        /,
    ) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def subscribe(
        self,
        /,
    ) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=SUBSCRIBER_BACKLOG)
        for line in self._history:
            try:
                queue.put_nowait(line)
            except asyncio.QueueFull:
                break
        self._subscribers.add(queue)
        return queue

    def unsubscribe(
        self,
        /,
        queue: asyncio.Queue[str],
    ) -> None:
        self._subscribers.discard(queue)

    def publish(
        self,
        /,
        line: str,
    ) -> None:
        self._history.append(line)
        for queue in self._subscribers:
            try:
                queue.put_nowait(line)
            except asyncio.QueueFull:
                # A session that cannot keep up loses its oldest record rather
                # than stalling the pump for everyone else.
                try:
                    queue.get_nowait()
                    queue.put_nowait(line)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    continue

    async def _pump(
        self,
        /,
    ) -> None:
        while True:
            for raw_message in drain_terminal_interface_logs():
                self.publish(render_dmesg(raw_message))
            await asyncio.sleep(POLL_SECONDS)

    def history(
        self,
        /,
        limit: int = HISTORY_LIMIT,
    ) -> list[Any]:
        records = list(self._history)
        return records[-limit:] if limit > 0 else []
