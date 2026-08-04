"""Events from the orchestrator to the UI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, ClassVar

from kotonoha._typing import override


@dataclass(slots=True)
class UiEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Assumes a single consumer (the TUI). Drops silently when nobody listens."""
    __slots__: ClassVar[tuple[str, ...]] = (
        "queue",
    )

    queue: asyncio.Queue[UiEvent]

    @override
    def __init__(
        self,
        /,
        maxsize: int = 512,
    ) -> None:
        self.queue = asyncio.Queue(maxsize=maxsize)

    def emit(
        self,
        /,
        kind: str,
        **payload: Any,
    ) -> None:
        try:
            self.queue.put_nowait(UiEvent(kind, payload))
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(UiEvent(kind, payload))
            except Exception:  # noqa: BLE001
                pass

    async def get(
        self,
        /,
    ) -> UiEvent:
        return await self.queue.get()

    def drain_nowait(
        self,
        /,
        maximum: int = 127,
    ) -> list[UiEvent]:
        """Remove a bounded burst so the consumer can commit it as one frame."""
        events: list[UiEvent] = []
        for _event_index in range(maximum):
            try:
                events.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events
