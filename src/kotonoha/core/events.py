"""Events from the orchestrator to the UI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UiEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Assumes a single consumer (the TUI). Drops silently when nobody listens."""

    def __init__(self, maxsize: int = 512):
        self.q: asyncio.Queue[UiEvent] = asyncio.Queue(maxsize=maxsize)

    def emit(self, kind: str, **payload: Any) -> None:
        try:
            self.q.put_nowait(UiEvent(kind, payload))
        except asyncio.QueueFull:
            try:
                self.q.get_nowait()
                self.q.put_nowait(UiEvent(kind, payload))
            except Exception:  # noqa: BLE001
                pass

    async def get(self) -> UiEvent:
        return await self.q.get()
