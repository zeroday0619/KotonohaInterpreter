"""오케스트레이터 → UI 이벤트."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UiEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """단일 소비자(TUI) 전제. 소비자가 없으면 조용히 버린다."""

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
