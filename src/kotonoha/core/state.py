"""§4 상태 기계.

    IDLE ──음성 감지──▶ LISTENING ──침묵 800ms──▶ PROCESSING ──첫 절──▶ SPEAKING
      ▲                                                                    │
      └──────────────────────── 큐 소진 ───────────────────────────────────┘

SPEAKING 진입 시 마이크를 닫는 것이 반이중 게이팅의 전부다. 여기서만 닫고
여기서만 연다 — 게이팅 로직이 여러 군데로 흩어지면 반드시 한 군데가 빠진다.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum


class State(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"


ALLOWED: dict[State, set[State]] = {
    State.IDLE: {State.LISTENING},
    State.LISTENING: {State.PROCESSING, State.IDLE},  # 너무 짧은 발화는 IDLE 로 되돌림
    State.PROCESSING: {State.SPEAKING, State.IDLE},  # 빈 전사·LLM 타임아웃은 IDLE 로
    State.SPEAKING: {State.IDLE},
}


class IllegalTransition(RuntimeError):
    pass


class Machine:
    def __init__(self, on_change: Callable[[State, State, str], None] | None = None):
        self._state = State.IDLE
        self._on_change = on_change

    @property
    def state(self) -> State:
        return self._state

    def can(self, to: State) -> bool:
        return to in ALLOWED[self._state]

    def to(self, target: State, reason: str = "") -> State:
        if target is self._state:
            return self._state
        if target not in ALLOWED[self._state]:
            raise IllegalTransition(f"{self._state.value} → {target.value} ({reason})")
        prev, self._state = self._state, target
        if self._on_change:
            self._on_change(prev, target, reason)
        return self._state

    def force_idle(self, reason: str = "reset") -> None:
        """오류 복구 경로. 어떤 상태에서든 IDLE 로 되돌린다."""
        prev, self._state = self._state, State.IDLE
        if prev is not State.IDLE and self._on_change:
            self._on_change(prev, State.IDLE, reason)
