"""The state machine (§4).

    IDLE ──speech detected──▶ LISTENING ──800ms silence──▶ PROCESSING ──first clause──▶ SPEAKING
      ▲                                                                                   │
      └────────────────────────────── queue drained ──────────────────────────────────────┘

Closing the microphone on entry to SPEAKING is the whole of half-duplex gating.
It happens in exactly one place — spread the gating logic around and one path
will inevitably miss it.
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
    State.LISTENING: {State.PROCESSING, State.IDLE},  # too-short utterance goes back to IDLE
    State.PROCESSING: {State.SPEAKING, State.IDLE},  # empty ASR or LLM timeout goes to IDLE
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
        """Error-recovery path. Returns to IDLE from any state."""
        prev, self._state = self._state, State.IDLE
        if prev is not State.IDLE and self._on_change:
            self._on_change(prev, State.IDLE, reason)
