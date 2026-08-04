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
    # Typed input skips LISTENING: there is no utterance to segment.
    State.IDLE: {State.LISTENING, State.PROCESSING},
    State.LISTENING: {State.PROCESSING, State.IDLE},  # too-short utterance goes back to IDLE
    State.PROCESSING: {State.SPEAKING, State.IDLE},  # empty ASR or LLM timeout goes to IDLE
    State.SPEAKING: {State.IDLE},
}


class IllegalTransition(RuntimeError):
    pass


class Machine:
    _state: State
    _on_change: Callable[[State, State, str], None] | None

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
        previous, self._state = self._state, target
        if self._on_change:
            self._on_change(previous, target, reason)
        return self._state

    def force_idle(self, reason: str = "reset") -> None:
        """Error-recovery path. Returns to IDLE from any state."""
        previous, self._state = self._state, State.IDLE
        if previous is not State.IDLE and self._on_change:
            self._on_change(previous, State.IDLE, reason)
