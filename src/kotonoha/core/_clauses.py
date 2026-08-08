"""Clause-level streaming split (§5.4).

Rather than waiting for the LLM to finish, each clause goes to the TTS queue as
soon as it is complete. As the spec notes, this only works when the LLM sustains
at least 5 tok/s: roughly 4-5 tokens are needed per second of speech, so below
that the next clause is not ready before the current one finishes playing.

Two things to watch:
  · Cut too finely and TTS applies sentence-final intonation to every fragment.
  · Cut too coarsely and the first audio is late. So the first clause is allowed
    to be short and later ones accumulate more.

And the text after the ⟦SRC⟧ marker (the reconstructed source) must never reach
TTS. That includes the case where the marker arrives split across stream deltas.
"""

from __future__ import annotations

from typing import ClassVar

from kotonoha._typing import override
from kotonoha.prompts._translate import SRC_MARKER

HARD_TERMINATORS = "。．.!?！？…‥\n"
SOFT_TERMINATORS = "、，,;；:：）)】」』"


class ClauseStreamer:
    __slots__: ClassVar[tuple[str, ...]] = (
        "_all",
        "_count",
        "_emitted",
        "_stopped",
        "first_min_chars",
        "marker",
        "max_chars",
        "min_chars",
    )
    first_min_chars: int
    min_chars: int
    max_chars: int
    marker: str
    _all: str
    _emitted: int
    _count: int
    _stopped: bool

    @override
    def __init__(
        self,
        /,
        first_min_chars: int = 6,
        min_chars: int = 14,
        max_chars: int = 90,
        marker: str = SRC_MARKER,
    ) -> None:
        self.first_min_chars = first_min_chars
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.marker = marker
        self._all = ""
        self._emitted = 0
        self._count = 0
        self._stopped = False

    # -- state -----------------------------------------------------------
    @property
    def stopped(
        self,
        /,
    ) -> bool:
        return self._stopped

    @property
    def raw(
        self,
        /,
    ) -> str:
        return self._all

    @property
    def translation(
        self,
        /,
    ) -> str:
        marker_index = self._all.find(self.marker)
        return (self._all if marker_index < 0 else self._all[:marker_index]).strip()

    @property
    def clause_count(
        self,
        /,
    ) -> int:
        return self._count

    # -- streaming -------------------------------------------------------
    def push(
        self,
        /,
        delta: str,
    ) -> list[str]:
        self._all += delta
        if self._stopped:
            return []

        marker_index = self._all.find(self.marker)
        if marker_index >= 0:
            self._stopped = True
            available = self._all[:marker_index]
            clauses = self._cut(available, final=True)
            return clauses

        hold = self._partial_marker_len(self._all)
        available = self._all[: len(self._all) - hold] if hold else self._all
        return self._cut(available, final=False)

    def flush(
        self,
        /,
    ) -> list[str]:
        marker_index = self._all.find(self.marker)
        available = self._all if marker_index < 0 else self._all[:marker_index]
        self._stopped = True
        return self._cut(available, final=True)

    # -- internals -------------------------------------------------------
    def _partial_marker_len(
        self,
        /,
        text: str,
    ) -> int:
        """Length of a tail that might be a marker split across delta boundaries."""
        for k in range(min(len(self.marker) - 1, len(text)), 0, -1):
            if self.marker.startswith(text[-k:]):
                return k
        return 0

    def _threshold(
        self,
        /,
    ) -> int:
        return self.first_min_chars if self._count == 0 else self.min_chars

    def _cut(
        self,
        /,
        available: str,
        final: bool,
    ) -> list[str]:
        region = available[self._emitted :]
        if not region:
            return []

        clauses: list[str] = []
        start = 0
        for index, character in enumerate(region):
            segment_length = index - start + 1
            if character in HARD_TERMINATORS:
                segment = region[start : index + 1].strip()
                if segment:
                    clauses.append(segment)
                    self._count += 1
                start = index + 1
            elif character in SOFT_TERMINATORS and segment_length >= self._threshold():
                segment = region[start : index + 1].strip()
                if segment:
                    clauses.append(segment)
                    self._count += 1
                start = index + 1
            elif segment_length >= self.max_chars:
                segment = region[start : index + 1].strip()
                if segment:
                    clauses.append(segment)
                    self._count += 1
                start = index + 1

        self._emitted += start

        if final:
            rest = available[self._emitted :].strip()
            if rest:
                clauses.append(rest)
                self._count += 1
                self._emitted = len(available)

        return [clause for clause in clauses if clause]
