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

from ..prompts.translate import SRC_MARKER

HARD_TERMINATORS = "。．.!?！？…‥\n"
SOFT_TERMINATORS = "、，,;；:：）)】」』"


class ClauseStreamer:
    def __init__(
        self,
        first_min_chars: int = 6,
        min_chars: int = 14,
        max_chars: int = 90,
        marker: str = SRC_MARKER,
    ):
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
    def stopped(self) -> bool:
        return self._stopped

    @property
    def raw(self) -> str:
        return self._all

    @property
    def translation(self) -> str:
        idx = self._all.find(self.marker)
        return (self._all if idx < 0 else self._all[:idx]).strip()

    @property
    def clause_count(self) -> int:
        return self._count

    # -- streaming -------------------------------------------------------
    def push(self, delta: str) -> list[str]:
        self._all += delta
        if self._stopped:
            return []

        idx = self._all.find(self.marker)
        if idx >= 0:
            self._stopped = True
            avail = self._all[:idx]
            out = self._cut(avail, final=True)
            return out

        hold = self._partial_marker_len(self._all)
        avail = self._all[: len(self._all) - hold] if hold else self._all
        return self._cut(avail, final=False)

    def flush(self) -> list[str]:
        idx = self._all.find(self.marker)
        avail = self._all if idx < 0 else self._all[:idx]
        self._stopped = True
        return self._cut(avail, final=True)

    # -- internals -------------------------------------------------------
    def _partial_marker_len(self, text: str) -> int:
        """Length of a tail that might be a marker split across delta boundaries."""
        for k in range(min(len(self.marker) - 1, len(text)), 0, -1):
            if self.marker.startswith(text[-k:]):
                return k
        return 0

    def _threshold(self) -> int:
        return self.first_min_chars if self._count == 0 else self.min_chars

    def _cut(self, avail: str, final: bool) -> list[str]:
        region = avail[self._emitted :]
        if not region:
            return []

        out: list[str] = []
        start = 0
        for i, ch in enumerate(region):
            seg_len = i - start + 1
            if ch in HARD_TERMINATORS:
                seg = region[start : i + 1].strip()
                if seg:
                    out.append(seg)
                    self._count += 1
                start = i + 1
            elif ch in SOFT_TERMINATORS and seg_len >= self._threshold():
                seg = region[start : i + 1].strip()
                if seg:
                    out.append(seg)
                    self._count += 1
                start = i + 1
            elif seg_len >= self.max_chars and ch in " 　":
                seg = region[start : i + 1].strip()
                if seg:
                    out.append(seg)
                    self._count += 1
                start = i + 1

        self._emitted += start

        if final:
            rest = avail[self._emitted :].strip()
            if rest:
                out.append(rest)
                self._count += 1
                self._emitted = len(avail)

        return [s for s in out if s]
