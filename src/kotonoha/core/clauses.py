"""절 단위 스트리밍 분할 (§5.4).

LLM 출력을 끝까지 기다리지 않고 절이 완성되는 즉시 TTS 큐에 넣는다.
성립 조건은 명세대로 LLM 이 5 tok/s 이상일 때다. 음성 1초에 약 4~5토큰이
필요하므로, 그 아래면 첫 절을 재생하는 동안 다음 절이 준비되지 못해 끊긴다.

두 가지를 조심해야 한다.
  · 너무 잘게 자르면 TTS 가 문말 억양을 매 조각마다 붙여 어색해진다.
  · 너무 크게 자르면 첫 음성이 늦는다. 그래서 첫 절만 짧게 허용하고
    이후는 더 크게 모은다.

그리고 ⟦SRC⟧ 마커 뒤(정정된 원문)는 절대 TTS 로 보내면 안 된다. 스트리밍
델타 경계에서 마커가 쪼개져 도착하는 경우까지 처리한다.
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

    # ── 상태 ────────────────────────────────────────────────────────────
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

    # ── 스트리밍 ────────────────────────────────────────────────────────
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

    # ── 내부 ────────────────────────────────────────────────────────────
    def _partial_marker_len(self, text: str) -> int:
        """마커가 델타 경계에 걸쳐 쪼개졌을 가능성이 있는 꼬리 길이."""
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
