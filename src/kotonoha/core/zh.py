"""Traditional Chinese post-processing (§5).

ASR can hear Taiwanese Mandarin and still emit Simplified. So can the
translation LLM. OpenCC `s2twp` is therefore applied at both points. s2twp
converts glyphs and swaps in Taiwanese vocabulary, but it is not exhaustive, so
the zh_rules table in the database gets one final pass.
"""

from __future__ import annotations

import re

from ..logging_setup import get_logger

log = get_logger(__name__)


class TraditionalizeTW:
    def __init__(
        self,
        opencc_config: str = "s2twp",
        extra_rules: list[tuple[str, str, bool]] | None = None,
    ):
        self._cc = None
        self._config = opencc_config
        try:
            from opencc import OpenCC  # type: ignore[import-not-found]

            self._cc = OpenCC(opencc_config)
        except Exception as e:  # noqa: BLE001
            log.error("opencc.unavailable", error=repr(e), config=opencc_config)
        self.set_rules(extra_rules or [])

    def set_rules(self, rules: list[tuple[str, str, bool]]) -> None:
        self._plain: list[tuple[str, str]] = []
        self._regex: list[tuple[re.Pattern[str], str]] = []
        for pattern, repl, is_regex in rules:
            if is_regex:
                self._regex.append((re.compile(pattern), repl))
            else:
                self._plain.append((pattern, repl))
        # Longest first, so a partial match cannot break a longer replacement.
        self._plain.sort(key=lambda t: len(t[0]), reverse=True)

    @property
    def available(self) -> bool:
        return self._cc is not None

    def __call__(self, text: str) -> str:
        if not text:
            return text
        out = self._cc.convert(text) if self._cc is not None else text
        for pat, repl in self._plain:
            out = out.replace(pat, repl)
        for rx, repl in self._regex:
            out = rx.sub(repl, out)
        return out


_SIMPLIFIED_HINTS = "软视频信息鼠标网络program这么说话时间点击应该图书馆"


def looks_simplified(text: str) -> bool:
    """Rough check for leaked Simplified characters. For logs and TUI warnings,
    not for making decisions."""
    return any(ch in text for ch in _SIMPLIFIED_HINTS)
