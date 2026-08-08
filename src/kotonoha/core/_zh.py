"""Traditional Chinese post-processing (§5).

ASR can hear Taiwanese Mandarin and still emit Simplified. So can the
translation LLM. OpenCC `s2twp` is therefore applied at both points. s2twp
converts glyphs and swaps in Taiwanese vocabulary, but it is not exhaustive, so
the zh_rules table in the database gets one final pass.
"""

from __future__ import annotations

from typing import Any, ClassVar

from kotonoha._logging_setup import get_logger
from kotonoha._typing import override

log = get_logger(__name__)


class TraditionalChineseConverter:
    __slots__: ClassVar[tuple[str, ...]] = (
        "_config",
        "_converter",
        "_plain_rules",
    )
    _converter: Any | None
    _config: str
    _plain_rules: list[tuple[str, str]]

    @override
    def __init__(
        self,
        /,
        opencc_config: str = "s2twp",
        extra_rules: list[tuple[str, str, bool]] | None = None,
    ) -> None:
        self._converter = None
        self._config = opencc_config
        try:
            from opencc import OpenCC  # type: ignore[import-not-found]

            self._converter = OpenCC(opencc_config)
        except Exception as error:  # noqa: BLE001
            log.error("opencc.unavailable", error=repr(error), config=opencc_config)
        self.set_rules(extra_rules or [])

    def set_rules(
        self,
        /,
        rules: list[tuple[str, str, bool]],
    ) -> None:
        self._plain_rules = []
        for pattern, replacement, is_regex in rules:
            if is_regex:
                # The stdlib engine has no execution timeout. A persisted nested
                # repetition could otherwise block the complete interpreter turn.
                log.warning("zh.regex_rule_disabled", pattern_length=len(pattern))
                continue
            self._plain_rules.append((pattern, replacement))
        # Longest first, so a partial match cannot break a longer replacement.
        self._plain_rules.sort(key=lambda rule: len(rule[0]), reverse=True)

    @property
    def available(
        self,
        /,
    ) -> bool:
        return self._converter is not None

    def __call__(
        self,
        /,
        text: str,
    ) -> str:
        if not text:
            return text
        output = self._converter.convert(text) if self._converter is not None else text
        for pattern, replacement in self._plain_rules:
            output = output.replace(pattern, replacement)
        return output


_SIMPLIFIED_HINTS = "软视频信息鼠标网络program这么说话时间点击应该图书馆"


def looks_simplified(
    text: str,
    /,
) -> bool:
    """Rough check for leaked Simplified characters. For logs and TUI warnings,
    not for making decisions."""
    return any(character in text for character in _SIMPLIFIED_HINTS)
