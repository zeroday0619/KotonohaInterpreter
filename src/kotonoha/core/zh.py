"""번체 중문 후처리 (§5).

ASR 은 대만어 발화를 듣고도 간체를 뱉을 수 있다. 번역 LLM 도 마찬가지다.
그래서 두 지점 모두에 OpenCC `s2twp` 를 건다. s2twp 는 자형 변환에 더해
대만 관용 어휘까지 바꿔주지만 완전하지 않으므로, DB 의 zh_rules 로 마지막에
한 번 더 치환한다.
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
        # 긴 것부터 치환해야 부분 매치로 깨지지 않는다
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
    """간체가 섞여 나왔는지 대충 본다. 로그·TUI 경고용이지 판정용이 아니다."""
    return any(ch in text for ch in _SIMPLIFIED_HINTS)
