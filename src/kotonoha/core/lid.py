"""언어 판별 정규화 · 폴백 · 타깃 라우팅 (§5, §10).

핵심은 폴백이다. 어떤 모델도 "네 / OK / はい" 한 마디의 언어는 맞히지 못한다.
1초 미만 발화나 저신뢰 판정에서는 직전 판정 언어를 그대로 승계하고, 그 사실을
화면에 표시한다 — 조용히 틀린 언어로 통역하는 것보다 낫다.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import LidCfg, SessionCfg

# Qwen3-ASR / whisper 가 내는 다양한 표기를 내부 코드로 정규화
_ALIASES: dict[str, str] = {
    "ko": "ko", "kor": "ko", "korean": "ko", "한국어": "ko",
    "en": "en", "eng": "en", "english": "en",
    "ja": "ja", "jpn": "ja", "japanese": "ja", "日本語": "ja",
    "zh": "zh-TW", "chinese": "zh-TW", "mandarin": "zh-TW",
    "zh-cn": "zh-TW", "zh_cn": "zh-TW", "zh-hans": "zh-TW",
    "zh-tw": "zh-TW", "zh_tw": "zh-TW", "zh-hant": "zh-TW",
    "yue": "zh-TW", "cantonese": "zh-TW",
    "中文": "zh-TW", "繁體中文": "zh-TW", "简体中文": "zh-TW",
}
# 주의: zh-CN 도 zh-TW 로 접는다. 이 기기는 대만 번체만 다룬다(§1).
# 대륙 화자의 발화도 입력으로는 받되 출력은 번체로 낸다.


def normalize_lang(raw: str | None) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower().replace("_", "-")
    if key in _ALIASES:
        return _ALIASES[key]
    # "Chinese (Taiwan)" 같은 형태
    for alias, code in _ALIASES.items():
        if alias in key:
            return code
    return None


@dataclass
class LangDecision:
    lang: str
    source: str  # lid | inherited | default
    confidence: float | None
    note: str = ""


def decide_language(
    raw_lang: str | None,
    confidence: float | None,
    duration_s: float,
    cfg: LidCfg,
    last_lang: str | None,
    allowed: list[str],
) -> LangDecision:
    norm = normalize_lang(raw_lang)
    fallback = last_lang if last_lang in allowed else (allowed[0] if allowed else "en")

    if duration_s < cfg.min_duration_s:
        return LangDecision(
            fallback,
            "inherited",
            confidence,
            f"utterance {duration_s:.2f}s < {cfg.min_duration_s}s",
        )

    if norm is None or norm not in allowed:
        return LangDecision(fallback, "inherited", confidence, f"unusable LID: {raw_lang!r}")

    if confidence is not None and confidence < cfg.min_confidence:
        return LangDecision(
            fallback, "inherited", confidence, f"confidence {confidence:.2f} < {cfg.min_confidence}"
        )

    return LangDecision(norm, "lid", confidence)


def route_targets(src_lang: str, cfg: SessionCfg) -> list[str]:
    """§9 Phase 4 — 타깃 언어 라우팅 3종."""
    if cfg.routing == "pair":
        a, b = cfg.pair
        if src_lang == a:
            return [b]
        if src_lang == b:
            return [a]
        # 페어 밖 언어가 들어오면 페어의 첫 언어로 보낸다
        return [a]
    if cfg.routing == "fixed":
        return [] if src_lang == cfg.fixed_target else [cfg.fixed_target]
    # broadcast
    return [t for t in cfg.broadcast_targets if t != src_lang]
