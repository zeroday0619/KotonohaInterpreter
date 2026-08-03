"""Language-ID normalisation, fallback, and target routing (§5, §10).

The fallback is the important part. No model can tell you the language of a
single "네 / OK / はい". For utterances under a second, or low-confidence
verdicts, we inherit the previously detected language and show that on screen —
better than quietly interpreting from the wrong language.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import LidCfg, SessionCfg

# Normalise the various spellings Qwen3-ASR and whisper emit into our codes.
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
# Note: zh-CN folds into zh-TW as well. This device only deals in Taiwan
# Traditional (§1). Mainland speech is accepted as input, but output is Traditional.


def normalize_lang(raw: str | None) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower().replace("_", "-")
    if key in _ALIASES:
        return _ALIASES[key]
    # Forms like "Chinese (Taiwan)".
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
    """The three target-routing modes (§9, Phase 4)."""
    if cfg.routing == "pair":
        a, b = cfg.pair
        if src_lang == a:
            return [b]
        if src_lang == b:
            return [a]
        # A language outside the pair goes to the pair's first language.
        return [a]
    if cfg.routing == "fixed":
        return [] if src_lang == cfg.fixed_target else [cfg.fixed_target]
    # broadcast
    return [t for t in cfg.broadcast_targets if t != src_lang]
