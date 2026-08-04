"""Language-ID normalisation, fallback, and target routing (§5, §10).

No model reliably identifies the language of a single "네 / OK / はい". For
utterances under one second or low-confidence verdicts, the implementation
inherits the previously detected language and displays that decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from kotonoha.config import LanguageIdentificationConfig, SessionConfig

# Normalize model-specific language labels into the application's language codes.
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


def normalize_language(raw_language: str | None) -> str | None:
    if not raw_language:
        return None
    key = raw_language.strip().lower().replace("_", "-")
    if key in _ALIASES:
        return _ALIASES[key]
    # Forms like "Chinese (Taiwan)".
    for alias, code in _ALIASES.items():
        if alias in key:
            return code
    return None


@dataclass
class LanguageDecision:
    language: str
    source: str  # lid | inherited | default
    confidence: float | None
    note: str = ""


def decide_language(
    raw_language: str | None,
    confidence: float | None,
    duration_seconds: float,
    config: LanguageIdentificationConfig,
    last_language: str | None,
    allowed_languages: list[str],
) -> LanguageDecision:
    normalized = normalize_language(raw_language)
    fallback = (
        last_language
        if last_language in allowed_languages
        else (allowed_languages[0] if allowed_languages else "en")
    )

    if duration_seconds < config.min_duration_s:
        return LanguageDecision(
            fallback,
            "inherited",
            confidence,
            f"utterance {duration_seconds:.2f}s < {config.min_duration_s}s",
        )

    if normalized is None or normalized not in allowed_languages:
        return LanguageDecision(
            fallback,
            "inherited",
            confidence,
            f"unusable LID: {raw_language!r}",
        )

    if confidence is not None and confidence < config.min_confidence:
        return LanguageDecision(
            fallback,
            "inherited",
            confidence,
            f"confidence {confidence:.2f} < {config.min_confidence}",
        )

    return LanguageDecision(normalized, "lid", confidence)


# Typed input has no audio, so there is no acoustic LID to consult. Script is a
# strong signal for these four languages and costs nothing.
_HANGUL = ((0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F))
_KANA = ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9D))
_HAN = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))


def _codepoint_in_ranges(codepoint: int, ranges) -> bool:
    return any(low <= codepoint <= high for low, high in ranges)


def detect_script(text: str) -> tuple[str | None, float]:
    """Detect the language of typed text from its script.

    Returns (language, share of decisive characters) or (None, 0.0).

    Han without kana is read as Chinese. Japanese written entirely in kanji does
    occur, but not in the conversational input this mode is for, and the operator
    can override the language explicitly.
    """
    hangul = kana = han = latin = 0
    for character in text:
        codepoint = ord(character)
        if _codepoint_in_ranges(codepoint, _HANGUL):
            hangul += 1
        elif _codepoint_in_ranges(codepoint, _KANA):
            kana += 1
        elif _codepoint_in_ranges(codepoint, _HAN):
            han += 1
        elif character.isascii() and character.isalpha():
            latin += 1

    decisive = hangul + kana + han + latin
    if not decisive:
        return None, 0.0
    if hangul:
        # Korean text may carry hanja; both count as evidence for Korean.
        return "ko", round((hangul + han) / decisive, 3)
    if kana:
        return "ja", round((kana + han) / decisive, 3)
    if han:
        return "zh-TW", round(han / decisive, 3)
    return "en", round(latin / decisive, 3)


def decide_typed_language(
    text: str,
    configured: str,
    last_language: str | None,
    allowed_languages: list[str],
    min_confidence: float = 0.5,
) -> LanguageDecision:
    """Language for a typed utterance.

    An explicit setting wins. Otherwise the script decides, and when the script is
    inconclusive the previous language is inherited, exactly as §5 does for a short
    spoken utterance.
    """
    fallback = (
        last_language
        if last_language in allowed_languages
        else (allowed_languages[0] if allowed_languages else "en")
    )

    if configured != "auto":
        if configured in allowed_languages:
            return LanguageDecision(configured, "forced", None)
        return LanguageDecision(fallback, "inherited", None, f"unusable setting: {configured!r}")

    language, confidence = detect_script(text)
    if language is None or language not in allowed_languages:
        return LanguageDecision(fallback, "inherited", confidence, "no decisive script")
    if confidence < min_confidence:
        return LanguageDecision(
            fallback, "inherited", confidence, f"script share {confidence:.2f} < {min_confidence}"
        )
    return LanguageDecision(language, "script", confidence)


def route_targets(source_language: str, config: SessionConfig) -> list[str]:
    """The three target-routing modes (§9, Phase 4)."""
    if config.routing == "pair":
        first_language, second_language = config.pair
        if source_language == first_language:
            return [second_language]
        if source_language == second_language:
            return [first_language]
        # A language outside the pair goes to the pair's first language.
        return [first_language]
    if config.routing == "fixed":
        return (
            []
            if source_language == config.fixed_target
            else [config.fixed_target]
        )
    # broadcast
    return [
        target
        for target in config.broadcast_targets
        if target != source_language
    ]
