from __future__ import annotations

from kotonoha._config import LanguageIdentificationConfig, SessionConfig, load_settings
from kotonoha.core._lid import decide_language, normalize_language, route_targets
from kotonoha.core._quality import character_error_rate, is_divergent, should_cross_verify

ALLOWED = ["ko", "en", "zh-TW", "ja"]


def test_normalize_lang_folds_chinese_variants_to_traditional() -> None:
    for raw in ("zh", "zh-CN", "Chinese", "Mandarin", "简体中文", "zh_TW"):
        assert normalize_language(raw) == "zh-TW"
    assert normalize_language("Korean") == "ko"
    assert normalize_language(None) is None
    assert normalize_language("Klingon") is None


def test_short_utterance_inherits_previous_language() -> None:
    """No model can call the language of "네 / OK / はい" (§5)."""
    decision = decide_language(
        "English",
        0.9,
        duration_seconds=0.4,
        config=LanguageIdentificationConfig(),
        last_language="ko",
        allowed_languages=ALLOWED,
    )
    assert decision.language == "ko"
    assert decision.source == "inherited"
    assert "0.40s" in decision.note


def test_low_confidence_inherits() -> None:
    decision = decide_language(
        "Japanese",
        0.3,
        2.0,
        LanguageIdentificationConfig(min_confidence=0.6),
        "en",
        ALLOWED,
    )
    assert (decision.language, decision.source) == ("en", "inherited")


def test_confident_long_utterance_uses_lid() -> None:
    decision = decide_language(
        "Japanese",
        0.95,
        2.0,
        LanguageIdentificationConfig(),
        "en",
        ALLOWED,
    )
    assert (decision.language, decision.source) == ("ja", "lid")


def test_routing_modes() -> None:
    pair = SessionConfig(routing="pair", pair=["ko", "en"])
    assert route_targets("ko", pair) == ["en"]
    assert route_targets("en", pair) == ["ko"]

    fixed = SessionConfig(routing="fixed", fixed_target="en")
    assert route_targets("ja", fixed) == ["en"]
    assert route_targets("en", fixed) == []  # never interpret into the source language

    bc = SessionConfig(routing="broadcast", broadcast_targets=ALLOWED)
    assert route_targets("ko", bc) == ["en", "zh-TW", "ja"]


def test_cross_verify_only_fires_when_needed() -> None:
    fire, _ = should_cross_verify(
        -0.2,
        -0.55,
        ["안녕하세요 반갑습니다"],
        duration_seconds=3.0,
    )
    assert not fire, "must not add 0.8 s per turn to a high-confidence utterance"

    fire, why = should_cross_verify(-0.9, -0.55, ["안녕하세요"], duration_seconds=3.0)
    assert fire and "avg_logprob" in why


def test_cross_verify_fires_on_nbest_disagreement() -> None:
    fire, why = should_cross_verify(
        -0.1,
        -0.55,
        ["오늘 회의는 세 시입니다", "완전히 다른 문장이 나왔습니다요"],
        duration_seconds=3.0,
    )
    assert fire and "disagreement" in why


def test_cer_is_punctuation_and_space_insensitive() -> None:
    assert character_error_rate("오늘 회의는 세 시입니다.", "오늘회의는세시입니다") == 0.0
    assert is_divergent("소프트웨어 정보", "완전히 다른 말", 0.25)


def test_default_config_loads_and_is_consistent() -> None:
    s = load_settings()
    assert 200 <= s.frontend.vad.preroll_ms <= 500  # §5.1
    assert s.asr.n_best == 5  # §5.2
    assert s.session.mode == "push_to_talk"  # §4
    assert (s.llm.model_path / "config.json").name == "config.json"
    assert s.llm.active.quantization == "awq"
    # The §6 stage budgets must add up to the stated total.
    b = s.budget_ms
    assert (
        b.silence + b.frontend + b.asr + b.verify + b.llm_first_clause + b.tts_first_packet
        == b.total
    )
