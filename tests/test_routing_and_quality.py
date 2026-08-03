from __future__ import annotations

from kotonoha.config import LidCfg, SessionCfg, load_settings
from kotonoha.core.lid import decide_language, normalize_lang, route_targets
from kotonoha.core.quality import cer, is_divergent, should_cross_verify

ALLOWED = ["ko", "en", "zh-TW", "ja"]


def test_normalize_lang_folds_chinese_variants_to_traditional():
    for raw in ("zh", "zh-CN", "Chinese", "Mandarin", "简体中文", "zh_TW"):
        assert normalize_lang(raw) == "zh-TW"
    assert normalize_lang("Korean") == "ko"
    assert normalize_lang(None) is None
    assert normalize_lang("Klingon") is None


def test_short_utterance_inherits_previous_language():
    """'네 / OK / はい' 는 어떤 모델도 못 맞힌다(§5)."""
    d = decide_language(
        "English", 0.9, duration_s=0.4, cfg=LidCfg(), last_lang="ko", allowed=ALLOWED
    )
    assert d.lang == "ko"
    assert d.source == "inherited"
    assert "0.40s" in d.note


def test_low_confidence_inherits():
    d = decide_language("Japanese", 0.3, 2.0, LidCfg(min_confidence=0.6), "en", ALLOWED)
    assert (d.lang, d.source) == ("en", "inherited")


def test_confident_long_utterance_uses_lid():
    d = decide_language("Japanese", 0.95, 2.0, LidCfg(), "en", ALLOWED)
    assert (d.lang, d.source) == ("ja", "lid")


def test_routing_modes():
    pair = SessionCfg(routing="pair", pair=["ko", "en"])
    assert route_targets("ko", pair) == ["en"]
    assert route_targets("en", pair) == ["ko"]

    fixed = SessionCfg(routing="fixed", fixed_target="en")
    assert route_targets("ja", fixed) == ["en"]
    assert route_targets("en", fixed) == []  # 자기 자신으로는 통역하지 않는다

    bc = SessionCfg(routing="broadcast", broadcast_targets=ALLOWED)
    assert route_targets("ko", bc) == ["en", "zh-TW", "ja"]


def test_cross_verify_only_fires_when_needed():
    fire, _ = should_cross_verify(-0.2, -0.55, ["안녕하세요 반갑습니다"], duration_s=3.0)
    assert not fire, "고신뢰 발화에 매 턴 0.8초를 붙이면 안 된다"

    fire, why = should_cross_verify(-0.9, -0.55, ["안녕하세요"], duration_s=3.0)
    assert fire and "avg_logprob" in why


def test_cross_verify_fires_on_nbest_disagreement():
    fire, why = should_cross_verify(
        -0.1, -0.55, ["오늘 회의는 세 시입니다", "완전히 다른 문장이 나왔습니다요"], duration_s=3.0
    )
    assert fire and "disagreement" in why


def test_cer_is_punctuation_and_space_insensitive():
    assert cer("오늘 회의는 세 시입니다.", "오늘회의는세시입니다") == 0.0
    assert is_divergent("소프트웨어 정보", "완전히 다른 말", 0.25)


def test_default_config_loads_and_is_consistent():
    s = load_settings()
    assert 200 <= s.frontend.vad.preroll_ms <= 500  # §5.1
    assert s.asr.n_best == 5  # §5.2
    assert s.session.mode == "push_to_talk"  # §4
    assert s.llm.gguf_path.name.endswith(".gguf")
    # §6 합계가 명세와 맞는지
    b = s.budget_ms
    assert (
        b.silence + b.frontend + b.asr + b.verify + b.llm_first_clause + b.tts_first_packet
        == b.total
    )
