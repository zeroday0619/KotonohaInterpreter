"""Localization: catalog integrity, locale resolution, and the configuration editor."""

from __future__ import annotations

import string

import pytest
import yaml

from kotonoha import i18n
from kotonoha.config import load_settings
from kotonoha.i18n import CATALOGS, DEFAULT_LOCALE, normalize_locale, set_locale, t
from kotonoha.tui.config_app import (
    FIELDS,
    SECTIONS,
    apply_changes,
    effective_value,
    get_path,
    set_path,
    validate_candidate,
)

REFERENCE = CATALOGS[DEFAULT_LOCALE]


@pytest.fixture(autouse=True)
def _reset_locale():
    yield
    set_locale(None)


# -- catalog integrity ------------------------------------------------------
@pytest.mark.parametrize("locale", [c for c in CATALOGS if c != DEFAULT_LOCALE])
def test_catalogs_have_the_same_keys_as_english(locale):
    """A string added to English without a translation must fail here.

    Without this, the omission only shows up as an English fragment inside an
    otherwise translated screen.
    """
    missing = sorted(set(REFERENCE) - set(CATALOGS[locale]))
    extra = sorted(set(CATALOGS[locale]) - set(REFERENCE))
    assert not missing, f"{locale} is missing: {missing}"
    assert not extra, f"{locale} has keys not in {DEFAULT_LOCALE}: {extra}"


@pytest.mark.parametrize("locale", list(CATALOGS))
def test_format_placeholders_match_the_reference(locale):
    """A translation that drops or renames a placeholder would format incorrectly."""
    for key, reference in REFERENCE.items():
        expected = {f for _, f, _, _ in string.Formatter().parse(reference) if f}
        actual = {f for _, f, _, _ in string.Formatter().parse(CATALOGS[locale][key]) if f}
        assert actual == expected, f"{locale}:{key} placeholders {actual} != {expected}"


@pytest.mark.parametrize("locale", list(CATALOGS))
def test_no_empty_messages(locale):
    empty = [k for k, v in CATALOGS[locale].items() if not v.strip()]
    assert not empty, f"{locale} has empty messages: {empty}"


def test_every_editable_field_has_a_description():
    for spec in FIELDS:
        key = f"cfg.f.{spec.path}"
        assert key in REFERENCE, f"no description for {spec.path}"


def test_every_field_section_is_rendered():
    assert {spec.section for spec in FIELDS} <= set(SECTIONS)


# -- locale resolution ------------------------------------------------------
def test_normalize_locale_accepts_environment_forms():
    assert normalize_locale("ko_KR.UTF-8") == "ko"
    assert normalize_locale("ja_JP") == "ja"
    assert normalize_locale("zh_TW.UTF-8") == "zh-TW"
    assert normalize_locale("zh-Hant") == "zh-TW"
    assert normalize_locale("en_GB") == "en"
    assert normalize_locale("C") == "en"
    assert normalize_locale("de_DE") is None
    assert normalize_locale(None) is None


def test_default_locale_is_english():
    assert DEFAULT_LOCALE == "en"
    set_locale(None)
    i18n._detect.cache_clear()


def test_set_locale_switches_the_catalog():
    set_locale("ja")
    assert t("cli.app.help") == CATALOGS["ja"]["cli.app.help"]
    set_locale("zh-TW")
    assert t("cli.app.help") == CATALOGS["zh-TW"]["cli.app.help"]


def test_unknown_key_returns_the_key():
    """A missing string must be visible, not raise during a turn."""
    assert t("cli.definitely.not.a.key") == "cli.definitely.not.a.key"


def test_missing_format_argument_returns_the_template():
    set_locale("en")
    assert "{path}" in t("cli.replay.turn_log")


def test_formatting_applies_in_every_locale():
    for locale in CATALOGS:
        set_locale(locale)
        assert "/tmp/x.jsonl" in t("cli.replay.turn_log", path="/tmp/x.jsonl")


# -- configuration editor ---------------------------------------------------
def test_dotted_path_helpers():
    data: dict = {}
    set_path(data, "remote.services.llm", "http://x")
    assert data == {"remote": {"services": {"llm": "http://x"}}}
    assert get_path(data, "remote.services.llm") == "http://x"
    assert get_path(data, "remote.services.missing") is None
    assert get_path(data, "nothing.here") is None


def test_effective_value_reads_through_the_model():
    s = load_settings()
    assert effective_value(s, "frontend.vad.preroll_ms") == s.frontend.vad.preroll_ms
    assert effective_value(s, "remote.services.llm") == s.remote.services.llm


def test_valid_candidate_is_accepted():
    assert validate_candidate(None, {"perf_mode": "hybrid"}) is None


def test_invalid_candidate_is_rejected_with_a_reason():
    problem = validate_candidate(None, {"perf_mode": "turbo"})
    assert problem is not None and "perf_mode" in problem


def test_preroll_below_the_floor_is_rejected():
    """§5.1 is enforced by the model, so the editor cannot write past it."""
    problem = validate_candidate(None, {"frontend": {"vad": {"preroll_ms": 50}}})
    assert problem is not None and "preroll_ms" in problem


def test_apply_changes_writes_local_yaml(tmp_path):
    target = tmp_path / "local.yaml"
    result = apply_changes(
        {"perf_mode": "hybrid", "remote.enabled": True, "ui.language": "ja"},
        local_path=target,
    )
    assert result.written
    assert result.changed == ["perf_mode", "remote.enabled", "ui.language"]

    written = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert written == {"perf_mode": "hybrid", "remote": {"enabled": True}, "ui": {"language": "ja"}}


def test_apply_changes_refuses_to_write_an_invalid_configuration(tmp_path):
    """An unloadable configuration on a device is worse than a rejected edit."""
    target = tmp_path / "local.yaml"
    result = apply_changes({"asr": {"n_best": "many"}}, local_path=target)
    assert not result.written
    assert result.error is not None
    assert not target.exists()


def test_apply_changes_preserves_unrelated_existing_values(tmp_path):
    target = tmp_path / "local.yaml"
    target.write_text(yaml.safe_dump({"audio": {"input_device": 3}}), encoding="utf-8")
    assert apply_changes({"perf_mode": "hybrid"}, local_path=target).written
    written = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert written["audio"]["input_device"] == 3
    assert written["perf_mode"] == "hybrid"
