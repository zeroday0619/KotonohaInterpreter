"""Localization: catalog integrity, locale resolution, and the configuration editor."""

from __future__ import annotations

import importlib.util
import string
from collections import Counter
from typing import Any

import pytest
import yaml

from kotonoha import _i18n as i18n
from kotonoha._config import REPO_ROOT, load_settings
from kotonoha._config_store import apply_changes, get_path, set_path, validate_candidate
from kotonoha._i18n import (
    DEFAULT_LOCALE,
    _,
    available_locales,
    mo_path,
    normalize_locale,
    po_path,
    set_locale,
    translate_to,
    translation,
)
from kotonoha.tui._config_app import (
    FIELD_DESCRIPTIONS,
    FIELDS,
    SECTION_LABELS,
    SECTIONS,
    VALUE_KIND_DESCRIPTIONS,
    effective_value,
    field_description,
)

# Babel is a development dependency, so catalog checks skip where it is absent.
babel_pofile = pytest.importorskip("babel.messages.pofile")

TRANSLATED_LOCALES = [code for code in available_locales() if code != DEFAULT_LOCALE]
PROBE = "Consecutive four-language offline speech interpreter"


@pytest.fixture(autouse=True)
def _reset_locale() -> Any:
    yield
    set_locale(None)


def read_catalog(
    path: Any,
    /,
) -> Any:
    with path.open("rb") as handle:
        return babel_pofile.read_po(handle)


@pytest.fixture(scope="module")
def i18n_tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "kotonoha_i18n_tool", REPO_ROOT / "scripts" / "i18n.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def template(
    _positional_only: object | None = None,
    /,
    *,
    i18n_tool: Any,
) -> Any:
    """Rebuild message ids so a stale committed template cannot hide source drift."""
    return i18n_tool.build_template()


# -- catalog integrity ------------------------------------------------------
def test_english_has_no_catalog() -> None:
    """English message ids are the source strings, so a catalog would be redundant."""
    assert not po_path(DEFAULT_LOCALE).exists()
    assert translation(DEFAULT_LOCALE).gettext("Quit") == "Quit"


@pytest.mark.parametrize("locale", TRANSLATED_LOCALES)
def test_every_extracted_message_is_translated(
    _positional_only: object | None = None,
    /,
    *,
    locale: Any,
    template: Any,
) -> None:
    """A string added to the source without a translation must fail here.

    Without this, the omission only shows up as an English fragment inside an
    otherwise translated screen.
    """
    catalog = read_catalog(po_path(locale))
    extracted = {entry.id for entry in template if entry.id}
    present = {entry.id for entry in catalog if entry.id}

    missing = sorted(extracted - present)
    obsolete = sorted(present - extracted)
    untranslated = sorted(entry.id for entry in catalog if entry.id and not entry.string)

    assert not missing, f"{locale}: absent from catalog: {missing[:5]}"
    assert not obsolete, f"{locale}: no longer in source: {obsolete[:5]}"
    assert not untranslated, f"{locale}: untranslated: {untranslated[:5]}"


@pytest.mark.parametrize("locale", TRANSLATED_LOCALES)
def test_format_placeholders_match_the_message_id(
    _positional_only: object | None = None,
    /,
    *,
    locale: Any,
) -> None:
    """Names, repetition, conversions, and format specifications must remain exact."""
    parse = string.Formatter().parse
    for entry in read_catalog(po_path(locale)):
        if not entry.id or not entry.string:
            continue
        expected = Counter(
            (name, specification, conversion)
            for _text, name, specification, conversion in parse(entry.id)
            if name is not None
        )
        actual = Counter(
            (name, specification, conversion)
            for _text, name, specification, conversion in parse(entry.string)
            if name is not None
        )
        assert actual == expected, f"{locale}: {entry.id!r} has {actual}, expected {expected}"


@pytest.mark.parametrize("locale", TRANSLATED_LOCALES)
def test_catalog_preserves_runtime_sensitive_text(
    _positional_only: object | None = None,
    /,
    *,
    locale: Any,
    i18n_tool: Any,
) -> None:
    violations = [
        (entry.id, violation)
        for entry in read_catalog(po_path(locale))
        if entry.id and isinstance(entry.id, str) and isinstance(entry.string, str)
        for violation in i18n_tool.translation_violations(
            entry.id,
            entry.string,
            locale=locale,
        )
    ]
    assert not violations, f"{locale}: {violations[:5]}"


@pytest.mark.parametrize("locale", TRANSLATED_LOCALES)
def test_no_fuzzy_entries(
    _positional_only: object | None = None,
    /,
    *,
    locale: Any,
) -> None:
    """Compilation drops fuzzy entries, so one would silently show English."""
    fuzzy = [entry.id for entry in read_catalog(po_path(locale)) if entry.id and entry.fuzzy]
    assert not fuzzy, f"{locale} has fuzzy entries: {fuzzy[:5]}"


@pytest.mark.parametrize("locale", TRANSLATED_LOCALES)
def test_compiled_catalog_matches_its_source(
    _positional_only: object | None = None,
    /,
    *,
    locale: Any,
) -> None:
    """.mo is generated and committed, so it must not drift from its .po.

    A stale .mo serves yesterday's text with no other symptom.
    """
    assert mo_path(locale).exists(), f"{locale}: run scripts/i18n.py compile"
    catalog = translation(locale)
    stale = [
        entry.id
        for entry in read_catalog(po_path(locale))
        if entry.id
        and entry.string
        and isinstance(entry.id, str)
        and catalog.gettext(entry.id) != entry.string
    ]
    assert not stale, f"{locale}: .mo is stale for {stale[:5]}; run scripts/i18n.py compile"


@pytest.mark.parametrize("locale", TRANSLATED_LOCALES)
def test_translations_are_not_blank(
    _positional_only: object | None = None,
    /,
    *,
    locale: Any,
) -> None:
    blank = [
        entry.id
        for entry in read_catalog(po_path(locale))
        if entry.id and entry.string and not str(entry.string).strip()
    ]
    assert not blank, f"{locale} has blank translations: {blank}"


def test_import_time_tables_are_extracted(
    _positional_only: object | None = None,
    /,
    *,
    template: Any,
) -> None:
    """N_ tables are invisible to a naive extractor; confirm the keyword is configured."""
    extracted = {entry.id for entry in template if entry.id}
    for text in (
        *FIELD_DESCRIPTIONS.values(),
        *SECTION_LABELS.values(),
        *VALUE_KIND_DESCRIPTIONS.values(),
    ):
        assert text in extracted, f"not extracted: {text[:60]}"


def test_every_editable_field_has_a_description() -> None:
    for spec in FIELDS:
        assert field_description(spec).strip(), f"no description for {spec.path}"


def test_every_field_section_is_rendered() -> None:
    assert {spec.section for spec in FIELDS} <= set(SECTIONS)
    assert set(SECTION_LABELS) == set(SECTIONS)


def test_editor_exposes_the_complete_settings_schema() -> None:
    paths = {spec.path for spec in FIELDS}
    assert len(paths) == len(FIELDS)
    assert len(paths) >= 100
    for required in (
        "session.broadcast_targets",
        "audio.capture_sample_rate",
        "frontend.vad.max_utterance_ms",
        "shm.slot_seconds",
        "services.asr",
        "placement",
        "remote.verify_tls",
        "asr.lid.min_confidence",
        "asr_verify.compute_type",
        "llm.profiles",
        "tts.voices",
        "zh.apply_to",
        "logging.turn_log_path",
        "budget_ms.total",
    ):
        assert required in paths
    assert "root" not in paths


# -- locale resolution ------------------------------------------------------
def test_normalize_locale_accepts_environment_forms() -> None:
    assert normalize_locale("ko_KR.UTF-8") == "ko"
    assert normalize_locale("ja_JP") == "ja"
    assert normalize_locale("zh_TW.UTF-8") == "zh-TW"
    assert normalize_locale("zh-Hant") == "zh-TW"
    assert normalize_locale("en_GB") == "en"
    assert normalize_locale("C") == "en"
    assert normalize_locale("de_DE") is None
    assert normalize_locale(None) is None


def test_default_locale_is_english() -> None:
    assert DEFAULT_LOCALE == "en"
    set_locale(None)
    i18n._detect.cache_clear()


def test_set_locale_switches_the_catalog() -> None:
    set_locale("ja")
    assert _(PROBE) == translate_to("ja", PROBE)
    set_locale("zh-TW")
    assert _(PROBE) == translate_to("zh-TW", PROBE)
    assert translate_to("ja", PROBE) != translate_to("zh-TW", PROBE)


def test_untranslated_message_falls_through_to_english() -> None:
    """The fallback is the message id, which is already the English text."""
    set_locale("ko")
    assert _("A string that is not in any catalog") == "A string that is not in any catalog"


def test_missing_format_argument_returns_the_template() -> None:
    set_locale("en")
    assert "{path}" in _("Turn log: {path}")


def test_formatting_applies_in_every_locale() -> None:
    for locale in available_locales():
        set_locale(locale)
        assert "/tmp/x.jsonl" in _("Turn log: {path}", path="/tmp/x.jsonl")


# -- configuration editor ---------------------------------------------------
def test_dotted_path_helpers() -> None:
    data: dict = {}
    set_path(data, "remote.services.llm", "http://x")
    assert data == {"remote": {"services": {"llm": "http://x"}}}
    assert get_path(data, "remote.services.llm") == "http://x"
    assert get_path(data, "remote.services.missing") is None
    assert get_path(data, "nothing.here") is None


def test_effective_value_reads_through_the_model() -> None:
    s = load_settings()
    assert effective_value(s, "frontend.vad.preroll_ms") == s.frontend.vad.preroll_ms
    assert effective_value(s, "remote.services.llm") == s.remote.services.llm


def test_valid_candidate_is_accepted() -> None:
    assert validate_candidate(None, {"perf_mode": "hybrid"}) is None


def test_invalid_candidate_is_rejected_with_a_reason() -> None:
    problem = validate_candidate(None, {"perf_mode": "turbo"})
    assert problem is not None and "perf_mode" in problem


def test_preroll_below_the_floor_is_rejected() -> None:
    """§5.1 is enforced by the model, so the editor cannot write past it."""
    problem = validate_candidate(None, {"frontend": {"vad": {"preroll_ms": 50}}})
    assert problem is not None and "preroll_ms" in problem


def test_apply_changes_writes_local_yaml(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    target = tmp_path / "local.yaml"
    result = apply_changes(
        {"perf_mode": "hybrid", "remote.enabled": True, "ui.language": "ja"},
        local_path=target,
    )
    assert result.written
    assert result.changed == ["perf_mode", "remote.enabled", "ui.language"]

    written = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert written == {"perf_mode": "hybrid", "remote": {"enabled": True}, "ui": {"language": "ja"}}


def test_apply_changes_refuses_to_write_an_invalid_configuration(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    """An unloadable configuration on a device is worse than a rejected edit."""
    target = tmp_path / "local.yaml"
    result = apply_changes({"asr": {"n_best": "many"}}, local_path=target)
    assert not result.written
    assert result.error is not None
    assert not target.exists()


def test_apply_changes_preserves_unrelated_existing_values(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    target = tmp_path / "local.yaml"
    target.write_text(yaml.safe_dump({"audio": {"input_device": 3}}), encoding="utf-8")
    assert apply_changes({"perf_mode": "hybrid"}, local_path=target).written
    written = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert written["audio"]["input_device"] == 3
    assert written["perf_mode"] == "hybrid"
