"""Localization for operator-facing text, on gettext.

English source strings are the message ids, following the Django and gettext
convention. Catalogs live where gettext expects them:

    src/kotonoha/locale/<language>/LC_MESSAGES/kotonoha.po
                                              kotonoha.mo

There is deliberately no English catalog. An untranslated string falls through to
its message id, which is already the English text, so a missing translation
degrades to readable English instead of a symbolic key.

`.po` is committed and is the source of truth. `.mo` is generated at install time
by the build hook in hatch_build.py and is never committed, so there is no
artifact in the repository that can fall out of step with its source. An installed
copy — including the editable install the Jetson containers use — therefore always
carries catalogs compiled from the current `.po`.

A source checkout that has not been installed has no `.mo` and falls back to
English. `kotonoha doctor` reports that; `scripts/py/i18n.py compile` fixes it.

    uv run python scripts/py/i18n.py extract    # rebuild the .pot template
    uv run python scripts/py/i18n.py update     # merge new strings into each .po
    uv run python scripts/py/i18n.py compile    # compile without reinstalling
    uv run python scripts/py/i18n.py check      # report untranslated and uncompiled

Resolution order, highest first:

    1. KOTONOHA_LANG
    2. ui.language in the configuration, when it is not "auto"
    3. LC_ALL, LC_MESSAGES, or LANG
    4. English

Typer renders command help at import time, so the locale must resolve without the
--config path, which is not known then. KOTONOHA_LANG covers forcing help text to
a specific language.
"""

from __future__ import annotations

import gettext as _gettext
import os
from functools import lru_cache
from pathlib import Path

DOMAIN = "kotonoha"
LOCALE_DIR = Path(__file__).resolve().parent / "locale"
DEFAULT_LOCALE = "en"

# Interface languages. English is the source language and has no catalog.
LOCALE_NAMES: dict[str, str] = {
    "en": "English",
    "ko": "한국어",
    "ja": "日本語",
    "zh-TW": "繁體中文",
}

# Application language codes use `zh-TW`. Gettext directories use the POSIX form,
# so the two forms are mapped explicitly.
GETTEXT_NAMES: dict[str, str] = {"en": "en", "ko": "ko", "ja": "ja", "zh-TW": "zh_TW"}

# Accepts the forms found in LANG and in configuration files.
_ALIASES: dict[str, str] = {
    "en": "en", "en_us": "en", "en_gb": "en", "c": "en", "posix": "en",
    "ko": "ko", "ko_kr": "ko", "kor": "ko",
    "ja": "ja", "jpn": "ja", "ja_jp": "ja",
    "zh-tw": "zh-TW", "zh_tw": "zh-TW", "zh-hant": "zh-TW", "zh_hant": "zh-TW",
    "zh-hk": "zh-TW", "zh_hk": "zh-TW", "zh": "zh-TW",
}

_override: str | None = None


def available_locales() -> list[str]:
    return list(LOCALE_NAMES)


def po_path(
    locale: str,
    /,
) -> Path:
    return LOCALE_DIR / GETTEXT_NAMES[locale] / "LC_MESSAGES" / f"{DOMAIN}.po"


def mo_path(
    locale: str,
    /,
) -> Path:
    return LOCALE_DIR / GETTEXT_NAMES[locale] / "LC_MESSAGES" / f"{DOMAIN}.mo"


def normalize_locale(
    raw: str | None,
    /,
) -> str | None:
    """Map a locale string to a supported code, or None when unsupported."""
    if not raw:
        return None
    key = raw.strip().split(".")[0].split("@")[0].replace("-", "_").lower()
    if key in _ALIASES:
        return _ALIASES[key]
    return _ALIASES.get(key.replace("_", "-"))


def _from_config() -> str | None:
    """Read ui.language without letting a broken configuration break --help."""
    try:
        from kotonoha._config import load_settings

        value = load_settings().ui.language
    except Exception:  # noqa: BLE001
        return None
    return None if value == "auto" else normalize_locale(value)


def _from_environment() -> str | None:
    for variable in ("LC_ALL", "LC_MESSAGES", "LANG"):
        code = normalize_locale(os.environ.get(variable))
        if code:
            return code
    return None


@lru_cache(maxsize=1)
def _detect() -> str:
    return (
        normalize_locale(os.environ.get("KOTONOHA_LANG"))
        or _from_config()
        or _from_environment()
        or DEFAULT_LOCALE
    )


def current_locale() -> str:
    return _override or _detect()


def set_locale(
    code: str | None,
    /,
) -> str:
    """Override the resolved locale for the remainder of the process.

    Used by the --lang option and by the configuration editor. Passing None
    restores automatic resolution.
    """
    global _override
    _override = normalize_locale(code) if code else None
    return current_locale()


@lru_cache(maxsize=len(LOCALE_NAMES) + 1)
def translation(
    locale: str,
    /,
) -> _gettext.NullTranslations:
    """The compiled catalog for one locale.

    English, and any locale whose .mo is absent, resolve to NullTranslations, which
    returns each message id unchanged. That is the English text, so the interface
    stays readable rather than showing symbolic keys.
    """
    if locale == DEFAULT_LOCALE:
        return _gettext.NullTranslations()
    try:
        return _gettext.translation(
            DOMAIN,
            localedir=str(LOCALE_DIR),
            languages=[GETTEXT_NAMES.get(locale, locale)],
            fallback=False,
        )
    except FileNotFoundError:
        return _gettext.NullTranslations()


def _(
    message: str,
    /,
    **format_arguments: object,
) -> str:
    """Translate, then apply str.format when arguments are given.

    gettext itself takes only the message id; formatting is a convenience so call
    sites stay on one line. Babel extracts the first string literal, so the .po
    files remain standard.

    A missing format argument returns the unformatted template rather than
    raising: a turn must not fail because a translation dropped a placeholder.
    """
    text = translation(current_locale()).gettext(message)
    if not format_arguments:
        return text
    try:
        return text.format(**format_arguments)
    except (KeyError, IndexError, ValueError):
        return text


def N_(
    message: str,
    /,
) -> str:
    """Mark a string for extraction without translating it yet.

    Tables built at import time — configuration field notes, operation labels —
    cannot call the translator, because the locale may change afterwards. They hold
    N_-marked English, and the caller applies `_` when the value is rendered.
    """
    return message


def pgettext(
    context: str,
    message: str,
    /,
    **format_arguments: object,
) -> str:
    """Disambiguate one English string that needs different translations."""
    text = translation(current_locale()).pgettext(context, message)
    if not format_arguments:
        return text
    try:
        return text.format(**format_arguments)
    except (KeyError, IndexError, ValueError):
        return text


def translate_to(
    locale: str,
    message: str,
    /,
    **format_arguments: object,
) -> str:
    """Translate into a named locale, independent of the active one. Used by tests."""
    text = translation(locale).gettext(message)
    if not format_arguments:
        return text
    try:
        return text.format(**format_arguments)
    except (KeyError, IndexError, ValueError):
        return text
