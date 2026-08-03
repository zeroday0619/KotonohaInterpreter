"""Localization for operator-facing text.

English is the default and the reference catalog. Korean, Japanese and Traditional
Chinese are translations of it.

Message catalogs are plain dictionaries rather than gettext. The string count is small,
the catalogs are typed source files that ruff and the test suite already inspect, and
there is no compilation step to keep in sync with the build.

Resolution order, highest first:

    1. KOTONOHA_LANG
    2. ui.language in the configuration, when it is not "auto"
    3. LC_ALL, LC_MESSAGES, or LANG
    4. English

Locale is resolved once and cached. Typer renders command help at import time, so the
resolution must not depend on the --config path, which is not known then. KOTONOHA_LANG
covers the case where help text must be forced to a specific language.
"""

from __future__ import annotations

import os
from functools import lru_cache

from .locales import en as _en
from .locales import ja as _ja
from .locales import ko as _ko
from .locales import zh_tw as _zh_tw

DEFAULT_LOCALE = "en"

CATALOGS: dict[str, dict[str, str]] = {
    "en": _en.MESSAGES,
    "ko": _ko.MESSAGES,
    "ja": _ja.MESSAGES,
    "zh-TW": _zh_tw.MESSAGES,
}

LOCALE_NAMES: dict[str, str] = {
    "en": "English",
    "ko": "한국어",
    "ja": "日本語",
    "zh-TW": "繁體中文",
}

# Accepts the forms found in LANG and in configuration files.
_ALIASES: dict[str, str] = {
    "en": "en", "en_us": "en", "en_gb": "en", "c": "en", "posix": "en",
    "ko": "ko", "ko_kr": "ko", "kor": "ko",
    "ja": "ja", "ja_jp": "ja", "jpn": "ja",
    "zh-tw": "zh-TW", "zh_tw": "zh-TW", "zh-hant": "zh-TW", "zh_hant": "zh-TW",
    "zh-hk": "zh-TW", "zh_hk": "zh-TW", "zh": "zh-TW",
}

_override: str | None = None


def available_locales() -> list[str]:
    return list(CATALOGS)


def normalize_locale(raw: str | None) -> str | None:
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
        from .config import load_settings

        value = load_settings().ui.language
    except Exception:  # noqa: BLE001
        return None
    return None if value == "auto" else normalize_locale(value)


def _from_environment() -> str | None:
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        code = normalize_locale(os.environ.get(var))
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


def set_locale(code: str | None) -> str:
    """Override the resolved locale for the remainder of the process.

    Used by the --lang option and by the configuration editor's live preview.
    Passing None restores automatic resolution.
    """
    global _override
    _override = normalize_locale(code) if code else None
    return current_locale()


def t(key: str, /, **fmt: object) -> str:
    """Look up a message and format it.

    An unknown key returns the key itself, which makes the omission visible in the
    interface instead of raising during a turn. A missing format argument returns the
    unformatted template for the same reason.
    """
    catalog = CATALOGS.get(current_locale(), CATALOGS[DEFAULT_LOCALE])
    template = catalog.get(key) or CATALOGS[DEFAULT_LOCALE].get(key) or key
    if not fmt:
        return template
    try:
        return template.format(**fmt)
    except (KeyError, IndexError, ValueError):
        return template
