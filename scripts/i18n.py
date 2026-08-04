#!/usr/bin/env python3
"""Translation catalog maintenance.

    uv run python scripts/i18n.py extract    # rebuild the .pot template from source
    uv run python scripts/i18n.py update     # merge the template into each .po
    uv run python scripts/i18n.py compile    # regenerate the .mo files
    uv run python scripts/i18n.py check      # report untranslated, fuzzy and stale

Babel is a development dependency; the runtime needs only the standard library's
gettext. English is the source language and has no catalog.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from babel.messages.catalog import Catalog
from babel.messages.extract import extract_from_dir
from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po, write_po

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kotonoha._i18n import (  # noqa: E402
    DEFAULT_LOCALE,
    DOMAIN,
    GETTEXT_NAMES,
    LOCALE_DIR,
    LOCALE_NAMES,
    mo_path,
    po_path,
)

SOURCE_DIR = REPO_ROOT / "src" / "kotonoha"
POT_PATH = LOCALE_DIR / f"{DOMAIN}.pot"

# `_` and `t` are the same callable. N_ marks strings held in import-time tables.
KEYWORDS = {"_": None, "t": None, "N_": None, "pgettext": ((1, "c"), 2)}
# The convenience wrapper formats with str.format, so entries carry that flag.
METHOD_MAP = [("**.py", "python")]
OPTIONS_MAP = {"**.py": {}}

TRANSLATOR_LANGUAGES = [code for code in LOCALE_NAMES if code != DEFAULT_LOCALE]


def build_template() -> Catalog:
    template = Catalog(
        project="Kotonoha Interpreter",
        version="0.1.0",
        msgid_bugs_address="",
        copyright_holder="Kotonoha Interpreter",
        charset="utf-8",
    )
    for filename, lineno, message, comments, context in extract_from_dir(
        SOURCE_DIR,
        method_map=METHOD_MAP,
        options_map=OPTIONS_MAP,
        keywords=KEYWORDS,
        comment_tags=("i18n:",),
    ):
        path = Path(SOURCE_DIR.name) / filename
        flags = ["python-brace-format"] if _has_placeholder(message) else []
        template.add(
            message,
            locations=[(path.as_posix(), lineno)],
            auto_comments=comments,
            context=context,
            flags=flags,
        )
    return template


def _has_placeholder(
    message: Any,
    /,
) -> bool:
    text = message if isinstance(message, str) else (message[0] if message else "")
    return "{" in text and "}" in text


def cmd_extract(
    _args: Any,
    /,
) -> int:
    template = build_template()
    POT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POT_PATH.open("wb") as handle:
        write_po(handle, template, width=88)
    print(f"{POT_PATH.relative_to(REPO_ROOT)}: {len(template)} messages")
    return 0


def cmd_update(
    _args: Any,
    /,
) -> int:
    template = build_template()
    for locale in TRANSLATOR_LANGUAGES:
        target = po_path(locale)
        if target.exists():
            with target.open("rb") as handle:
                catalog = read_po(handle, locale=GETTEXT_NAMES[locale], domain=DOMAIN)
            catalog.update(template, no_fuzzy_matching=False)
        else:
            catalog = Catalog(locale=GETTEXT_NAMES[locale], domain=DOMAIN)
            catalog.update(template)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            write_po(handle, catalog, width=88)
        missing = _untranslated(catalog)
        print(f"{target.relative_to(REPO_ROOT)}: {len(catalog)} messages, {missing} untranslated")
    return 0


def cmd_compile(
    _args: Any,
    /,
) -> int:
    for locale in TRANSLATOR_LANGUAGES:
        source = po_path(locale)
        if not source.exists():
            print(f"missing: {source.relative_to(REPO_ROOT)}", file=sys.stderr)
            return 1
        with source.open("rb") as handle:
            catalog = read_po(handle, locale=GETTEXT_NAMES[locale], domain=DOMAIN)
        target = mo_path(locale)
        with target.open("wb") as handle:
            write_mo(handle, catalog, use_fuzzy=False)
        translated = len(catalog) - _untranslated(catalog)
        print(f"{target.relative_to(REPO_ROOT)}: {translated} translated")
    return 0


def cmd_check(
    _args: Any,
    /,
) -> int:
    template = build_template()
    template_ids = {(entry.context, entry.id) for entry in template if entry.id}
    problems = 0

    for locale in TRANSLATOR_LANGUAGES:
        source = po_path(locale)
        if not source.exists():
            print(f"{locale}: no catalog", file=sys.stderr)
            problems += 1
            continue
        with source.open("rb") as handle:
            catalog = read_po(handle, locale=GETTEXT_NAMES[locale], domain=DOMAIN)

        catalog_ids = {(entry.context, entry.id) for entry in catalog if entry.id}
        missing = sorted(text for _context, text in template_ids - catalog_ids)
        obsolete = sorted(text for _context, text in catalog_ids - template_ids)
        untranslated = sorted(entry.id for entry in catalog if entry.id and not entry.string)
        fuzzy = sorted(entry.id for entry in catalog if entry.id and entry.fuzzy)

        for label, entries in (
            ("absent from catalog", missing),
            ("no longer in source", obsolete),
            ("untranslated", untranslated),
            ("fuzzy", fuzzy),
        ):
            if entries:
                problems += len(entries)
                print(f"{locale}: {len(entries)} {label}")
                for text in entries[:5]:
                    print(f"    {text[:76]}")

        if not mo_path(locale).exists():
            print(f"{locale}: .mo not compiled", file=sys.stderr)
            problems += 1

    print(f"template: {len(template_ids)} messages")
    if problems:
        print(f"{problems} problems; run extract, update, translate, then compile")
    return 1 if problems else 0


def _untranslated(
    catalog: Catalog,
    /,
) -> int:
    return sum(1 for entry in catalog if entry.id and not entry.string)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("extract", cmd_extract),
        ("update", cmd_update),
        ("compile", cmd_compile),
        ("check", cmd_check),
    ):
        sub.add_parser(name).set_defaults(handler=handler)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
