"""Test-wide isolation.

The suite must run offline and produce the same result on every host (AGENTS.md).
config/local.yaml is a real device's configuration — it carries remote endpoints and
a bearer token — so reading it would point tests at the external server and make
results depend on whose machine they run on.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True, scope="session")
def _compile_translation_catalogs() -> Any:
    """Build the .mo files the interface reads.

    .mo is generated at install time and is not committed, so a fresh checkout has
    none. Compiling here from the committed .po also means the suite can never
    assert against a stale artifact left over from an earlier edit.
    """
    babel_pofile = pytest.importorskip("babel.messages.pofile")
    babel_mofile = pytest.importorskip("babel.messages.mofile")

    from kotonoha._i18n import DEFAULT_LOCALE, DOMAIN, available_locales, mo_path, po_path

    for locale in available_locales():
        if locale == DEFAULT_LOCALE:
            continue
        source = po_path(locale)
        if not source.exists():
            continue
        with source.open("rb") as handle:
            catalog = babel_pofile.read_po(handle, domain=DOMAIN)
        with mo_path(locale).open("wb") as handle:
            babel_mofile.write_mo(handle, catalog, use_fuzzy=False)
    yield


@pytest.fixture(autouse=True, scope="session")
def _ignore_device_local_config() -> Any:
    import os

    previous = os.environ.get("KOTONOHA_SKIP_LOCAL_CONFIG")
    os.environ["KOTONOHA_SKIP_LOCAL_CONFIG"] = "1"
    yield
    if previous is None:
        os.environ.pop("KOTONOHA_SKIP_LOCAL_CONFIG", None)
    else:
        os.environ["KOTONOHA_SKIP_LOCAL_CONFIG"] = previous
