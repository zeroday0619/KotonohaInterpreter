"""Test-wide isolation.

The suite must run offline and produce the same result on every host (AGENTS.md).
The deployment database is real device state and carries service endpoints and
a bearer token — so reading it would point tests at the external server and make
results depend on whose machine they run on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx2
import pytest

httpx2.alias_httpx()


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


@pytest.fixture(autouse=True)
def _isolate_settings_database(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give every test an empty SQLite configuration state."""
    monkeypatch.setenv("KOTONOHA_SKIP_LOCAL_CONFIG", "1")
    monkeypatch.setenv("KOTONOHA_SETTINGS_DATABASE", str(tmp_path / "kotonoha.db"))
