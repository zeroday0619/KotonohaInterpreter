"""Load a `.env` file before anything reads the environment.

Two kinds of variable reach this program. `KOTONOHA__SECTION__FIELD` is consumed by
pydantic-settings as the highest-precedence configuration layer, and a handful of
plain variables such as `KOTONOHA_CONFIG` are read through `os.environ` directly,
before any Settings object exists. A loader that only fed pydantic would therefore
cover half of them, so the file is merged into `os.environ` instead.

A real environment variable always wins over the file. Deployment sets variables
through Compose and systemd, and a checked-out `.env` that silently overrode them
would make a container behave unlike its own definition.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

DEFAULT_ENV_FILENAME: Final = ".env"
# Only variables this program owns are accepted, so a shared .env cannot inject
# PATH, LD_PRELOAD or anything else into the process.
ACCEPTED_PREFIXES: Final = ("KOTONOHA_",)


def parse_env_file(
    text: str,
    /,
) -> dict[str, str]:
    """Parse the `KEY=value` subset that deployment actually uses.

    Supported: comments, blank lines, `export ` prefixes, and single or double
    quoted values. Not supported: interpolation and multi-line values, because a
    variable whose meaning depends on evaluation order is harder to reason about
    than one that does not exist.
    """
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        if not separator:
            continue
        name = name.strip()
        if not name.isidentifier():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[name] = value
    return values


def load_env_file(
    path: Path | None = None,
    /,
    override: bool = False,
) -> dict[str, str]:
    """Merge `.env` into the process environment and return what was applied.

    `KOTONOHA_ENV_FILE` selects a different file. Setting it to an empty value
    disables the mechanism, which is what a container does when its variables
    come from Compose and a stray file must not participate.
    """
    if path is None:
        configured = os.environ.get("KOTONOHA_ENV_FILE")
        if configured is not None and not configured.strip():
            return {}
        path = Path(configured) if configured else Path(DEFAULT_ENV_FILENAME)

    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError, PermissionError):
        return {}

    applied: dict[str, str] = {}
    for name, value in parse_env_file(text).items():
        if not name.startswith(ACCEPTED_PREFIXES):
            continue
        if not override and name in os.environ:
            continue
        os.environ[name] = value
        applied[name] = value
    return applied
