"""The lock must stay installable on the Python each deployment image provides.

uv resolves from distribution metadata. A release whose metadata admits an
interpreter but which publishes no wheel for it resolves cleanly and then fails
at install time, inside the container build:

    error: Distribution `onnxruntime==1.24.3` can't be installed because it
    doesn't have a source distribution or wheel for the current platform
    hint: You're using CPython 3.10 (`cp310`), but `onnxruntime` (v1.24.3) only
    has wheels with the following Python implementation tags: `cp311`, ...

Python 3.10 remains a supported wheel target, so the constraint in pyproject.toml
holds those distributions to the last release that ships cp310 wheels.
"""

from __future__ import annotations

import pathlib
import re

import tomllib

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCK_PATH = REPOSITORY_ROOT / "uv.lock"
PROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"

# Verified against the PyPI release index: 1.23.2 is the final onnxruntime
# release publishing cp310 aarch64 wheels. 1.24.0 onward are cp311 and later.
LAST_CP310_ONNXRUNTIME = (1, 23, 2)


def _version(
    text: str,
    /,
) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text))


def test_the_python_310_constraint_is_declared() -> None:
    project = tomllib.loads(PROJECT_PATH.read_text(encoding="utf-8"))
    constraints = project["tool"]["uv"]["constraint-dependencies"]

    assert any(
        "onnxruntime" in constraint and "3.11" in constraint for constraint in constraints
    ), "the cp310 onnxruntime ceiling is missing; the Jetson asr-verify image will not build"


def test_the_locked_python_310_onnxruntime_publishes_cp310_wheels() -> None:
    lock = LOCK_PATH.read_text(encoding="utf-8")
    pinned = {
        _version(match.group("version"))
        for match in re.finditer(
            r'name = "onnxruntime", version = "(?P<version>[^"]+)"[^\n]*'
            r"python_full_version < '3\.11'",
            lock,
        )
    }

    assert pinned, "no onnxruntime resolution found for Python 3.10"
    for version in pinned:
        assert version <= LAST_CP310_ONNXRUNTIME, (
            f"onnxruntime {'.'.join(map(str, version))} is locked for Python 3.10 but "
            "publishes no cp310 wheel"
        )
