"""Validated command construction for the Web operations panel."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from kotonoha._i18n import set_locale
from kotonoha._operation_catalog import ToolInputError, build_tool_command


@pytest.fixture(autouse=True)
def _english_locale() -> Any:
    set_locale("en")
    yield
    set_locale(None)


def test_replay_command_includes_global_and_command_options(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    wave_file = tmp_path / "probe.wav"
    wave_file.touch()
    command = build_tool_command(
        "replay",
        {"wav": str(wave_file), "replay-seconds": "12.5"},
        Path("device.yaml"),
    )
    assert command == [
        sys.executable,
        "-m",
        "kotonoha._cli",
        "--config",
        "device.yaml",
        "--lang",
        "en",
        "replay",
        str(wave_file),
        "--seconds",
        "12.5",
    ]


@pytest.mark.parametrize(
    ("operation", "values", "expected_tail"),
    [
        ("devices", {}, ["devices"]),
        (
            "serve",
            {"service": "verify", "host": "127.0.0.1", "port": "9002"},
            ["serve", "verify", "--host", "127.0.0.1", "--port", "9002"],
        ),
        ("doctor", {}, ["doctor"]),
        (
            "netcheck",
            {"samples": "20", "netcheck-seconds": "8"},
            ["netcheck", "--samples", "20", "--seconds", "8.0"],
        ),
        ("glossary_list", {}, ["glossary", "list"]),
        ("completion_show", {}, ["--show-completion"]),
        ("completion_install", {}, ["--install-completion"]),
    ],
)
def test_operation_command_covers_each_option(
    _positional_only: object | None = None,
    /,
    *,
    operation: str,
    values: dict[str, str],
    expected_tail: list[str],
) -> None:
    command = build_tool_command(operation, values)
    assert command[:5] == [sys.executable, "-m", "kotonoha._cli", "--lang", "en"]
    assert command[5:] == expected_tail


@pytest.mark.parametrize(
    ("operation", "values"),
    [
        ("replay", {"wav": "missing.wav", "replay-seconds": "30"}),
        ("serve", {"service": "llm", "host": "0.0.0.0", "port": ""}),
        ("serve", {"service": "asr", "host": "", "port": ""}),
        ("serve", {"service": "asr", "host": "0.0.0.0", "port": "70000"}),
        ("netcheck", {"samples": "0", "netcheck-seconds": "6"}),
    ],
)
def test_invalid_operation_input_is_rejected(
    _positional_only: object | None = None,
    /,
    *,
    operation: str,
    values: dict[str, str],
) -> None:
    with pytest.raises(ToolInputError):
        build_tool_command(operation, values)
