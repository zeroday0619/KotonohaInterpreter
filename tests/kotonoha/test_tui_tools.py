"""Command construction for the integrated operations interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from kotonoha._i18n import set_locale
from kotonoha.tui._tools_app import ToolInputError, build_tool_command


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
    wav = tmp_path / "probe.wav"
    wav.touch()

    command = build_tool_command(
        "replay",
        {"wav": str(wav), "replay-seconds": "12.5"},
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
        str(wav),
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


def test_glossary_import_requires_and_passes_an_existing_file(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    glossary = tmp_path / "glossary.yaml"
    glossary.touch()

    command = build_tool_command("glossary_import", {"glossary-path": str(glossary)})

    assert command[-3:] == ["glossary", "import", str(glossary)]


@pytest.mark.parametrize(
    ("operation", "values"),
    [
        ("replay", {"wav": "missing.wav", "replay-seconds": "30"}),
        ("replay", {"wav": __file__, "replay-seconds": "0"}),
        ("serve", {"service": "llm", "host": "0.0.0.0", "port": ""}),
        ("serve", {"service": "asr", "host": "", "port": ""}),
        ("serve", {"service": "asr", "host": "0.0.0.0", "port": "70000"}),
        ("netcheck", {"samples": "0", "netcheck-seconds": "6"}),
    ],
)
def test_invalid_tool_input_is_rejected_before_process_start(
    _positional_only: object | None = None,
    /,
    *,
    operation: str,
    values: dict[str, str],
) -> None:
    with pytest.raises(ToolInputError):
        build_tool_command(operation, values)
