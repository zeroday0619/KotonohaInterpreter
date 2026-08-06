"""CLI command execution paths that require asynchronous runtime wiring."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar, Final

import httpx2
import pytest
from typer.testing import CliRunner

from kotonoha._cli import app
from kotonoha._i18n import set_locale

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _reset_locale() -> Any:
    yield
    set_locale(None)


class HealthyResponse:
    __slots__: ClassVar[tuple[str, ...]] = ()
    status_code: Final = 200

    def raise_for_status(
        self,
        /,
    ) -> None:
        return None


class HealthyAsyncClient:
    __slots__: ClassVar[tuple[str, ...]] = (
        "options",
    )
    def __init__(
        self,
        /,
        **options: Any,
    ) -> None:
        self.options = options

    async def __aenter__(
        self,
        /,
    ) -> HealthyAsyncClient:
        return self

    async def __aexit__(
        self,
        /,
        *exception_details: Any,
    ) -> None:
        return None

    async def get(
        self,
        /,
        url: str,
    ) -> HealthyResponse:
        return HealthyResponse()

    async def post(
        self,
        /,
        url: str,
        **options: Any,
    ) -> HealthyResponse:
        return HealthyResponse()


def test_netcheck_executes_localized_output_before_measurement_loops(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(httpx2, "AsyncClient", HealthyAsyncClient)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--lang",
            "en",
            "--config",
            str(PROJECT_ROOT / "config" / "performance.yaml"),
            "netcheck",
            "--samples",
            "1",
            "--seconds",
            "0.01",
        ],
    )

    assert result.exit_code == 0, result.exception
    assert "perf_mode   remote" in result.stdout
    assert "asr         UP" in result.stdout
    assert "Estimated link overhead" in result.stdout


def test_integrated_tui_command_is_registered() -> None:
    result = CliRunner().invoke(app, ["tui", "--help"])

    assert result.exit_code == 0, result.exception
    assert "kotonoha tui" in result.stdout


@pytest.mark.parametrize(
    ("locale", "usage", "options", "commands", "completion", "help_text"),
    (
        ("ko", "사용법: ", "옵션", "명령", "현재 셸에 자동 완성을 설치합니다.", "이 도움말"),
        ("ja", "使用法： ", "オプション", "コマンド", "現在のシェルに補完", "このヘルプ"),
        ("zh-TW", "用法： ", "選項", "命令", "在目前命令殼層安裝", "顯示此說明"),
    ),
)
def test_framework_help_text_uses_the_import_time_locale(
    _positional_only: object | None = None,
    /,
    *,
    locale: str,
    usage: str,
    options: str,
    commands: str,
    completion: str,
    help_text: str,
) -> None:
    environment = os.environ.copy()
    environment["KOTONOHA_LANG"] = locale
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from kotonoha._cli import main; main()",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for expected in (usage, options, commands, completion, help_text):
        assert expected in completed.stdout
