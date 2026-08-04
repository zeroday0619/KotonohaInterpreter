"""CLI command execution paths that require asynchronous runtime wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Final

import httpx
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
    monkeypatch.setattr(httpx, "AsyncClient", HealthyAsyncClient)
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
