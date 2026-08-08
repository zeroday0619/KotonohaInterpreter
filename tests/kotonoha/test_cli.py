"""CLI command execution paths that require asynchronous runtime wiring."""

from __future__ import annotations

import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, ClassVar, Final

import httpx2
import numpy as np
import pytest
from typer.testing import CliRunner

from kotonoha._cli import MAXIMUM_REPLAY_SECONDS, WAVE_READ_CHUNK_FRAMES, app, load_wave_file
from kotonoha._i18n import set_locale

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_wave_loader_streams_and_downmixes_multiple_chunks(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    path = tmp_path / "stereo.wav"
    frame_count = WAVE_READ_CHUNK_FRAMES + 17
    left = np.full(frame_count, 16384, dtype="<i2")
    right = np.full(frame_count, -8192, dtype="<i2")
    stereo = np.column_stack((left, right)).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as wave_writer:
        wave_writer.setnchannels(2)
        wave_writer.setsampwidth(2)
        wave_writer.setframerate(16000)
        wave_writer.writeframes(stereo.tobytes())

    audio = load_wave_file(path, 16000)

    assert audio.shape == (frame_count,)
    assert audio.dtype == np.float32
    assert np.allclose(audio, 0.125)


def test_wave_loader_caps_the_loaded_duration_before_allocation(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    path = tmp_path / "long.wav"
    sample_rate = 16000
    samples = np.arange(sample_rate * 2, dtype="<i2")
    with wave.open(str(path), "wb") as wave_writer:
        wave_writer.setnchannels(1)
        wave_writer.setsampwidth(2)
        wave_writer.setframerate(sample_rate)
        wave_writer.writeframes(samples.tobytes())

    audio = load_wave_file(path, sample_rate, maximum_seconds=1.0)

    assert audio.shape == (sample_rate,)


def test_replay_rejects_non_finite_or_excessive_duration(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.wav"
    path.touch()
    runner = CliRunner()

    for value in ("nan", str(MAXIMUM_REPLAY_SECONDS + 1.0)):
        result = runner.invoke(app, ["replay", str(path), "--seconds", value])
        assert result.exit_code == 2


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


def test_web_command_serves_the_browser_client() -> None:
    """The web command builds the browser app rather than relaying a terminal."""
    from kotonoha.web._server import create_app

    application = create_app()
    routes = {getattr(route, "path", None) for route in application.routes}

    assert "/ws" in routes
    assert "/api/logs" in routes
    assert "/health" in routes


def test_module_entry_point_exists() -> None:
    """python -m kotonoha is what the web server runs; a missing file fails at runtime."""
    import kotonoha

    assert (Path(kotonoha.__file__).parent / "__main__.py").is_file()
