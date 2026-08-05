"""Unified setup and benchmark management script contracts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANAGEMENT_SCRIPT = PROJECT_ROOT / "scripts" / "manage.sh"


def _run_management_script(
    arguments: tuple[str, ...],
    /,
    *,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(MANAGEMENT_SCRIPT), *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_management_script_has_valid_shell_syntax_and_help() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(MANAGEMENT_SCRIPT)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    help_result = _run_management_script(("--help",))

    assert syntax.returncode == 0, syntax.stderr
    assert help_result.returncode == 0
    assert "setup workstation" in help_result.stdout
    assert "models verify" in help_result.stdout
    assert "benchmark jetson" in help_result.stdout
    assert "benchmark link" in help_result.stdout
    assert "deploy a6000" in help_result.stdout
    assert os.access(MANAGEMENT_SCRIPT, os.X_OK)


@pytest.mark.parametrize(
    ("arguments", "expected_commands"),
    (
        (("setup", "workstation", "--eval"), ("uv sync --group eval", "i18n.py compile")),
        (("setup", "jetson"), ("deploy.sh jetson --prepare-only",)),
        (("models", "fetch"), ("fetch_models.sh",)),
        (("benchmark", "a6000", "--only", "3"), ("run_all.sh a6000 --only 3",)),
        (("benchmark", "link", "--samples", "2"), ("netcheck --samples 2",)),
        (("deploy", "a6000", "--no-build"), ("deploy.sh a6000 --no-build",)),
        (("uninstall", "jetson"), ("deploy.sh uninstall jetson",)),
        (("gpu", "allocate", "--force"), ("allocate_gpus.py --force",)),
    ),
)
def test_management_dry_run_routes_to_existing_workflows(
    _positional_only: object | None = None,
    /,
    *,
    arguments: tuple[str, ...],
    expected_commands: tuple[str, ...],
) -> None:
    completed = _run_management_script(("--dry-run", *arguments))

    assert completed.returncode == 0, completed.stderr
    assert all(command in completed.stdout for command in expected_commands)


def test_model_verification_reports_every_missing_artifact(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    completed = _run_management_script(
        ("models", "verify"),
        environment={**os.environ, "MODELS_DIR": str(tmp_path)},
    )

    assert completed.returncode == 1
    assert completed.stderr.count("MISSING:") == 8
    assert "8 required model artifact(s) are missing" in completed.stderr
