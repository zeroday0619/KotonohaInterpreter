"""Unified setup and benchmark management script contracts."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANAGEMENT_SCRIPT = PROJECT_ROOT / "scripts" / "manage.sh"


def _run_management_script(
    arguments: tuple[str, ...],
    /,
    *,
    assume_yes: bool = True,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = ["bash", str(MANAGEMENT_SCRIPT)]
    if assume_yes:
        command.append("--yes")
    command.extend(arguments)
    return subprocess.run(
        command,
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
    assert "setup [auto|workstation|jetson|a6000]" in help_result.stdout
    assert "models verify" in help_result.stdout
    assert "i18n [extract|update|compile|check]" in help_result.stdout
    assert "benchmark [auto|jetson|a6000]" in help_result.stdout
    assert "benchmark link" in help_result.stdout
    assert "deploy [auto|jetson|a6000]" in help_result.stdout
    assert "web [auto|jetson|a6000]" in help_result.stdout
    assert "tui" in help_result.stdout
    assert "detect" in help_result.stdout
    assert "--yes" in help_result.stdout
    assert "--keep-images" in help_result.stdout
    assert os.access(MANAGEMENT_SCRIPT, os.X_OK)


@pytest.mark.parametrize(
    ("arguments", "expected_commands"),
    (
        (
            ("setup", "workstation", "--eval"),
            ("uv sync --group eval", "scripts/py/i18n.py compile"),
        ),
        (("setup", "jetson"), ("deploy.sh jetson --prepare-only",)),
        (("models", "fetch"), ("fetch_models.sh",)),
        (("i18n", "compile"), ("scripts/py/i18n.py compile",)),
        (("benchmark", "a6000", "--only", "3"), ("run_all.sh a6000 --only 3",)),
        (("benchmark", "link", "--samples", "2"), ("netcheck --samples 2",)),
        (("deploy", "a6000", "--no-build"), ("deploy.sh a6000 --no-build",)),
        (("tui",), ("docker compose", "docker/compose.yaml", "run --rm orchestrator")),
        (
            ("web", "a6000"),
            (
                "docker/compose.remote.yaml",
                "docker/compose.web.yaml",
                "docker/compose.web.a6000.yaml",
                "up -d web",
            ),
        ),
        (
            ("uninstall", "jetson"),
            ("deploy.sh uninstall jetson --remove-images",),
        ),
        (("gpu", "allocate", "--force"), ("scripts/py/allocate_gpus.py --force",)),
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


def test_management_commands_require_confirmation_in_noninteractive_sessions() -> None:
    completed = _run_management_script(
        ("--dry-run", "check"),
        assume_yes=False,
    )

    assert completed.returncode == 1
    assert "confirmation requires an interactive terminal" in completed.stderr


@pytest.mark.parametrize("equipment", ("workstation", "jetson", "a6000"))
def test_equipment_detection_accepts_automation_override(
    _positional_only: object | None = None,
    /,
    *,
    equipment: str,
) -> None:
    completed = _run_management_script(
        ("detect",),
        environment={**os.environ, "KOTONOHA_EQUIPMENT": equipment},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.rstrip().endswith(equipment)


def test_automatic_target_routes_deployment_to_detected_equipment() -> None:
    completed = _run_management_script(
        ("--dry-run", "deploy"),
        environment={**os.environ, "KOTONOHA_EQUIPMENT": "jetson"},
    )

    assert completed.returncode == 0, completed.stderr
    assert "scripts/deploy.sh jetson" in completed.stdout


def test_uninstall_can_preserve_project_images_explicitly() -> None:
    completed = _run_management_script(
        ("--dry-run", "uninstall", "jetson", "--keep-images"),
    )

    assert completed.returncode == 0, completed.stderr
    assert "deploy.sh uninstall jetson" in completed.stdout
    assert "--remove-images" not in completed.stdout


def test_python_management_tools_live_under_scripts_py() -> None:
    assert not list((PROJECT_ROOT / "scripts").glob("*.py"))
    assert (PROJECT_ROOT / "scripts" / "py" / "allocate_gpus.py").is_file()
    assert (PROJECT_ROOT / "scripts" / "py" / "i18n.py").is_file()


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


def test_check_reports_step_progress() -> None:
    """The quality gates run three long tools; the operator sees which one is active."""
    result = _run_management_script(
        ("-y", "check", "--dry-run"),
        environment={**os.environ, "KOTONOHA_EQUIPMENT": "workstation"},
    )

    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "== Quality gates (3 steps)" in combined
    assert "(1/3) ruff" in combined
    assert "(2/3) pytest" in combined
    assert "(3/3) translation catalogs" in combined
    assert "100%" in combined


def test_progress_never_reports_more_steps_than_it_declares() -> None:
    """A miscounted total silently produces a bar over 100 percent."""
    script = MANAGEMENT_SCRIPT.read_text(encoding="utf-8")

    declared = re.findall(r"progress_begin (\d+) ", script)
    steps_per_block = [
        len(re.findall(r"^\s*progress_step ", block, re.MULTILINE))
        for block in script.split("progress_begin ")[1:]
    ]

    assert declared, "no progress block found"
    assert [int(total) for total in declared] == steps_per_block
