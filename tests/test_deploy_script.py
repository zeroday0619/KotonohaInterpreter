"""Deployment script interface and configuration templates."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from kotonoha.config import read_yaml
from kotonoha.config_store import validate_candidate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = PROJECT_ROOT / "scripts" / "deploy.sh"


def test_deploy_script_has_valid_shell_syntax_and_help() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(DEPLOY_SCRIPT)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    help_result = subprocess.run(
        ["bash", str(DEPLOY_SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "scripts/deploy.sh jetson" in help_result.stdout
    assert "scripts/deploy.sh a6000" in help_result.stdout
    assert os.access(DEPLOY_SCRIPT, os.X_OK)


def test_jetson_override_template_validates() -> None:
    template = read_yaml(PROJECT_ROOT / "config" / "jetson.local.example.yaml")
    assert validate_candidate(None, template) is None


def test_remote_override_template_validates() -> None:
    template = read_yaml(PROJECT_ROOT / "config" / "remote-server.local.example.yaml")
    remote_base = PROJECT_ROOT / "config" / "remote-server.yaml"
    assert validate_candidate(remote_base, template) is None
