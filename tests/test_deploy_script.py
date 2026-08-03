"""Deployment script interface and configuration templates."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

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
    assert "scripts/deploy.sh uninstall jetson" in help_result.stdout
    assert "scripts/deploy.sh uninstall a6000" in help_result.stdout
    assert os.access(DEPLOY_SCRIPT, os.X_OK)


def test_remove_images_requires_uninstall() -> None:
    result = subprocess.run(
        ["bash", str(DEPLOY_SCRIPT), "jetson", "--remove-images"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "valid only with uninstall" in result.stderr


def test_jetson_override_template_validates() -> None:
    template = read_yaml(PROJECT_ROOT / "config" / "jetson.local.example.yaml")
    assert validate_candidate(None, template) is None


def test_remote_override_template_validates() -> None:
    template = read_yaml(PROJECT_ROOT / "config" / "remote-server.local.example.yaml")
    remote_base = PROJECT_ROOT / "config" / "remote-server.yaml"
    assert validate_candidate(remote_base, template) is None


def test_remote_compose_uses_distinct_role_images_and_preserves_llama_binary() -> None:
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker" / "compose.remote.yaml").read_text(encoding="utf-8")
    )
    services = compose["services"]
    python_roles = ("asr", "asr-verify", "tts")

    assert len({services[role]["image"] for role in python_roles}) == len(python_roles)
    assert {services[role]["build"]["target"] for role in python_roles} == set(python_roles)

    llama_volumes = services["llm"]["volumes"]
    assert all(volume.split(":")[1] != "/app" for volume in llama_volumes)
    assert services["llm"]["entrypoint"] == ["bash", "/opt/kotonoha/run_llm.sh"]


def test_remote_lock_and_dockerfile_install_into_x86_conda_python() -> None:
    lock_text = (PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.remote").read_text(encoding="utf-8")

    assert "platform_machine == 'x86_64' and sys_platform == 'linux'" in lock_text
    assert "UV_PYTHON=/opt/conda/bin/python" in dockerfile
    assert "import kotonoha, pydantic_settings" in dockerfile


def test_remote_tts_image_builds_and_verifies_required_dependencies() -> None:
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker" / "compose.remote.yaml").read_text(encoding="utf-8")
    )
    remote_config = read_yaml(PROJECT_ROOT / "config" / "remote-server.yaml")
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.remote").read_text(encoding="utf-8")

    assert "sox libsox-fmt-all" in dockerfile
    assert "FROM ${TTS_BUILD_IMAGE} AS tts-flash-builder" in dockerfile
    assert '"flash-attn==${FLASH_ATTN_VERSION}"' in dockerfile
    assert "import flash_attn, pydantic_settings, qwen_tts, sox, torch" in dockerfile
    assert "melotts" not in dockerfile
    assert "|| echo" not in dockerfile
    assert "TTS_BUILD_IMAGE" in compose["services"]["tts"]["build"]["args"]
    assert remote_config["tts"]["fallback"] == "none"
