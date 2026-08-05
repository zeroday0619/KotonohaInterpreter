"""Deployment script interface and configuration templates."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

from kotonoha._config import read_yaml
from kotonoha._config_store import validate_candidate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = PROJECT_ROOT / "scripts" / "deploy.sh"
LLM_SCRIPT = PROJECT_ROOT / "scripts" / "run_vllm_llm.sh"


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
    assert "--reallocate-gpus" in help_result.stdout
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


def test_remote_services_default_to_mounted_offline_models() -> None:
    remote_config = read_yaml(PROJECT_ROOT / "config" / "remote-server.yaml")
    compose = (PROJECT_ROOT / "docker" / "compose.remote.yaml").read_text(encoding="utf-8")
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert remote_config["asr"]["vllm_model_id"] == "/models/Qwen3-ASR-1.7B"
    assert remote_config["asr_verify"]["model_id"] == "/models/faster-whisper-large-v3"
    assert remote_config["tts"]["model_id"] == "/models/Qwen3-TTS-0.6B"
    assert "HF_HUB_OFFLINE=${TRANSFORMERS_OFFLINE:-0}" in compose
    assert 'Qwen3-ASR-1.7B/config.json"' in deploy_script
    assert 'faster-whisper-large-v3/config.json"' in deploy_script
    assert 'Qwen3-TTS-0.6B/config.json"' in deploy_script


def test_remote_compose_uses_distinct_role_images_and_vllm_translation() -> None:
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker" / "compose.remote.yaml").read_text(encoding="utf-8")
    )
    services = compose["services"]
    python_roles = ("asr", "asr-verify", "tts")

    assert len({services[role]["image"] for role in python_roles}) == len(python_roles)
    assert {services[role]["build"]["target"] for role in python_roles} == set(python_roles)

    assert "vllm/vllm-openai:v0.19.1" in services["llm"]["image"]
    assert services["llm"]["entrypoint"] == [
        "bash",
        "/opt/kotonoha/run_vllm_llm.sh",
    ]
    assert any(
        "KOTONOHA_SERVICE_TOKEN=" in value
        for value in services["llm"]["environment"]
    )


def test_asr_images_use_vllm_runtimes_with_qwen3_support_checks() -> None:
    jetson_compose = yaml.safe_load(
        (PROJECT_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")
    )
    remote_compose = yaml.safe_load(
        (PROJECT_ROOT / "docker" / "compose.remote.yaml").read_text(encoding="utf-8")
    )
    jetson_dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.asr").read_text(
        encoding="utf-8"
    )
    remote_dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.remote").read_text(
        encoding="utf-8"
    )

    jetson_base = jetson_compose["services"]["asr"]["build"]["args"]["BASE_IMAGE"]
    remote_base = remote_compose["services"]["asr"]["build"]["args"]["ASR_BASE_IMAGE"]
    assert "nvidia-ai-iot/vllm" in jetson_base
    assert "vllm/vllm-openai:v0.19.1" in remote_base
    assert "rglob('qwen3_asr.py')" in jetson_dockerfile
    assert "rglob('qwen3_asr.py')" in remote_dockerfile
    assert "import vllm" not in jetson_dockerfile
    assert "import vllm" not in remote_dockerfile
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    assert "import torch, vllm" in deploy_script
    assert 'verify_vllm_cuda_runtime "$compose_file" "$environment_file" llm' in deploy_script
    assert "ENTRYPOINT []" in remote_dockerfile


def test_remote_lock_and_dockerfile_install_into_x86_conda_python() -> None:
    lock_text = (PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.remote").read_text(encoding="utf-8")

    assert "platform_machine == 'x86_64' and sys_platform == 'linux'" in lock_text
    assert "UV_PYTHON=/opt/conda/bin/python" in dockerfile
    assert "import kotonoha, pydantic_settings" in dockerfile


def test_editable_container_installs_include_the_custom_build_hook() -> None:
    standalone_dockerfiles = (
        PROJECT_ROOT / "docker" / "Dockerfile.asr",
        PROJECT_ROOT / "docker" / "Dockerfile.asr-verify",
        PROJECT_ROOT / "docker" / "Dockerfile.orchestrator",
        PROJECT_ROOT / "docker" / "Dockerfile.tts",
    )
    required_copy = "COPY pyproject.toml uv.lock README.md LICENSE hatch_build.py ./"

    for dockerfile_path in standalone_dockerfiles:
        dockerfile = dockerfile_path.read_text(encoding="utf-8")
        assert required_copy in dockerfile
        editable_install = "uv pip install --no-cache --no-deps -e ."
        assert dockerfile.index(required_copy) < dockerfile.index(editable_install)

    remote_dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.remote").read_text(
        encoding="utf-8"
    )
    editable_stages = tuple(
        stage for stage in remote_dockerfile.split("\nFROM ") if "-e ." in stage
    )
    assert len(editable_stages) == 2
    for stage in editable_stages:
        assert required_copy in stage
        assert stage.index(required_copy) < stage.index("-e .")


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


def test_python_service_containers_force_uvloop() -> None:
    compose_paths = (
        PROJECT_ROOT / "docker" / "compose.yaml",
        PROJECT_ROOT / "docker" / "compose.remote.yaml",
    )
    for compose_path in compose_paths:
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        for role in ("asr", "asr-verify", "tts"):
            assert "--loop uvloop" in compose["services"][role]["command"]

    dockerfile_paths = (
        PROJECT_ROOT / "docker" / "Dockerfile.asr",
        PROJECT_ROOT / "docker" / "Dockerfile.asr-verify",
        PROJECT_ROOT / "docker" / "Dockerfile.tts",
        PROJECT_ROOT / "docker" / "Dockerfile.remote",
    )
    for dockerfile_path in dockerfile_paths:
        dockerfile = dockerfile_path.read_text(encoding="utf-8")
        assert '"--loop", "uvloop"' in dockerfile


def _write_executable(
    path: Path,
    /,
    source: str,
) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_vllm_launcher_builds_the_authenticated_server_command(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    tool_directory = tmp_path / "tools"
    tool_directory.mkdir()
    capture_path = tmp_path / "arguments.txt"

    _write_executable(
        tool_directory / "vllm",
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_PATH"\n',
    )

    environment = {
        **os.environ,
        "PATH": f"{tool_directory}:{os.environ['PATH']}",
        "LLM_MODEL": str(model_path),
        "CAPTURE_PATH": str(capture_path),
        "KOTONOHA_SERVICE_TOKEN": "test-secret",
        "KOTONOHA_LLM_CONFIG_ENV": str(tmp_path / "missing.env"),
    }
    result = subprocess.run(
        ["bash", str(LLM_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    arguments = capture_path.read_text(encoding="utf-8").splitlines()
    assert arguments[:2] == ["serve", str(model_path)]
    assert "--served-model-name" in arguments
    assert "kotonoha-translation" in arguments
    assert "--quantization" in arguments
    assert "awq" in arguments
    assert arguments[-2:] == ["--api-key", "test-secret"]


def test_vllm_launcher_rejects_an_incomplete_model_snapshot(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    tool_directory = tmp_path / "tools"
    tool_directory.mkdir()
    _write_executable(tool_directory / "vllm", "#!/bin/sh\nexit 0\n")
    model_path = tmp_path / "model"
    model_path.mkdir()

    environment = {
        **os.environ,
        "PATH": f"{tool_directory}:{os.environ['PATH']}",
        "LLM_MODEL": str(model_path),
        "KOTONOHA_LLM_CONFIG_ENV": str(tmp_path / "missing.env"),
    }
    result = subprocess.run(
        ["bash", str(LLM_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "vLLM model snapshot is incomplete" in result.stdout
