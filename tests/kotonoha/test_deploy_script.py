"""Deployment script interface and configuration templates."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml

from kotonoha._config import read_yaml
from kotonoha._config_store import validate_candidate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = PROJECT_ROOT / "scripts" / "deploy.sh"
LLM_SERVER = PROJECT_ROOT / "src" / "kotonoha" / "services" / "_llm_server.py"
TTS_SERVER = PROJECT_ROOT / "src" / "kotonoha" / "services" / "_tts_server.py"


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
    assert "--prepare-only" in help_result.stdout
    assert os.access(DEPLOY_SCRIPT, os.X_OK)


def test_deploy_script_preserves_compose_variables_through_sudo() -> None:
    source = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        r"docker_environment_names=\(\n(?P<variables>.*?)\n\)",
        source,
        flags=re.DOTALL,
    )

    assert match is not None
    preserved_variables = set(match.group("variables").split())
    compose_variables: set[str] = set()
    for compose_name in ("compose.yaml", "compose.remote.yaml"):
        compose_source = (PROJECT_ROOT / "docker" / compose_name).read_text(
            encoding="utf-8"
        )
        compose_variables.update(re.findall(r"\$\{([A-Z][A-Z0-9_]*)", compose_source))

    assert compose_variables - {"KOTONOHA_SERVICE_TOKEN"} <= preserved_variables
    assert "KOTONOHA_SERVICE_TOKEN" not in preserved_variables
    assert 'sudo env "${docker_environment[@]}" docker "$@"' in source
    assert "docker_command" not in source


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


def test_prepare_only_rejects_service_stopping_gpu_reallocation() -> None:
    result = subprocess.run(
        [
            "bash",
            str(DEPLOY_SCRIPT),
            "a6000",
            "--prepare-only",
            "--reallocate-gpus",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "cannot be combined" in result.stderr


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

    assert remote_config["asr"]["vllm_model_id"] == (
        "/models/Voxtral-Mini-4B-Realtime-2602"
    )
    assert remote_config["asr"]["vllm_realtime_architecture"] == "voxtral"
    assert remote_config["asr"]["vllm_gpu_memory_utilization"] == 0.28
    assert remote_config["asr_verify"]["model_id"] == "/models/faster-whisper-large-v3"
    assert "vllm/vllm-omni:v0.26.0" in compose
    assert "HF_HUB_OFFLINE=${TRANSFORMERS_OFFLINE:-0}" in compose
    assert 'Voxtral-Mini-4B-Realtime-2602/config.json"' in deploy_script
    assert "ASR_GPU_MEMORY_MIB=14336" in deploy_script
    assert "ASR_GPU_MEMORY_MIB to at least 14336" in deploy_script
    assert 'faster-whisper-large-v3/config.json"' in deploy_script
    assert 'Qwen3-TTS-0.6B/config.json"' in deploy_script
    assert 'llm/translategemma-12b-it/config.json"' in deploy_script
    assert remote_config["llm"]["profiles"]["translategemma"]["directory"] == (
        "translategemma-12b-it"
    )


def test_remote_compose_uses_distinct_role_images_and_in_process_translation() -> None:
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker" / "compose.remote.yaml").read_text(encoding="utf-8")
    )
    services = compose["services"]
    python_roles = ("asr", "asr-verify", "llm")

    assert len({services[role]["image"] for role in python_roles}) == len(python_roles)
    assert {services[role]["build"]["target"] for role in python_roles} == set(python_roles)
    assert services["tts"]["image"] == "kotonohainterpreter-tts"
    assert services["tts"]["build"]["dockerfile"] == "docker/Dockerfile.tts"
    assert "vllm/vllm-omni:v0.26.0" in services["tts"]["build"]["args"]["BASE_IMAGE"]

    assert services["llm"]["image"] == "kotonohainterpreter-llm"
    assert services["llm"]["build"]["target"] == "llm"
    assert "nvcr.io/nvidia/vllm:26.07-py3" in services["llm"]["build"]["args"][
        "LLM_BASE_IMAGE"
    ]
    assert "kotonoha.services._llm_server:app" in services["llm"]["command"]
    assert any(
        "KOTONOHA_SERVICE_TOKEN=" in value
        for value in services["llm"]["environment"]
    )


def test_asr_images_use_target_vllm_runtimes_with_realtime_support_checks() -> None:
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
    assert "ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04" in jetson_base
    assert "nvcr.io/nvidia/vllm:26.07-py3" in remote_base
    assert "rglob('qwen3_asr.py')" in jetson_dockerfile
    assert "rglob('qwen3_asr_realtime.py')" in jetson_dockerfile
    assert "rglob('voxtral_realtime.py')" in remote_dockerfile
    assert "rglob('realtime/connection.py')" in remote_dockerfile
    assert "vllm-0.24.0-voxtral-mixed-prefill.patch" in remote_dockerfile
    assert 'patch --batch --forward "$voxtral_module"' in remote_dockerfile
    assert '"$UV_PYTHON" -m py_compile "$voxtral_module"' in remote_dockerfile
    voxtral_patch = (
        PROJECT_ROOT / "docker" / "patches" / "vllm-0.24.0-voxtral-mixed-prefill.patch"
    ).read_text(encoding="utf-8")
    assert "if mm_embeds_flat.shape[0] == input_ids.shape[0]" in voxtral_patch
    assert "mixed_embeddings[is_multimodal] = mm_embeds_flat" in voxtral_patch
    assert "return mixed_embeddings" in voxtral_patch
    assert "import vllm" not in jetson_dockerfile
    assert "import vllm" not in remote_dockerfile
    assert "uv venv --python python3 --system-site-packages /opt/kotonoha-venv" in (
        remote_dockerfile
    )
    assert "uv sync --active --frozen --no-dev" in remote_dockerfile
    assert "uv pip install --system" not in remote_dockerfile
    asr_stage = remote_dockerfile.split("FROM ${LLM_BASE_IMAGE} AS llm", 1)[0]
    assert "--no-install-package numpy" not in asr_stage
    final_sync = asr_stage.rindex("uv sync --active")
    numpy_override = asr_stage.index(
        '--reinstall-package numpy "numpy>=2,<2.3"'
    )
    dependency_check = asr_stage.index('uv pip check --python "$UV_PYTHON"')
    assert final_sync < numpy_override < dependency_check
    assert 'uv pip check --python "$UV_PYTHON"' in asr_stage
    assert "import kotonoha, mistral_common, numpy, scipy, sklearn, soundfile, soxr" in (
        remote_dockerfile
    )
    assert "from transformers import GenerationMixin" in remote_dockerfile
    assert "(2, 0) <= numpy_release < (2, 3)" in remote_dockerfile
    assert "Path(numpy.__file__).is_relative_to('/opt/kotonoha-venv')" in (
        remote_dockerfile
    )
    assert "distribution('nvidia-cufft')" in asr_stage
    assert "--reinstall-package nvidia-cufft" in asr_stage
    assert "--reinstall-package nvidia-nvjitlink" in asr_stage
    assert "library.parent / 'libnvJitLink.so.13'" in asr_stage
    assert "path.stat().st_size > 1048576" in asr_stage
    assert 'ln -s "$cufft_library_directory" /opt/kotonoha-cufft' in asr_stage
    assert "ENV LD_LIBRARY_PATH=/opt/kotonoha-cufft:${LD_LIBRARY_PATH}" in asr_stage
    remote_llm_stage = remote_dockerfile.split("FROM ${LLM_BASE_IMAGE} AS llm", 1)[1]
    remote_llm_stage = remote_llm_stage.split("FROM common AS asr-verify", 1)[0]
    jetson_llm_dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.llm").read_text(
        encoding="utf-8"
    )
    for llm_stage in (remote_llm_stage, jetson_llm_dockerfile):
        assert "--no-install-package numpy" not in llm_stage
        assert "Path(numpy.__file__).is_relative_to('/opt/kotonoha-venv')" in llm_stage
        assert 'uv pip check --python "$UV_PYTHON"' in llm_stage
    remote_llm_final_sync = remote_llm_stage.rindex("uv sync --active")
    remote_llm_numpy_override = remote_llm_stage.index(
        '--reinstall-package numpy "numpy>=2,<2.3"'
    )
    remote_llm_dependency_check = remote_llm_stage.index(
        'uv pip check --python "$UV_PYTHON"'
    )
    assert remote_llm_final_sync < remote_llm_numpy_override
    assert remote_llm_numpy_override < remote_llm_dependency_check
    assert "import kotonoha, numpy, scipy, sklearn, websockets" in remote_llm_stage
    assert "from transformers import GenerationMixin" in remote_llm_stage
    assert "(2, 0) <= numpy_release < (2, 3)" in remote_llm_stage
    assert "import kotonoha, numpy, websockets" in jetson_llm_dockerfile
    assert "Path('/opt/venv/lib').glob('python*/site-packages')" in (
        jetson_llm_dockerfile
    )
    assert "vendor-vllm.pth" in jetson_llm_dockerfile
    assert "root.is_relative_to('/opt/venv')" in jetson_llm_dockerfile
    jetson_final_sync = jetson_llm_dockerfile.rindex("uv sync --active")
    jetson_vendor_path = jetson_llm_dockerfile.index("vendor-vllm.pth")
    jetson_vllm_check = jetson_llm_dockerfile.index("root.is_relative_to('/opt/venv')")
    assert jetson_final_sync < jetson_vendor_path < jetson_vllm_check
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    assert 'a6000_vllm_image="nvcr.io/nvidia/vllm:26.07-py3"' in deploy_script
    assert "must set REMOTE_ASR_BASE=$a6000_vllm_image" in deploy_script
    assert "must set LLM_IMAGE=$a6000_vllm_image" in deploy_script
    assert "import torch, vllm" in deploy_script
    assert 'verify_vllm_cuda_runtime "$compose_file" "$environment_file" llm' in deploy_script
    assert "ENTRYPOINT []" in remote_dockerfile
    assert "TRANSFORMERS_FALLBACK_VERSION=5.13.0" in jetson_dockerfile
    assert "SPIKE_TRANSFORMERS_PYTHON=/opt/transformers-fallback/bin/python" in (
        jetson_dockerfile
    )
    assert jetson_dockerfile.count("env -u UV_CONSTRAINT uv pip install") == 2
    fallback_install = jetson_dockerfile.index(
        '"transformers==${TRANSFORMERS_FALLBACK_VERSION}" librosa soundfile'
    )
    vendor_packages = jetson_dockerfile.index("vendor-vllm.pth")
    fallback_import = jetson_dockerfile.index(
        "from transformers import AutoModelForMultimodalLM, AutoProcessor"
    )
    assert fallback_install < vendor_packages < fallback_import
    assert "transformers.__version__ == '${TRANSFORMERS_FALLBACK_VERSION}'" in (
        jetson_dockerfile
    )
    assert "Path(torch.__file__).is_relative_to('/opt/venv')" in jetson_dockerfile
    assert "from transformers import AutoModelForMultimodalLM, AutoProcessor" in (
        jetson_dockerfile
    )


def test_jetson_images_use_pinned_r36_4_tegra_runtime() -> None:
    jetson_image = "ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04"
    compose_source = (PROJECT_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")
    deploy_source = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    dockerfiles = tuple(
        (PROJECT_ROOT / "docker" / name).read_text(encoding="utf-8")
        for name in (
            "Dockerfile.asr",
            "Dockerfile.asr-verify",
            "Dockerfile.orchestrator",
        )
    )

    assert "Jetson Linux 39.2" in compose_source
    assert compose_source.count(jetson_image) == 4
    assert "vllm/vllm-omni:v0.26.0" in compose_source
    assert "Jetson Linux 39.2" in deploy_source
    assert "R39.*REVISION: 2" in deploy_source
    assert all(jetson_image in source for source in dockerfiles)
    assert all("ENTRYPOINT []" in source for source in dockerfiles)


def test_remote_lock_and_dockerfile_use_target_specific_python_environments() -> None:
    lock_text = (PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.remote").read_text(encoding="utf-8")
    project_configuration = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "platform_machine == 'x86_64' and sys_platform == 'linux'" in lock_text
    assert '"mistral-common[audio]>=1.11.3"' in project_configuration
    assert (
        '"nvidia-cufft>=12,<13 ; platform_machine == \'x86_64\' and '
        'sys_platform == \'linux\'"'
    ) in project_configuration
    assert "UV_PYTHON=/opt/conda/bin/python" in dockerfile
    assert "UV_PYTHON=/opt/kotonoha-venv/bin/python" in dockerfile
    assert dockerfile.count("--extra a6000-asr") == 2
    assert "import kotonoha, mistral_common, numpy" in dockerfile
    assert "soundfile, soxr" in dockerfile
    assert "import kotonoha, pydantic_settings" in dockerfile


def test_editable_container_installs_include_the_custom_build_hook() -> None:
    standalone_dockerfiles = (
        PROJECT_ROOT / "docker" / "Dockerfile.asr",
        PROJECT_ROOT / "docker" / "Dockerfile.asr-verify",
        PROJECT_ROOT / "docker" / "Dockerfile.orchestrator",
    )
    required_copy = "COPY pyproject.toml uv.lock README.md LICENSE hatch_build.py ./"

    for dockerfile_path in standalone_dockerfiles:
        dockerfile = dockerfile_path.read_text(encoding="utf-8")
        assert required_copy in dockerfile
        editable_install = "--no-cache --no-deps -e ."
        assert dockerfile.index(required_copy) < dockerfile.index(editable_install)

    remote_dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.remote").read_text(
        encoding="utf-8"
    )
    editable_stages = tuple(
        stage for stage in remote_dockerfile.split("\nFROM ") if "-e ." in stage
    )
    assert len(editable_stages) == 1
    for stage in editable_stages:
        assert required_copy in stage
        assert stage.index(required_copy) < stage.index("-e .")

    asr_stage = next(
        stage
        for stage in remote_dockerfile.split("\nFROM ")
        if stage.startswith("${ASR_BASE_IMAGE}")
    )
    assert required_copy in asr_stage
    assert asr_stage.index(required_copy) < asr_stage.index("--no-install-project")
    assert asr_stage.index("COPY src ./src") < asr_stage.rindex("uv sync --active")


def test_tts_uses_the_fastapi_service_on_the_official_vllm_omni_base() -> None:
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker" / "compose.remote.yaml").read_text(encoding="utf-8")
    )
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.tts").read_text(encoding="utf-8")
    server = TTS_SERVER.read_text(encoding="utf-8")
    service = compose["services"]["tts"]

    assert service["image"] == "kotonohainterpreter-tts"
    assert service["build"]["dockerfile"] == "docker/Dockerfile.tts"
    assert "vllm/vllm-omni:v0.26.0" in service["build"]["args"]["BASE_IMAGE"]
    assert "uvicorn kotonoha.services._tts_server:app" in service["command"]
    assert "ARG BASE_IMAGE=vllm/vllm-omni:v0.26.0" in dockerfile
    assert "uv sync --active --frozen" in dockerfile
    assert "--system-site-packages" in dockerfile
    final_sync = dockerfile.rindex("uv sync --active")
    numpy_override = dockerfile.index('--reinstall-package numpy "numpy>=2,<2.3"')
    dependency_check = dockerfile.index('uv pip check --python "$UV_PYTHON"')
    assert final_sync < numpy_override < dependency_check
    assert "'long' in numpy.__dict__" in dockerfile
    assert 'app = FastAPI(title="kotonoha-tts"' in server
    assert '@app.post("/v1/audio/speech")' in server
    assert "AsyncOmni" in server
    assert "OmniOpenAIServingSpeech" in server
    assert "create_subprocess_exec" not in server
    assert "install_auth(app, \"tts\")" in server


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
        PROJECT_ROOT / "docker" / "Dockerfile.remote",
        PROJECT_ROOT / "docker" / "Dockerfile.tts",
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


def test_privileged_uninstall_forwards_generated_compose_token(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    tool_directory = tmp_path / "tools"
    tool_directory.mkdir()
    capture_path = tmp_path / "docker-environment.txt"
    environment_file = tmp_path / "missing.env"
    _write_executable(
        tool_directory / "docker",
        """#!/bin/sh
if [ "${KOTONOHA_TEST_SUDO:-0}" = 1 ]; then
  previous=""
  for argument in "$@"; do
    if [ "$previous" = --env-file ]; then
      sed -n 's/^KOTONOHA_SERVICE_TOKEN=//p' "$argument" >> "$CAPTURE_PATH"
    fi
    previous=$argument
  done
  exit 0
fi
exit 1
""",
    )
    _write_executable(
        tool_directory / "sudo",
        """#!/bin/sh
if [ "$1" = docker ] && [ "$2" = info ]; then
  exit 0
fi
if [ "$1" = env ]; then
  shift
  exec env KOTONOHA_TEST_SUDO=1 "$@"
fi
exit 1
""",
    )
    environment = {
        **os.environ,
        "PATH": f"{tool_directory}:{os.environ['PATH']}",
        "CAPTURE_PATH": str(capture_path),
        "TMPDIR": str(tmp_path),
    }

    result = subprocess.run(
        [
            "bash",
            str(DEPLOY_SCRIPT),
            "uninstall",
            "a6000",
            "--env-file",
            str(environment_file),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "using sudo docker" in result.stdout
    captured_tokens = capture_path.read_text(encoding="utf-8").splitlines()
    assert captured_tokens[-1] == "uninstall-only"
    assert not tuple(tmp_path.glob("kotonoha-uninstall.*"))


def test_llm_service_owns_the_engine_without_a_nested_server() -> None:
    source = LLM_SERVER.read_text(encoding="utf-8")

    assert "build_async_engine_client_from_engine_args" in source
    assert "create_subprocess_exec" not in source
    assert '"serve"' not in source
