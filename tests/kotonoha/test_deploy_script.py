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


def test_deploy_inline_python_checks_have_valid_syntax() -> None:
    source = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    commands = re.findall(r"-c \\\n\s+'([^']+)' \\", source)

    assert len(commands) == 6
    for command in commands:
        compile(command, "scripts/deploy.sh", "exec")


def test_vllm_probes_terminate_before_torch_finalizers_run() -> None:
    """The three torch/vLLM probes must end with os._exit(0).

    Loading vLLM registers torch.library custom operators whose weakref finalizers
    fail at interpreter shutdown with a UnicodeDecodeError from
    torch._C._jit_get_operation. CPython treats that as an ignored exception and
    still exits 0, so the probe passes while printing a traceback that reads as a
    deployment failure. Terminating first removes the noise; the assertions have
    already run, so a real failure still exits non-zero.
    """
    source = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    commands = re.findall(r"-c \\\n\s+'([^']+)' \\", source)
    guarded = [command for command in commands if "_os._exit(0)" in command]
    vllm_probes = [command for command in commands if "vllm.__version__" in command]

    assert len(vllm_probes) == 3
    assert sorted(guarded) == sorted(vllm_probes)
    for command in guarded:
        assert command.rstrip().endswith("_os._exit(0)")
        assert "_sys.stdout.flush()" in command


def test_jetson_power_commands_that_need_root_are_elevated() -> None:
    """`jetson_clocks` refuses to run as a non-root user, `--show` included.

    Docker elevates itself, so an operator who runs the script without sudo gets
    as far as the power readback and then stops with

        Error: Run this script(/usr/bin/jetson_clocks) as a root user

    `nvpmodel -q` does report the mode without elevation, so it stays unelevated.
    """
    source = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    host_check = source.split("check_jetson_host() {", 1)[1].split("\n}", 1)[0]

    for command in ("nvpmodel -m 0", "jetson_clocks\n", "jetson_clocks --show"):
        line = next(
            entry
            for entry in host_check.splitlines()
            if command.strip() in entry and "require_command" not in entry
        )
        assert "run_privileged" in line, f"{command.strip()} is not elevated: {line.strip()}"


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


def test_a6000_deploy_rejects_a_stale_asr_memory_override_before_start() -> None:
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    deploy_body = deploy_script.split("deploy_a6000() {", 1)[1].split(
        "remove_project_image() {", 1
    )[0]

    assert "minimum_gpu_memory_utilization=0.28" in deploy_script
    assert "load_settings().asr.vllm_gpu_memory_utilization" in deploy_script
    assert "update config/remote-server.local.yaml" in deploy_script
    validation_call = (
        'verify_a6000_asr_configuration "$compose_file" "$environment_file"'
    )
    assert validation_call in deploy_body
    assert deploy_body.index(validation_call) < deploy_body.index("up -d")


def test_jetson_deploy_rejects_stale_asr_overrides_before_start() -> None:
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    deploy_body = deploy_script.split("deploy_jetson() {", 1)[1].split(
        "deploy_a6000() {", 1
    )[0]
    primary_call = 'verify_jetson_asr_configuration "$compose_file"'
    verification_call = 'verify_jetson_asr_verification_runtime "$compose_file"'

    # Assert the required values and the diagnostics, not the probe's internals.
    assert '"/models/Qwen3-ASR-0.6B"' in deploy_script
    assert '"/models/Qwen3-ASR-0.6B-hf"' in deploy_script
    assert '"qwen3_asr"' in deploy_script
    assert "Stale asr overrides in config/local.yaml" in deploy_script
    assert "Stale asr_verify overrides in config/local.yaml" in deploy_script
    assert '("asr_verify.device", "cpu"' in deploy_script
    assert '("asr_verify.compute_type", "int8"' in deploy_script
    assert "get_supported_compute_types(" in deploy_script

    # Every mismatch is reported in one run. Exiting on the first stale key made an
    # operator repeat an image build and container start once per key.
    for probe_marker in ("Stale asr overrides", "Stale asr_verify overrides"):
        probe = deploy_script.split(probe_marker, 1)[0].rsplit("-c \\\n", 1)[1]
        assert "sys.exit(1)" in deploy_script.split(probe_marker, 1)[1]
        assert probe.count("sys.exit(") == 0, f"{probe_marker} probe exits early"
    for validation_call in (primary_call, verification_call):
        assert validation_call in deploy_body
        assert deploy_body.index(validation_call) < deploy_body.index("up -d")


def test_deploy_builds_and_validates_the_llm_image_before_start() -> None:
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    jetson_body = deploy_script.split("deploy_jetson() {", 1)[1].split(
        "deploy_a6000() {", 1
    )[0]
    a6000_body = deploy_script.split("deploy_a6000() {", 1)[1].split(
        "remove_project_image() {", 1
    )[0]
    validation_call = "verify_vllm_translation_runtime"

    assert "build asr asr-verify llm tts orchestrator" in jetson_body
    assert 'build asr asr-verify llm tts' in a6000_body
    for deploy_body in (jetson_body, a6000_body):
        assert validation_call in deploy_body
        assert deploy_body.index("build asr asr-verify llm tts") < deploy_body.index(
            validation_call
        )
        assert deploy_body.index(validation_call) < deploy_body.index("up -d")


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
    assert '--reinstall-package numpy "numpy>=2,<2.3"' not in asr_stage
    final_sync = asr_stage.rindex("uv sync --active")
    dependency_check = asr_stage.index('uv pip check --python "$UV_PYTHON"')
    assert final_sync < dependency_check
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
    assert '--reinstall-package numpy "numpy>=2,<2.3"' not in remote_llm_stage
    remote_llm_final_sync = remote_llm_stage.rindex("uv sync --active")
    remote_llm_dependency_check = remote_llm_stage.index(
        'uv pip check --python "$UV_PYTHON"'
    )
    assert remote_llm_final_sync < remote_llm_dependency_check
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
    assert "from transformers import GenerationMixin" in deploy_script
    assert "from vllm.engine.arg_utils import AsyncEngineArgs" in deploy_script
    assert "build_async_engine_client_from_engine_args" in deploy_script
    assert "create_engine_config()" in deploy_script
    assert "model_config.hf_text_config" in deploy_script
    assert 'nested_keys = {"full_attention", "sliding_attention"}' in deploy_script
    assert "Path(sys.executable).is_relative_to" in deploy_script
    assert 'verify_vllm_translation_runtime "$compose_file" "$environment_file"' in (
        deploy_script
    )
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


def test_jetson_llm_preserves_translategemma_nested_rope_configuration() -> None:
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.llm").read_text(
        encoding="utf-8"
    )
    patch_name = "vllm-0.19.0-translategemma-nested-rope.patch"
    patch_source = (
        PROJECT_ROOT / "docker" / "patches" / patch_name
    ).read_text(encoding="utf-8")

    assert patch_name in dockerfile
    assert 'test "$vllm_version" = "0.19.0"' in dockerfile
    assert 'patch --batch --forward --fuzz=0 "$vllm_configuration"' in dockerfile
    assert '/opt/venv/bin/python -m py_compile "$vllm_configuration"' in dockerfile
    guard = (
        "if rope_parameters is not None "
        "and not is_rope_parameters_nested(rope_parameters):"
    )
    assert "grep -Fq" in dockerfile
    assert f'"{guard}"' in dockerfile
    assert f"+        {guard}" in patch_source
    assert patch_source.count('+                rope_parameters["') == 3

    # Gemma3Config is a composite configuration with no top-level rope fields. The
    # first version of this patch read config.rope_parameters unconditionally,
    # where the upstream lines it replaced only touched the attribute inside
    # `if rope_theta is not None`, so vLLM raised AttributeError while loading
    # TranslateGemma. The read stays guarded.
    added = [line for line in patch_source.splitlines() if line.startswith("+")]
    assert any('getattr(config, "rope_parameters", None)' in line for line in added)
    assert not any("is_rope_parameters_nested(config.rope_parameters)" in line for line in added)
    assert "full_attention" not in patch_source
    assert "sliding_attention" not in patch_source
    assert '"rope_type": "linear"' not in patch_source


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
    assert 'name = "numpy"\nversion = "2.2.6"' in lock_text
    assert "numpy>=2,<2.3 ; sys_platform == 'linux'" in project_configuration
    assert "numpy>=1.24,<2 ; sys_platform != 'linux'" in project_configuration
    assert '"mistral-common[audio]>=1.11.3"' in project_configuration
    assert (
        '"nvidia-cufft>=12,<13 ; platform_machine == \'x86_64\' and '
        'sys_platform == \'linux\'"'
    ) in project_configuration
    assert "UV_PYTHON=/opt/conda/bin/python" in dockerfile
    assert "UV_PYTHON=/opt/kotonoha-venv/bin/python" in dockerfile
    assert "--extra a6000-runtime" not in dockerfile
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
        if editable_install in dockerfile:
            project_install = dockerfile.index(editable_install)
        else:
            project_install = dockerfile.rindex("uv sync --active")
        assert dockerfile.index(required_copy) < project_install

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
    # The lock is the single source of the NumPy version. A forced reinstall here
    # contradicts the project metadata, which is what `uv pip check` then reports.
    assert "--reinstall-package numpy" not in dockerfile
    final_sync = dockerfile.rindex("uv sync --active")
    dependency_check = dockerfile.index('uv pip check --python "$UV_PYTHON"')
    assert final_sync < dependency_check
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
