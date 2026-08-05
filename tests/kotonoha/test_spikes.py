"""Hardware spike target selection and reporting contracts."""

from __future__ import annotations

import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASR_DOCKERFILE = PROJECT_ROOT / "docker" / "Dockerfile.asr"
PERFORMANCE_DOCUMENT = PROJECT_ROOT / "docs" / "performance" / "measurement.md"
FETCH_MODELS_SCRIPT = PROJECT_ROOT / "scripts" / "fetch_models.sh"
REPORT_SCRIPT = PROJECT_ROOT / "spikes" / "report.py"
RUNNER_SCRIPT = PROJECT_ROOT / "spikes" / "run_all.sh"
SPIKE_README = PROJECT_ROOT / "spikes" / "README.md"
SPIKE1_SCRIPT = PROJECT_ROOT / "spikes" / "spike1_asr_load.py"
SPIKE2_SCRIPT = PROJECT_ROOT / "spikes" / "spike2_flash_attn.py"
SPIKE3_SCRIPT = PROJECT_ROOT / "spikes" / "spike3_llm_tokrate.py"
SPIKE_COMPOSE = PROJECT_ROOT / "docker" / "compose.spikes.yaml"
SPIKE_ENTRYPOINT = PROJECT_ROOT / "spikes" / "container_entrypoint.sh"
PROJECT_CONFIGURATION = PROJECT_ROOT / "pyproject.toml"


def test_a6000_report_generates_remote_configuration_patch(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    result = {
        "spike": 1,
        "target": "a6000",
        "audio": {"source": "probe.wav", "seconds": 6.0},
        "env": {"device": "RTX A6000", "capability": "8.6", "torch": "test"},
        "conditions": {
            "gpu_memory_utilization": 0.85,
            "max_model_len": 4096,
            "enforce_eager": False,
        },
        "vllm": {
            "loaded": True,
            "nbest_ok": True,
            "has_logprobs": True,
            "nbest_ms": 320.0,
        },
        "transformers": {"skipped": True},
        "verdict": {
            "recommended_backend": "vllm",
            "nbest_ms": 320.0,
            "note": "measured",
        },
    }
    (result_directory / "spike1.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )
    report_path = result_directory / "PERFORMANCE.md"
    patch_path = result_directory / "remote-server.local.yaml"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--target",
            "a6000",
            "--dir",
            str(result_directory),
            "--md",
            str(report_path),
            "--patch",
            str(patch_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "A6000" in report_path.read_text(encoding="utf-8")
    patch = patch_path.read_text(encoding="utf-8")
    assert "backend: vllm" in patch
    assert "vllm_gpu_memory_utilization: 0.85" in patch
    assert "vllm_enforce_eager: false" in patch


def test_spike_runner_keeps_target_outputs_separate() -> None:
    source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    spike3_source = SPIKE3_SCRIPT.read_text(encoding="utf-8")

    assert "jetson|a6000" in source
    assert "spikes/out/a6000" in source
    assert "PERFORMANCE.md" in source
    assert "remote-server.local.yaml" in source
    assert "docker/compose.spikes.yaml" in source
    assert 'run --rm asr' in source
    assert 'run --rm tts' in source
    assert 'run --rm llm' in source
    assert 'run --rm report' in source
    assert "--vllm-command" in source
    assert "Qwen/Qwen3-14B-AWQ" in spike3_source
    assert "ELVISIO/Qwen3-30B-A3B-Instruct-2507-AWQ" in spike3_source
    assert '"stream_options": {"include_usage": True}' in spike3_source
    assert "llama-server" not in spike3_source


def test_spike_runner_preserves_compose_variables_through_sudo() -> None:
    source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    compose_source = SPIKE_COMPOSE.read_text(encoding="utf-8")
    compose_variables = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)", compose_source))
    forwarded_variables = set(
        re.findall(r'^\s+"([A-Z][A-Z0-9_]*)=\$\1"$', source, flags=re.MULTILINE)
    )

    assert 'compose_command=(sudo env "${compose_environment[@]}" docker compose)' in source
    assert compose_variables <= forwarded_variables


def test_spike_runner_rejects_incomplete_model_snapshots(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        ["bash", str(RUNNER_SCRIPT), "jetson", "--only", "1"],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "ASR_ONLY": "vllm",
            "MODELS_DIR": str(tmp_path),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "Required model snapshot is missing" in completed.stderr
    assert "Qwen3-ASR-1.7B/config.json" in completed.stderr


def test_spike_runner_stops_after_asr_image_build_failure(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    tool_directory = tmp_path / "tools"
    tool_directory.mkdir()
    docker_command = tool_directory / "docker"
    docker_command.write_text(
        """#!/bin/sh
if [ "$1" = build ]; then
  exit 7
fi
exit 0
""",
        encoding="utf-8",
    )
    docker_command.chmod(0o755)
    models_directory = tmp_path / "models"
    for model_directory in (
        "Qwen3-ASR-1.7B",
        "Qwen3-TTS-0.6B",
        "llm/Qwen3-14B-AWQ",
        "llm/Qwen3-30B-A3B-Instruct-2507-AWQ",
    ):
        snapshot_directory = models_directory / model_directory
        snapshot_directory.mkdir(parents=True)
        (snapshot_directory / "config.json").write_text("{}", encoding="utf-8")

    completed = subprocess.run(
        ["bash", str(RUNNER_SCRIPT), "a6000"],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "MODELS_DIR": str(models_directory),
            "OUT": "spikes/out/test-build-failure",
            "PATH": f"{tool_directory}:{os.environ['PATH']}",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "ASR spike image build failed" in completed.stderr
    assert "Building TTS spike image" not in completed.stdout


def test_transformers_probe_uses_the_isolated_worker(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = tmp_path / "transformers-worker"
    worker.write_text(
        """#!/bin/sh
cat >/dev/null
printf '%s\n' 'KOTONOHA_TRANSFORMERS_RESULT={"backend":"transformers","loaded":true}'
""",
        encoding="utf-8",
    )
    worker.chmod(0o755)
    monkeypatch.setenv("SPIKE_TRANSFORMERS_PYTHON", str(worker))
    run_transformers = runpy.run_path(str(SPIKE1_SCRIPT))["run_transformers"]

    result = run_transformers(np.zeros(16000, dtype=np.float32), 1, "test-model")

    assert result == {"backend": "transformers", "loaded": True}


def test_vllm_omni_probe_uses_the_deployment_speech_api_contract() -> None:
    namespace = runpy.run_path(str(SPIKE2_SCRIPT))
    command = namespace["server_command"](
        "/models/Qwen3-TTS-0.6B",
        port=18004,
        gpu_memory_utilization=0.25,
        enforce_eager=True,
    )

    assert command[:4] == ["vllm", "serve", "/models/Qwen3-TTS-0.6B", "--omni"]
    assert "--served-model-name" in command
    assert "--stage-overrides" in command
    assert "--enforce-eager" in command


def test_performance_document_owns_measurement_procedure() -> None:
    performance_document = PERFORMANCE_DOCUMENT.read_text(encoding="utf-8")
    spike_readme = SPIKE_README.read_text(encoding="utf-8")

    assert "## ASR Measurement" in performance_document
    assert "## Link Measurement" in performance_document
    assert "900 ms or less" in performance_document
    assert "../docs/performance/measurement.md" in spike_readme
    assert "## ASR Measurement" not in spike_readme


def test_hardware_spikes_use_target_specific_docker_images() -> None:
    asr_dockerfile = ASR_DOCKERFILE.read_text(encoding="utf-8")
    project_configuration = PROJECT_CONFIGURATION.read_text(encoding="utf-8")
    performance_document = PERFORMANCE_DOCUMENT.read_text(encoding="utf-8")
    fetch_models_source = FETCH_MODELS_SCRIPT.read_text(encoding="utf-8")
    spike2_source = SPIKE2_SCRIPT.read_text(encoding="utf-8")
    runner_source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    compose_source = SPIKE_COMPOSE.read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_source)

    assert "spike-vllm" not in project_configuration
    assert set(compose["services"]) == {"asr", "tts", "llm", "report"}
    assert all(service["runtime"] == "nvidia" for service in compose["services"].values())
    assert all(service["user"] == "0:0" for service in compose["services"].values())
    assert compose["services"]["asr"]["entrypoint"][-1] == "spikes/spike1_asr_load.py"
    assert compose["services"]["tts"]["entrypoint"][-1] == "spikes/spike2_flash_attn.py"
    assert all(
        service["entrypoint"][:2] == ["bash", "spikes/container_entrypoint.sh"]
        for service in compose["services"].values()
    )
    assert all("SPIKE_PYTHON" in service["environment"] for service in compose["services"].values())
    assert "SPIKE_ASR_IMAGE" in compose["services"]["asr"]["image"]
    assert 'build_asr_image()' in runner_source
    assert '--file docker/Dockerfile.asr' in runner_source
    assert 'SPIKE_ASR_IMAGE=$SPIKE_ASR_IMAGE' in runner_source
    assert 'SPIKE_PYTHON:=/opt/venv/bin/python' in runner_source
    assert 'SPIKE_PYTHON:=python3' in runner_source
    assert 'UV_PYTHON=/opt/venv/bin/python' in asr_dockerfile
    assert 'uv pip install --python "$UV_PYTHON"' in asr_dockerfile
    assert "r36.4.tegra-aarch64-cu126-22.04" in compose_source
    assert "nvcr.io/nvidia/vllm:26.07-py3" in runner_source
    assert "vllm/vllm-omni:v0.26.0" in runner_source
    assert "prepare_tts_image()" in runner_source
    assert 'pull "$SPIKE_TTS_IMAGE"' in runner_source
    assert "SPIKE_TTS_PYTHON=$SPIKE_TTS_PYTHON" in runner_source
    assert "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" in fetch_models_source
    assert '"/v1/audio/speech"' in spike2_source
    assert '"stream_format": "audio"' in spike2_source
    assert "probe_flash_attention" in spike2_source
    assert "bash scripts/manage.sh benchmark a6000 --only 1" in performance_document


def test_spike_entrypoint_restores_output_ownership_contract() -> None:
    source = SPIKE_ENTRYPOINT.read_text(encoding="utf-8")

    assert "--out|--md|--patch" in source
    assert 'chown "${SPIKE_USER_ID}:${SPIKE_GROUP_ID}" "$output_directory"' in source
    assert 'chown "${SPIKE_USER_ID}:${SPIKE_GROUP_ID}" "$output_path"' in source
    assert '"$SPIKE_PYTHON" "$@"' in source
    assert 'exit "$command_status"' in source


def test_tts_runtime_is_not_installed_into_the_project_environment() -> None:
    project_configuration = PROJECT_CONFIGURATION.read_text(encoding="utf-8")
    compose_source = SPIKE_COMPOSE.read_text(encoding="utf-8")

    assert "vllm-omni" not in project_configuration
    assert "qwen-tts" not in project_configuration
    assert "melotts" not in project_configuration
    assert "vllm/vllm-omni:v0.26.0" in compose_source
