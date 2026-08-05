"""Hardware spike target selection and reporting contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE_DOCUMENT = PROJECT_ROOT / "docs" / "performance" / "measurement.md"
REPORT_SCRIPT = PROJECT_ROOT / "spikes" / "report.py"
RUNNER_SCRIPT = PROJECT_ROOT / "spikes" / "run_all.sh"
SPIKE_README = PROJECT_ROOT / "spikes" / "README.md"
SPIKE3_SCRIPT = PROJECT_ROOT / "spikes" / "spike3_llm_tokrate.py"
SPIKE_COMPOSE = PROJECT_ROOT / "docker" / "compose.spikes.yaml"
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


def test_performance_document_owns_measurement_procedure() -> None:
    performance_document = PERFORMANCE_DOCUMENT.read_text(encoding="utf-8")
    spike_readme = SPIKE_README.read_text(encoding="utf-8")

    assert "## ASR Measurement" in performance_document
    assert "## Link Measurement" in performance_document
    assert "900 ms or less" in performance_document
    assert "../docs/performance/measurement.md" in spike_readme
    assert "## ASR Measurement" not in spike_readme


def test_hardware_spikes_use_target_specific_docker_images() -> None:
    project_configuration = PROJECT_CONFIGURATION.read_text(encoding="utf-8")
    performance_document = PERFORMANCE_DOCUMENT.read_text(encoding="utf-8")
    compose_source = SPIKE_COMPOSE.read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_source)

    assert "spike-vllm" not in project_configuration
    assert set(compose["services"]) == {"asr", "tts", "llm", "report"}
    assert all(service["runtime"] == "nvidia" for service in compose["services"].values())
    assert compose["services"]["asr"]["entrypoint"] == [
        "python3",
        "spikes/spike1_asr_load.py",
    ]
    assert compose["services"]["tts"]["entrypoint"] == [
        "python3",
        "spikes/spike2_flash_attn.py",
    ]
    assert "r38.2.arm64-sbsa-cu130-24.04" in compose_source
    assert "bash spikes/run_all.sh a6000 --only 1" in performance_document
