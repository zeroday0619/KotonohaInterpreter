"""Hardware spike target selection and reporting contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE_DOCUMENT = PROJECT_ROOT / "docs" / "performance" / "measurement.md"
REPORT_SCRIPT = PROJECT_ROOT / "spikes" / "report.py"
RUNNER_SCRIPT = PROJECT_ROOT / "spikes" / "run_all.sh"
SPIKE_README = PROJECT_ROOT / "spikes" / "README.md"


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

    assert "jetson|a6000" in source
    assert "spikes/out/a6000" in source
    assert "PERFORMANCE.md" in source
    assert "remote-server.local.yaml" in source


def test_performance_document_owns_measurement_procedure() -> None:
    performance_document = PERFORMANCE_DOCUMENT.read_text(encoding="utf-8")
    spike_readme = SPIKE_README.read_text(encoding="utf-8")

    assert "## ASR Measurement" in performance_document
    assert "## Link Measurement" in performance_document
    assert "900 ms or less" in performance_document
    assert "../docs/performance/measurement.md" in spike_readme
    assert "## ASR Measurement" not in spike_readme
