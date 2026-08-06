"""Cross-platform system and accelerator resource reporting contracts."""

from __future__ import annotations

import sys

import pytest

from kotonoha.services import _resources


def test_resource_report_contains_system_kernel_and_accelerator_information() -> None:
    report = _resources.resource_report("asr", gpu_memory_utilization=0.15)

    assert report["role"] == "asr"
    assert report["system"]["os"]["machine"]
    assert report["system"]["kernel"]["release"]
    assert report["system"]["accelerator"]["backend"] in {
        "cpu",
        "cuda",
        "rocm",
        "mps",
        "xpu",
    }
    assert report["system"]["memory"]["architecture"] in {
        "unified",
        "discrete",
        "unknown",
    }


def test_jetson_device_tree_is_reported_as_unified_memory() -> None:
    architecture = _resources._memory_architecture(
        "cuda",
        ("Orin",),
        device_tree_model="NVIDIA Jetson AGX Orin",
    )

    assert architecture == "unified"


def test_mps_is_reported_as_unified_memory() -> None:
    architecture = _resources._memory_architecture("mps", ("Apple GPU",))

    assert architecture == "unified"


def test_resource_detection_falls_back_to_cpu_without_torch(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    accelerator = _resources._torch_accelerator()

    assert accelerator["backend"] == "cpu"
    assert accelerator["device_count"] == 0
