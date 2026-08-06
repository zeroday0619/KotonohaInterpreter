"""Memory-aware GPU allocation for the remote model-service stack."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from kotonoha.deployment import (
    GpuAllocationError,
    RoleReservation,
    allocate_roles,
    parse_gpu_inventory,
    read_environment_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOCATOR_SCRIPT = PROJECT_ROOT / "scripts" / "py" / "allocate_gpus.py"
REMOTE_COMPOSE = PROJECT_ROOT / "docker" / "compose.remote.yaml"


def _devices(
    free_memory_mib: int = 49_140,
    /,
) -> tuple:
    return parse_gpu_inventory(
        "0, GPU-0000, NVIDIA RTX A6000, 49140, "
        f"{free_memory_mib}\n"
        "1, GPU-1111, NVIDIA RTX A6000, 49140, "
        f"{free_memory_mib}\n"
    )


def _reservations() -> tuple[RoleReservation, ...]:
    return (
        RoleReservation("llm", "LLM_GPU_DEVICE", 27_648),
        RoleReservation("asr", "ASR_GPU_DEVICE", 14_336),
        RoleReservation("asr_verify", "ASR_VERIFY_GPU_DEVICE", 6_144),
        RoleReservation("tts", "TTS_GPU_DEVICE", 3_072),
    )


def test_allocator_spreads_roles_by_projected_memory_utilization() -> None:
    assignments = allocate_roles(
        _devices(),
        _reservations(),
        reserve_memory_mib=1_024,
    )

    assert assignments["llm"] == "GPU-0000"
    assert assignments["asr"] == "GPU-1111"
    assert assignments["asr_verify"] == "GPU-1111"
    assert assignments["tts"] == "GPU-1111"


def test_allocator_honors_manual_device_assignments() -> None:
    reservations = tuple(
        RoleReservation(
            reservation.role,
            reservation.environment_key,
            reservation.memory_mib,
            "0" if reservation.role == "llm" else "1",
        )
        for reservation in _reservations()
    )

    assignments = allocate_roles(
        _devices(),
        reservations,
        reserve_memory_mib=1_024,
        use_total_memory=True,
    )

    assert assignments == {
        "llm": "GPU-0000",
        "asr": "GPU-1111",
        "asr_verify": "GPU-1111",
        "tts": "GPU-1111",
    }


def test_allocator_rejects_single_a6000_for_current_role_budgets() -> None:
    reservations = tuple(
        RoleReservation(
            reservation.role,
            reservation.environment_key,
            reservation.memory_mib,
            "0",
        )
        for reservation in _reservations()
    )

    with pytest.raises(GpuAllocationError, match="cannot reserve"):
        allocate_roles(
            _devices()[:1],
            reservations,
            reserve_memory_mib=1_024,
            use_total_memory=True,
        )


def test_allocator_rejects_unsatisfied_memory_reservations() -> None:
    with pytest.raises(GpuAllocationError, match="no GPU can reserve"):
        allocate_roles(
            _devices(8_000),
            _reservations(),
            reserve_memory_mib=1_024,
        )


def test_allocator_reuses_stable_uuid_assignments(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    inventory_path = tmp_path / "inventory.csv"
    output_path = tmp_path / "remote-gpu.env"
    inventory_path.write_text(
        "0, GPU-0000, NVIDIA RTX A6000, 49140, 49140\n"
        "1, GPU-1111, NVIDIA RTX A6000, 49140, 49140\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(ALLOCATOR_SCRIPT),
        "--inventory-file",
        str(inventory_path),
        "--output",
        str(output_path),
    ]

    first = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stderr
    first_assignments = read_environment_file(output_path)

    inventory_path.write_text(
        "0, GPU-0000, NVIDIA RTX A6000, 49140, 1000\n"
        "1, GPU-1111, NVIDIA RTX A6000, 49140, 1000\n",
        encoding="utf-8",
    )
    second = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert second.returncode == 0, second.stderr
    assert read_environment_file(output_path) == first_assignments
    assert "cached stable assignment" in second.stdout
    assert output_path.stat().st_mode & 0o777 == 0o600


def test_remote_compose_pins_each_role_to_its_allocated_gpu() -> None:
    compose = yaml.safe_load(REMOTE_COMPOSE.read_text(encoding="utf-8"))
    expected_variables = {
        "asr": "${ASR_GPU_DEVICE:-0}",
        "asr-verify": "${ASR_VERIFY_GPU_DEVICE:-0}",
        "llm": "${LLM_GPU_DEVICE:-0}",
        "tts": "${TTS_GPU_DEVICE:-0}",
    }

    for role, expected_variable in expected_variables.items():
        devices = compose["services"][role]["deploy"]["resources"]["reservations"][
            "devices"
        ]
        assert devices == [
            {
                "driver": "nvidia",
                "device_ids": [expected_variable],
                "capabilities": ["gpu"],
            }
        ]
        assert "count" not in devices[0]
