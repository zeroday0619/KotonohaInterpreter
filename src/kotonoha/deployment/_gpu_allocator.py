"""Assign remote model services to NVIDIA GPUs by available memory."""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import ClassVar, Final


@dataclass(frozen=True, slots=True)
class GpuDevice:
    """Represent one physical GPU reported by ``nvidia-smi``."""

    index: str
    uuid: str
    name: str
    total_memory_mib: int
    free_memory_mib: int


@dataclass(frozen=True, slots=True)
class RoleReservation:
    """Describe a model-service GPU memory reservation."""

    role: str
    environment_key: str
    memory_mib: int
    requested_device: str | None = None


class GpuAllocationError(RuntimeError):
    """Report an invalid inventory or an unsatisfied reservation."""

    __slots__: ClassVar[tuple[str, ...]] = ()


ROLE_DEFINITIONS: Final = (
    ("llm", "LLM_GPU_DEVICE", "LLM_GPU_MEMORY_MIB", 27_648),
    ("asr", "ASR_GPU_DEVICE", "ASR_GPU_MEMORY_MIB", 14_336),
    ("asr_verify", "ASR_VERIFY_GPU_DEVICE", "ASR_VERIFY_GPU_MEMORY_MIB", 6_144),
    ("tts", "TTS_GPU_DEVICE", "TTS_GPU_MEMORY_MIB", 3_072),
)
DEFAULT_MEMORY_RESERVE_MIB: Final = 1_024
DEFAULT_GPU_NAME_FILTER: Final = "A6000"
ALLOCATION_MODES: Final = frozenset({"auto", "manual"})


def parse_gpu_inventory(
    output: str,
    /,
) -> tuple[GpuDevice, ...]:
    """Parse the stable CSV form produced by ``nvidia-smi --query-gpu``."""
    devices: list[GpuDevice] = []
    for line_number, row in enumerate(csv.reader(io.StringIO(output)), start=1):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 5:
            raise GpuAllocationError(
                f"invalid nvidia-smi row {line_number}: expected 5 columns, received {len(row)}"
            )
        index, uuid, name, total_memory, free_memory = (
            value.strip() for value in row
        )
        try:
            total_memory_mib = int(total_memory)
            free_memory_mib = int(free_memory)
        except ValueError as error:
            raise GpuAllocationError(
                f"invalid memory value in nvidia-smi row {line_number}"
            ) from error
        if total_memory_mib <= 0 or not 0 <= free_memory_mib <= total_memory_mib:
            raise GpuAllocationError(
                f"invalid memory capacity in nvidia-smi row {line_number}"
            )
        devices.append(
            GpuDevice(
                index=index,
                uuid=uuid,
                name=name,
                total_memory_mib=total_memory_mib,
                free_memory_mib=free_memory_mib,
            )
        )
    if not devices:
        raise GpuAllocationError("nvidia-smi did not report any GPUs")
    if len({device.uuid for device in devices}) != len(devices):
        raise GpuAllocationError("nvidia-smi reported duplicate GPU UUIDs")
    return tuple(devices)


def query_gpu_inventory(
    nvidia_smi_command: str,
    /,
) -> tuple[GpuDevice, ...]:
    command = [
        nvidia_smi_command,
        "--query-gpu=index,uuid,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise GpuAllocationError(f"nvidia-smi inventory query failed: {error}") from error
    return parse_gpu_inventory(completed.stdout)


def filter_gpu_inventory(
    devices: tuple[GpuDevice, ...],
    /,
    *,
    name_filter: str,
) -> tuple[GpuDevice, ...]:
    if not name_filter:
        return devices
    selected = tuple(
        device for device in devices if name_filter.casefold() in device.name.casefold()
    )
    if not selected:
        raise GpuAllocationError(
            f"no GPU name contains the configured filter: {name_filter}"
        )
    return selected


def _resolve_device(
    devices: tuple[GpuDevice, ...],
    /,
    *,
    identifier: str,
) -> GpuDevice:
    matches = tuple(
        device
        for device in devices
        if identifier in {device.index, device.uuid}
    )
    if len(matches) != 1:
        raise GpuAllocationError(f"GPU device is not available: {identifier}")
    return matches[0]


def _placement_order(
    device: GpuDevice,
    /,
    *,
    memory_mib: int,
    capacities: dict[str, int],
    initial_capacities: dict[str, int],
) -> tuple[float, int, str]:
    initial_capacity = initial_capacities[device.uuid]
    projected_used = initial_capacity - capacities[device.uuid] + memory_mib
    projected_utilization = (
        projected_used / initial_capacity if initial_capacity else 1.0
    )
    projected_remaining = capacities[device.uuid] - memory_mib
    return projected_utilization, -projected_remaining, device.uuid


def allocate_roles(
    devices: tuple[GpuDevice, ...],
    reservations: tuple[RoleReservation, ...],
    /,
    *,
    reserve_memory_mib: int,
    use_total_memory: bool = False,
) -> dict[str, str]:
    """Allocate roles while minimizing the highest projected GPU memory utilization."""
    if reserve_memory_mib < 0:
        raise GpuAllocationError("GPU memory reserve must not be negative")
    if len({reservation.role for reservation in reservations}) != len(reservations):
        raise GpuAllocationError("role reservations contain duplicate names")

    capacities: dict[str, int] = {}
    initial_capacities: dict[str, int] = {}
    by_uuid = {device.uuid: device for device in devices}
    for device in devices:
        reported_memory = (
            device.total_memory_mib if use_total_memory else device.free_memory_mib
        )
        capacity = max(0, reported_memory - reserve_memory_mib)
        capacities[device.uuid] = capacity
        initial_capacities[device.uuid] = capacity

    assignments: dict[str, str] = {}
    fixed = tuple(
        reservation
        for reservation in reservations
        if reservation.requested_device is not None
    )
    automatic = tuple(
        reservation
        for reservation in reservations
        if reservation.requested_device is None
    )
    for reservation in fixed:
        if reservation.memory_mib <= 0:
            raise GpuAllocationError(
                f"GPU memory reservation must be positive: {reservation.role}"
            )
        device = _resolve_device(
            devices,
            identifier=reservation.requested_device or "",
        )
        if capacities[device.uuid] < reservation.memory_mib:
            raise GpuAllocationError(
                f"GPU {device.uuid} cannot reserve {reservation.memory_mib} MiB "
                f"for {reservation.role}; available capacity is {capacities[device.uuid]} MiB"
            )
        capacities[device.uuid] -= reservation.memory_mib
        assignments[reservation.role] = device.uuid

    for reservation in sorted(
        automatic,
        key=lambda candidate: (-candidate.memory_mib, candidate.role),
    ):
        if reservation.memory_mib <= 0:
            raise GpuAllocationError(
                f"GPU memory reservation must be positive: {reservation.role}"
            )
        candidates = tuple(
            device
            for device in devices
            if capacities[device.uuid] >= reservation.memory_mib
        )
        if not candidates:
            details = ", ".join(
                f"{device.uuid}={capacities[device.uuid]} MiB"
                for device in devices
            )
            raise GpuAllocationError(
                f"no GPU can reserve {reservation.memory_mib} MiB for "
                f"{reservation.role}; remaining capacities: {details}"
            )

        selected = min(
            candidates,
            key=partial(
                _placement_order,
                memory_mib=reservation.memory_mib,
                capacities=capacities,
                initial_capacities=initial_capacities,
            ),
        )
        capacities[selected.uuid] -= reservation.memory_mib
        assignments[reservation.role] = selected.uuid

    if set(assignments) != {reservation.role for reservation in reservations}:
        raise GpuAllocationError("not every role received a GPU assignment")
    if any(device_uuid not in by_uuid for device_uuid in assignments.values()):
        raise GpuAllocationError("allocation produced an unknown GPU UUID")
    return assignments


def read_environment_file(
    path: Path | None,
    /,
) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise GpuAllocationError(
                f"invalid environment assignment in {path}:{line_number}"
            )
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _environment_value(
    key: str,
    /,
    *,
    file_values: dict[str, str],
    default: str | None = None,
) -> str | None:
    return os.environ.get(key, file_values.get(key, default))


def build_reservations(
    file_values: dict[str, str],
    /,
    *,
    mode: str,
    cached_values: dict[str, str],
) -> tuple[RoleReservation, ...]:
    reservations: list[RoleReservation] = []
    for role, device_key, memory_key, default_memory in ROLE_DEFINITIONS:
        raw_memory = _environment_value(
            memory_key,
            file_values=file_values,
            default=str(default_memory),
        )
        try:
            memory_mib = int(raw_memory or "")
        except ValueError as error:
            raise GpuAllocationError(
                f"{memory_key} must be a positive integer"
            ) from error

        requested_device: str | None = None
        if mode == "manual":
            requested_device = _environment_value(
                device_key,
                file_values=file_values,
            )
            if not requested_device:
                raise GpuAllocationError(
                    f"{device_key} is required when GPU_ALLOCATION_MODE=manual"
                )
        elif cached_values:
            requested_device = cached_values.get(device_key)
            if not requested_device:
                raise GpuAllocationError(
                    f"cached GPU allocation is missing {device_key}"
                )
        reservations.append(
            RoleReservation(
                role=role,
                environment_key=device_key,
                memory_mib=memory_mib,
                requested_device=requested_device,
            )
        )
    return tuple(reservations)


def write_allocation(
    path: Path,
    /,
    *,
    devices: tuple[GpuDevice, ...],
    reservations: tuple[RoleReservation, ...],
    assignments: dict[str, str],
    source: str,
) -> None:
    by_uuid = {device.uuid: device for device in devices}
    lines = [
        "# Generated by scripts/allocate_gpus.py. Do not edit manually.",
        f"# Allocation source: {source}",
    ]
    for reservation in reservations:
        device = by_uuid[assignments[reservation.role]]
        lines.append(
            f"# {reservation.role}: GPU {device.index}, {device.name}, "
            f"{reservation.memory_mib} MiB reserved, "
            f"{device.free_memory_mib}/{device.total_memory_mib} MiB free"
        )
        lines.append(f"{reservation.environment_key}={device.uuid}")
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary_file:
        temporary_file.write(content)
        temporary_path = Path(temporary_file.name)
    temporary_path.chmod(0o600)
    temporary_path.replace(path)


def print_allocation(
    devices: tuple[GpuDevice, ...],
    reservations: tuple[RoleReservation, ...],
    assignments: dict[str, str],
    /,
    *,
    source: str,
) -> None:
    by_uuid = {device.uuid: device for device in devices}
    print(f"GPU allocation source: {source}")
    print("role         reservation  gpu  free / total MiB  uuid")
    for reservation in reservations:
        device = by_uuid[assignments[reservation.role]]
        print(
            f"{reservation.role:<12} {reservation.memory_mib:>7} MiB  "
            f"{device.index:>3}  {device.free_memory_mib:>6} / "
            f"{device.total_memory_mib:<6}  {device.uuid}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assign Kotonoha remote services to GPUs by memory capacity."
    )
    parser.add_argument("--environment-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-file", type=Path)
    parser.add_argument("--nvidia-smi-command", default="nvidia-smi")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    file_values = read_environment_file(arguments.environment_file)
    mode = _environment_value(
        "GPU_ALLOCATION_MODE",
        file_values=file_values,
        default="auto",
    )
    if mode not in ALLOCATION_MODES:
        raise GpuAllocationError("GPU_ALLOCATION_MODE must be auto or manual")
    raw_reserve = _environment_value(
        "GPU_MEMORY_RESERVE_MIB",
        file_values=file_values,
        default=str(DEFAULT_MEMORY_RESERVE_MIB),
    )
    try:
        reserve_memory_mib = int(raw_reserve or "")
    except ValueError as error:
        raise GpuAllocationError("GPU_MEMORY_RESERVE_MIB must be an integer") from error

    if arguments.inventory_file:
        devices = parse_gpu_inventory(
            arguments.inventory_file.read_text(encoding="utf-8")
        )
    else:
        devices = query_gpu_inventory(arguments.nvidia_smi_command)
    name_filter = _environment_value(
        "GPU_NAME_FILTER",
        file_values=file_values,
        default=DEFAULT_GPU_NAME_FILTER,
    )
    devices = filter_gpu_inventory(devices, name_filter=name_filter or "")

    cached_values = {}
    source = "automatic free-memory allocation"
    use_total_memory = False
    if mode == "manual":
        source = "manual environment assignment"
        use_total_memory = True
    elif arguments.output.exists() and not arguments.force:
        cached_values = read_environment_file(arguments.output)
        source = "cached stable assignment"
        use_total_memory = True

    reservations = build_reservations(
        file_values,
        mode=mode,
        cached_values=cached_values,
    )
    assignments = allocate_roles(
        devices,
        reservations,
        reserve_memory_mib=reserve_memory_mib,
        use_total_memory=use_total_memory,
    )
    write_allocation(
        arguments.output,
        devices=devices,
        reservations=reservations,
        assignments=assignments,
        source=source,
    )
    print_allocation(
        devices,
        reservations,
        assignments,
        source=source,
    )
    return 0
