"""Report system and accelerator resources across supported device families."""

from __future__ import annotations

import os
import platform

# macOS memory discovery requires the fixed system sysctl command and never invokes a shell.
import subprocess  # nosec B404
from functools import lru_cache
from pathlib import Path
from typing import Any


def _read_text(
    path: str,
    /,
) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return ""


def _device_tree_model() -> str:
    return _read_text("/proc/device-tree/model")


def _device_tree_compatible() -> str:
    return _read_text("/proc/device-tree/compatible")


def _memory_architecture(
    backend: str,
    device_names: tuple[str, ...],
    /,
    *,
    device_tree_model: str = "",
    device_tree_compatible: str = "",
) -> str:
    if os.environ.get("KOTONOHA_MEMORY_ARCHITECTURE") in {"unified", "discrete"}:
        return os.environ["KOTONOHA_MEMORY_ARCHITECTURE"]
    lower_names = " ".join(device_names).casefold()
    lower_tree = f"{device_tree_model} {device_tree_compatible}".casefold()
    if "tegra" in lower_tree or "jetson" in lower_names or "orin" in lower_names:
        return "unified"
    if backend == "mps":
        return "unified"
    if backend in {"cuda", "rocm", "xpu"}:
        return "discrete"
    return "unknown"


def _torch_accelerator() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "backend": "cpu",
            "framework": None,
            "device_count": 0,
            "devices": [],
        }

    try:
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            devices: list[dict[str, Any]] = []
            for index in range(torch.cuda.device_count()):
                name = str(torch.cuda.get_device_name(index))
                device: dict[str, Any] = {"index": index, "name": name}
                try:
                    properties = torch.cuda.get_device_properties(index)
                    device["total_memory_mib"] = round(properties.total_memory / 1024**2, 1)
                    device["architecture"] = (
                        f"sm_{properties.major}{properties.minor}"
                        if backend == "cuda"
                        else None
                    )
                except (AttributeError, RuntimeError):
                    device["total_memory_mib"] = None
                    device["architecture"] = None
                devices.append(device)
            return {
                "backend": backend,
                "framework": "torch",
                "device_count": len(devices),
                "devices": devices,
            }
    except (AttributeError, RuntimeError):
        pass

    try:
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return {
                "backend": "mps",
                "framework": "torch",
                "device_count": 1,
                "devices": [{"index": 0, "name": "Apple GPU"}],
            }
    except (AttributeError, RuntimeError):
        pass

    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            devices = [
                {"index": index, "name": str(xpu.get_device_name(index))}
                for index in range(xpu.device_count())
            ]
            return {
                "backend": "xpu",
                "framework": "torch",
                "device_count": len(devices),
                "devices": devices,
            }
    except (AttributeError, RuntimeError):
        pass

    return {
        "backend": "cpu",
        "framework": "torch",
        "device_count": 0,
        "devices": [],
    }


def _system_memory() -> dict[str, float | None]:
    values: dict[str, float | None] = {"total_mib": None, "available_mib": None}
    meminfo = _read_text("/proc/meminfo")
    if meminfo:
        for line in meminfo.splitlines():
            name, separator, value = line.partition(":")
            if separator and name in {"MemTotal", "MemAvailable"}:
                parts = value.strip().split()
                if parts and parts[0].isdigit():
                    values[
                        "total_mib" if name == "MemTotal" else "available_mib"
                    ] = round(int(parts[0]) / 1024, 1)
        return values

    if platform.system() == "Darwin":
        try:
            # The executable path and every argument are fixed application constants.
            completed = subprocess.run(  # nosec B603
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            values["total_mib"] = round(int(completed.stdout.strip()) / 1024**2, 1)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return values


def _accelerator_memory_snapshot(
    accelerator: dict[str, Any],
    /,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "cuda_available": accelerator["backend"] == "cuda",
        "accelerator_available": accelerator["device_count"] > 0,
        "allocated_mib": None,
        "reserved_mib": None,
        "max_reserved_mib": None,
        "free_mib": None,
        "total_mib": None,
    }
    if accelerator["backend"] not in {"cuda", "rocm"}:
        return snapshot
    try:
        import torch

        snapshot["allocated_mib"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
        snapshot["reserved_mib"] = round(torch.cuda.memory_reserved() / 1024**2, 1)
        snapshot["max_reserved_mib"] = round(
            torch.cuda.max_memory_reserved() / 1024**2,
            1,
        )
        free_memory, total_memory = torch.cuda.mem_get_info()
        snapshot["free_mib"] = round(free_memory / 1024**2, 1)
        snapshot["total_mib"] = round(total_memory / 1024**2, 1)
    except (ImportError, RuntimeError, AttributeError):
        return snapshot
    return snapshot


@lru_cache(maxsize=1)
def _stable_system_snapshot() -> dict[str, Any]:
    """Cache host and accelerator identity, which cannot change at runtime."""
    uname = platform.uname()
    accelerator = _torch_accelerator()
    device_names = tuple(
        str(device.get("name")) for device in accelerator["devices"] if device.get("name")
    )
    memory_architecture = _memory_architecture(
        accelerator["backend"],
        device_names,
        device_tree_model=_device_tree_model(),
        device_tree_compatible=_device_tree_compatible(),
    )
    return {
        "os": {
            "name": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "distribution": platform.platform(),
        },
        "kernel": {
            "name": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "accelerator": {
            **accelerator,
            "memory_architecture": memory_architecture,
        },
    }


def system_snapshot() -> dict[str, Any]:
    """Return stable host identity with a current available-memory sample."""
    stable = _stable_system_snapshot()
    memory_architecture = stable["accelerator"]["memory_architecture"]
    memory = _system_memory()
    memory["architecture"] = memory_architecture
    # These values always come from system RAM. Discrete accelerator memory is
    # reported separately by `_accelerator_memory_snapshot`.
    memory["scope"] = "system"
    return {**stable, "memory": memory}


def resource_report(
    role: str,
    /,
    *,
    gpu_memory_utilization: float | None = None,
    max_num_seqs: int | None = None,
    prefix_caching: bool | None = None,
) -> dict[str, Any]:
    """Combine configuration, live memory counters, and host system data."""
    system = system_snapshot()
    report: dict[str, Any] = {
        "role": role,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "prefix_caching": prefix_caching,
        "system": system,
    }
    report.update(_accelerator_memory_snapshot(system["accelerator"]))
    return report
