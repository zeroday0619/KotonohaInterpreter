"""Deployment resource allocation interfaces."""

from kotonoha.deployment._gpu_allocator import (
    GpuAllocationError,
    GpuDevice,
    RoleReservation,
    allocate_roles,
    main,
    parse_gpu_inventory,
    read_environment_file,
)

__all__ = (
    "GpuAllocationError",
    "GpuDevice",
    "RoleReservation",
    "allocate_roles",
    "main",
    "parse_gpu_inventory",
    "read_environment_file",
)
