#!/usr/bin/env python3
"""Run the memory-aware GPU allocator for remote model services."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

SOURCE_DIRECTORY = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

deployment_module = import_module("kotonoha.deployment")
GpuAllocationError = deployment_module.GpuAllocationError
main = deployment_module.main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GpuAllocationError as error:
        raise SystemExit(f"GPU allocation failed: {error}") from error
