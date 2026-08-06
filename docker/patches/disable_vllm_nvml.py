#!/usr/bin/env python3
"""Add a Jetson-safe switch to vLLM CUDA platform detection."""

from __future__ import annotations

import sys
from pathlib import Path

START_MARKER = "nvml_available = False\ntry:\n"
SHUTDOWN_MARKER = "pynvml.nvmlShutdown()"
SWITCH_MARKER = "KOTONOHA_DISABLE_NVML"


def _vllm_cuda_files() -> tuple[Path, ...]:
    candidates: list[Path] = []
    for search_path in sys.path:
        if not search_path:
            continue
        candidate = Path(search_path) / "vllm" / "platforms" / "cuda.py"
        if candidate.is_file() and candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _patch_file(
    path: Path,
    /,
) -> bool:
    source = path.read_text(encoding="utf-8")
    if SWITCH_MARKER in source:
        return False

    start = source.find(START_MARKER)
    if start < 0:
        raise RuntimeError(f"vLLM NVML detection block not found: {path}")
    shutdown = source.find(SHUTDOWN_MARKER, start)
    if shutdown < 0:
        raise RuntimeError(f"vLLM NVML shutdown call not found: {path}")
    end = source.find("\n", shutdown)
    if end < 0:
        end = len(source)
    else:
        end += 1

    assignment_end = start + len("nvml_available = False\n")
    block = source[assignment_end:end]
    indented_block = "\n".join(
        f"    {line}" if line else line for line in block.splitlines()
    )
    replacement = (
        "nvml_available = False\n"
        'if os.environ.get("KOTONOHA_DISABLE_NVML") != "1":\n'
        f"{indented_block}\n"
    )
    path.write_text(source[:start] + replacement + source[end:], encoding="utf-8")
    return True


def main() -> int:
    files = _vllm_cuda_files()
    if not files:
        raise RuntimeError("vLLM CUDA platform source was not found")

    patched_count = sum(_patch_file(path) for path in files)
    for path in files:
        print(f"vLLM NVML platform source: {path}")
    print(f"vLLM NVML platform files patched: {patched_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
