"""UI-neutral history constants and export helpers."""

from __future__ import annotations

import json
from pathlib import Path

from kotonoha._secure_files import atomic_text_writer
from kotonoha.store._db import HistoryEntry

OUTCOMES = (
    "ok",
    "empty_asr",
    "asr_failed",
    "llm_timeout",
    "tts_failed",
    "oom",
    "aborted",
)


def export_jsonl(
    entries: list[HistoryEntry],
    /,
    path: Path,
) -> Path:
    """Write history entries as one JSON object per line."""
    with atomic_text_writer(path) as handle:
        for entry in entries:
            handle.write(f"{json.dumps(entry.as_dict(), ensure_ascii=False)}\n")
    return path
