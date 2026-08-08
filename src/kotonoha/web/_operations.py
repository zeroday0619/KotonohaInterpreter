"""Bounded subprocess jobs exposed by the browser operations panel."""

from __future__ import annotations

import asyncio
import os
from asyncio.subprocess import DEVNULL, PIPE, STDOUT, Process
from pathlib import Path
from typing import Any, ClassVar, Final

from kotonoha._i18n import current_locale
from kotonoha.tui._tools_app import build_tool_command

MAXIMUM_OUTPUT_LINES: Final[int] = 1000


class OperationManager:
    """Run one validated operator command without invoking a shell."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "_lock",
        "_reader_task",
        "config_path",
        "lines",
        "operation",
        "process",
        "return_code",
    )

    config_path: Path | None
    process: Process | None
    operation: str | None
    lines: list[str]
    return_code: int | None
    _reader_task: asyncio.Task[None] | None
    _lock: asyncio.Lock

    def __init__(
        self,
        /,
        config_path: Path | None,
    ) -> None:
        self.config_path = config_path
        self.process = None
        self.operation = None
        self.lines = []
        self.return_code = None
        self._reader_task = None
        self._lock = asyncio.Lock()

    async def start(
        self,
        operation: str,
        values: dict[str, str],
        /,
    ) -> dict[str, Any]:
        async with self._lock:
            if self.process is not None and self.process.returncode is None:
                raise RuntimeError("an operation is already running")
            command = await asyncio.to_thread(
                build_tool_command,
                operation,
                values,
                self.config_path,
            )
            environment = os.environ.copy()
            environment["KOTONOHA_LANG"] = current_locale()
            environment["PYTHONUNBUFFERED"] = "1"
            self.lines = []
            self.operation = operation
            self.return_code = None
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdin=DEVNULL,
                stdout=PIPE,
                stderr=STDOUT,
                env=environment,
            )
            self._reader_task = asyncio.create_task(
                self._read_output(self.process),
                name="web-operation-output",
            )
        return self.snapshot()

    async def _read_output(
        self,
        process: Process,
        /,
    ) -> None:
        if process.stdout is not None:
            while line := await process.stdout.readline():
                self.lines.append(line.decode(errors="replace").rstrip())
                del self.lines[:-MAXIMUM_OUTPUT_LINES]
        self.return_code = await process.wait()

    async def stop(
        self,
        /,
    ) -> dict[str, Any]:
        process = self.process
        if process is None or process.returncode is not None:
            return self.snapshot()
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except TimeoutError:
            process.kill()
            await process.wait()
        if self._reader_task is not None:
            await self._reader_task
        return self.snapshot()

    async def close(
        self,
        /,
    ) -> None:
        await self.stop()

    def snapshot(
        self,
        /,
    ) -> dict[str, Any]:
        process = self.process
        running = process is not None and process.returncode is None
        return {
            "operation": self.operation,
            "running": running,
            "return_code": None if running else self.return_code,
            "lines": list(self.lines),
        }
