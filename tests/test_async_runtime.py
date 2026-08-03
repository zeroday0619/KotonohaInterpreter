"""Event-loop selection for command and TUI entry points."""

from __future__ import annotations

import asyncio

import uvicorn

from kotonoha.async_runtime import run
from kotonoha.cli import ServiceName, serve


def test_run_uses_uvloop() -> None:
    async def identify_loop() -> str:
        return type(asyncio.get_running_loop()).__module__

    assert run(identify_loop()).startswith("uvloop")


def test_standalone_service_forces_uvloop(monkeypatch) -> None:
    invocation: dict = {}

    def capture(target: str, **options) -> None:
        invocation.update(target=target, **options)

    monkeypatch.setattr(uvicorn, "run", capture)

    serve(ServiceName.asr)

    assert invocation["target"] == "kotonoha.services.asr_server:app"
    assert invocation["loop"] == "uvloop"
