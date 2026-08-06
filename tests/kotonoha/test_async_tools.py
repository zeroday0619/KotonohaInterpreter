"""Tests for the aiotools-backed asynchronous lifecycle helpers."""

from __future__ import annotations

import asyncio

import pytest

from kotonoha import _async_tools


async def test_create_timer_runs_a_non_overlapping_callback(
    _positional_only: object | None = None,
    /,
) -> None:
    del _positional_only
    called = asyncio.Event()
    intervals: list[float] = []

    async def callback(
        interval: float,
        /,
    ) -> None:
        intervals.append(interval)
        called.set()

    timer = _async_tools.create_timer(callback, 0.001)
    try:
        await asyncio.wait_for(called.wait(), timeout=1.0)
    finally:
        await _async_tools.cancel_and_wait(timer)

    assert intervals == [pytest.approx(0.001)]


async def test_helpers_use_the_standard_asyncio_compatibility_path(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del _positional_only
    monkeypatch.setattr(_async_tools, "aiotools", None)
    called = asyncio.Event()

    async def callback(
        interval: float,
        /,
    ) -> None:
        del interval
        called.set()

    timer = _async_tools.create_timer(callback, 0.001)
    try:
        await asyncio.wait_for(called.wait(), timeout=1.0)
    finally:
        await _async_tools.cancel_and_wait(timer)

    assert timer.done()


async def test_cancel_and_wait_accepts_a_collection(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del _positional_only
    monkeypatch.setattr(_async_tools, "aiotools", None)
    first = asyncio.create_task(asyncio.sleep(60))
    second = asyncio.create_task(asyncio.sleep(60))

    await _async_tools.cancel_and_wait((first, second))

    assert first.cancelled()
    assert second.cancelled()
