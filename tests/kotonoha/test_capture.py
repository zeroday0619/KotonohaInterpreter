"""Microphone capture buffering and shutdown behavior."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import numpy as np

from kotonoha._config import load_settings
from kotonoha.audio._capture import FileCapture, Frame, MicCapture, RawRing


async def test_stop_does_not_block_when_the_callback_queue_is_full() -> None:
    settings = load_settings()
    capture = MicCapture(settings.audio, settings.frontend.vad)
    block = np.zeros(32, dtype=np.float32)
    for _block_index in range(capture._raw_queue.maxsize):
        capture._raw_queue.put_nowait((capture._gate_generation, block))

    await asyncio.wait_for(asyncio.to_thread(capture.stop), timeout=0.5)

    assert capture._raw_queue.empty()


def test_gate_generation_rejects_a_frame_scheduled_before_reopening() -> None:
    settings = load_settings()
    capture = MicCapture(settings.audio, settings.frontend.vad)
    stale_generation = capture._gate_generation
    capture.close_gate()
    capture.open_gate()

    capture._emit(
        Frame(index=0, pcm=np.ones(512, dtype=np.float32)),
        stale_generation,
    )

    assert capture.frames.empty()


def test_scheduled_frame_callbacks_are_bounded_by_the_frame_queue() -> None:
    settings = load_settings()
    capture = MicCapture(settings.audio, settings.frontend.vad)
    loop = Mock()
    loop.is_closed.return_value = False
    capture.loop = loop
    frame = Frame(index=0, pcm=np.ones(512, dtype=np.float32))

    for _frame_index in range(capture.frames.maxsize + 1):
        capture._schedule_emit(frame, capture._gate_generation)

    assert loop.call_soon_threadsafe.call_count == capture.frames.maxsize
    assert capture.overflows == 1


def test_raw_ring_clear_invalidates_data_without_zeroing_capacity() -> None:
    ring = RawRing(8)
    ring.push(np.arange(8, dtype=np.float32))

    ring.clear()

    assert ring.tail(8).size == 0
    ring.push(np.array([9.0, 10.0], dtype=np.float32))
    assert ring.tail(8).tolist() == [9.0, 10.0]


async def test_file_capture_produces_into_a_bounded_event_loop_queue() -> None:
    pcm = np.ones(512 * 1024, dtype=np.float32)
    capture = FileCapture(pcm)
    capture.loop = asyncio.get_running_loop()

    await asyncio.to_thread(capture.start)
    await asyncio.sleep(0)
    try:
        assert capture.frames.qsize() == capture.frames.maxsize
        assert capture._producer_future is not None
        assert not capture._producer_future.done()
    finally:
        await asyncio.to_thread(capture.stop)

    assert capture._producer_future is None
