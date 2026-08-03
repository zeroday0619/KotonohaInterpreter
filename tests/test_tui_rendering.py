"""Frame coalescing, level interpolation, and bounded event bursts."""

from __future__ import annotations

import pytest

from kotonoha.core.events import EventBus
from kotonoha.tui.app import METER_WIDTH, level_meter_text, level_meter_units
from kotonoha.tui.rendering import FrameAccumulator


def test_translation_deltas_collapse_to_the_latest_value_per_frame() -> None:
    accumulator = FrameAccumulator(now=0.0)
    for sequence in range(1_000):
        accumulator.push_translation(f"translation {sequence}")

    first_frame = accumulator.advance(now=1 / 60)
    second_frame = accumulator.advance(now=2 / 60)

    assert first_frame.translation == "translation 999"
    assert first_frame.translation_changed
    assert not second_frame.translation_changed


def test_level_interpolation_has_fast_attack_and_smooth_release() -> None:
    accumulator = FrameAccumulator(now=0.0)
    accumulator.push_level(0.08, now=0.0)

    attack = [accumulator.advance(now=frame / 60).level for frame in range(1, 7)]
    released = accumulator.advance(now=0.3).level

    assert attack == sorted(attack)
    assert 0.0 < attack[0] < attack[-1] < 0.08
    assert released < attack[-1]


def test_interpolation_is_independent_of_the_configured_frame_rate() -> None:
    sixty_hz = FrameAccumulator(now=0.0)
    fifteen_hz = FrameAccumulator(now=0.0)
    sixty_hz.push_level(0.05, now=0.0)
    fifteen_hz.push_level(0.05, now=0.0)

    for frame in range(1, 9):
        high_rate_level = sixty_hz.advance(now=frame / 60).level
    for frame in range(1, 3):
        low_rate_level = fifteen_hz.advance(now=frame / 15).level

    assert high_rate_level == pytest.approx(low_rate_level, abs=0.0001)


def test_level_meter_exposes_sixty_five_fixed_width_steps() -> None:
    units = {level_meter_units(step / (64 * 12)) for step in range(65)}
    meters = {level_meter_text(unit) for unit in units}

    assert units == set(range(65))
    assert len(meters) == 65
    assert all(len(meter) == METER_WIDTH for meter in meters)


def test_event_bus_drains_a_bounded_burst_without_reordering() -> None:
    event_bus = EventBus()
    for sequence in range(5):
        event_bus.emit("test", sequence=sequence)

    first_batch = event_bus.drain_nowait(maximum=3)
    second_batch = event_bus.drain_nowait(maximum=3)

    assert [event.payload["sequence"] for event in first_batch] == [0, 1, 2]
    assert [event.payload["sequence"] for event in second_batch] == [3, 4]
