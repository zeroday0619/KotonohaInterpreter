"""Frame-rate independent state coalescing for the interpreter interface."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

LEVEL_IDLE_SECONDS = 0.2
LEVEL_ATTACK_SECONDS = 0.045
LEVEL_RELEASE_SECONDS = 0.16


@dataclass(frozen=True)
class FrameUpdate:
    """Values that must be committed during one Textual refresh batch."""

    level: float
    translation: str | None
    translation_changed: bool


class FrameAccumulator:
    """Retain only the latest high-frequency values between display frames."""

    target_level: float
    visual_level: float
    last_level_at: float
    last_frame_at: float
    pending_translation: str | None
    translation_changed: bool

    def __init__(self, now: float | None = None) -> None:
        current_time = time.monotonic() if now is None else now
        self.target_level = 0.0
        self.visual_level = 0.0
        self.last_level_at = current_time
        self.last_frame_at = current_time
        self.pending_translation: str | None = None
        self.translation_changed = False

    def push_level(self, level: float, now: float | None = None) -> None:
        self.target_level = max(0.0, float(level))
        self.last_level_at = time.monotonic() if now is None else now

    def push_translation(self, text: str) -> None:
        self.pending_translation = text
        self.translation_changed = True

    def discard_translation(self) -> None:
        self.pending_translation = None
        self.translation_changed = False

    def advance(self, now: float | None = None) -> FrameUpdate:
        current_time = time.monotonic() if now is None else now
        elapsed = max(0.0, current_time - self.last_frame_at)
        self.last_frame_at = current_time

        target = (
            self.target_level
            if current_time - self.last_level_at <= LEVEL_IDLE_SECONDS
            else 0.0
        )
        response_seconds = (
            LEVEL_ATTACK_SECONDS if target > self.visual_level else LEVEL_RELEASE_SECONDS
        )
        blend = 1.0 - math.exp(-elapsed / response_seconds) if elapsed else 0.0
        self.visual_level += (target - self.visual_level) * blend
        if target == 0.0 and self.visual_level < 0.0001:
            self.visual_level = 0.0

        update = FrameUpdate(
            level=self.visual_level,
            translation=self.pending_translation,
            translation_changed=self.translation_changed,
        )
        self.translation_changed = False
        return update
