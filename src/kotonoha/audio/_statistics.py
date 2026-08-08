"""Allocation-conscious statistics for normalized mono PCM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SignalStatistics:
    sample_count: int
    peak: float
    square_sum: float
    clipped_sample_count: int

    @property
    def root_mean_square(
        self,
        /,
    ) -> float:
        if self.sample_count == 0:
            return 0.0
        return (self.square_sum / self.sample_count) ** 0.5


def signal_statistics(
    samples: np.ndarray,
    /,
) -> SignalStatistics:
    """Compute level totals with one temporary array regardless of input size."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return SignalStatistics(0, 0.0, 0.0, 0)
    absolute = np.abs(audio)
    return SignalStatistics(
        sample_count=int(audio.size),
        peak=float(np.max(absolute, initial=0.0)),
        # Normalized PCM and the configured duration caps keep float32 accumulation
        # within the precision required by 0.1 dB diagnostics. BLAS-backed dot avoids
        # the fixed contraction-planning cost paid by einsum for every audio chunk.
        square_sum=float(np.dot(audio, audio)),
        clipped_sample_count=int(np.count_nonzero(absolute >= 0.999)),
    )
