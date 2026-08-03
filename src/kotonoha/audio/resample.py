"""soxr resampling: 48k capture down to the 16k working rate, 24k TTS up to the
output device rate."""

from __future__ import annotations

import numpy as np

try:  # soxr publishes aarch64 wheels; a miss shows up at install time.
    import soxr
except ImportError:  # pragma: no cover
    soxr = None


class Resampler:
    """Streaming resampler. Keeps state so phase does not break at block edges."""

    def __init__(self, in_rate: int, out_rate: int, quality: str = "HQ"):
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._passthrough = in_rate == out_rate
        self._st = None
        if not self._passthrough:
            if soxr is None:
                raise RuntimeError("soxr is not installed: pip install soxr")
            self._st = soxr.ResampleStream(
                in_rate, out_rate, num_channels=1, dtype="float32", quality=quality
            )

    def __call__(self, x: np.ndarray, last: bool = False) -> np.ndarray:
        if self._passthrough:
            return x.astype(np.float32, copy=False)
        out = self._st.resample_chunk(x.astype(np.float32, copy=False), last=last)
        return np.asarray(out, dtype=np.float32)

    def reset(self) -> None:
        if not self._passthrough:
            self._st = soxr.ResampleStream(
                self.in_rate, self.out_rate, num_channels=1, dtype="float32"
            )


def resample_once(x: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """One-shot resample, for loading files and similar."""
    if in_rate == out_rate:
        return x.astype(np.float32, copy=False)
    if soxr is None:
        raise RuntimeError("soxr is not installed: pip install soxr")
    return np.asarray(
        soxr.resample(x.astype(np.float32, copy=False), in_rate, out_rate), dtype=np.float32
    )
