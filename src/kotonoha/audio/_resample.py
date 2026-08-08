"""soxr resampling: 48k capture down to the 16k working rate, 24k TTS up to the
output device rate."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from kotonoha._typing import override

try:
    import soxr
except ImportError:  # pragma: no cover
    soxr = None


class Resampler:
    """Streaming resampler. Keeps state so phase does not break at block edges."""
    __slots__: ClassVar[tuple[str, ...]] = (
        "_passthrough",
        "_quality",
        "_stream",
        "input_rate",
        "output_rate",
    )

    input_rate: int
    output_rate: int
    _passthrough: bool
    _quality: str
    _stream: Any | None

    @override
    def __init__(
        self,
        /,
        input_rate: int,
        output_rate: int,
        quality: str = "HQ",
    ) -> None:
        self.input_rate = input_rate
        self.output_rate = output_rate
        self._quality = quality
        self._passthrough = input_rate == output_rate
        self._stream = None
        if not self._passthrough:
            if soxr is None:
                raise RuntimeError("soxr is not installed")
            self._stream = soxr.ResampleStream(
                input_rate,
                output_rate,
                num_channels=1,
                dtype="float32",
                quality=self._quality,
            )

    def __call__(
        self,
        /,
        samples: np.ndarray,
        last: bool = False,
    ) -> np.ndarray:
        if self._passthrough:
            return samples.astype(np.float32, copy=False)
        output = self._stream.resample_chunk(
            samples.astype(np.float32, copy=False),
            last=last,
        )
        return np.asarray(output, dtype=np.float32)

    def reset(
        self,
        /,
    ) -> None:
        if not self._passthrough:
            self._stream = soxr.ResampleStream(
                self.input_rate,
                self.output_rate,
                num_channels=1,
                dtype="float32",
                quality=self._quality,
            )


def resample_once(
    samples: np.ndarray,
    /,
    input_rate: int,
    output_rate: int,
) -> np.ndarray:
    """One-shot resample, for loading files and similar."""
    if input_rate == output_rate:
        return samples.astype(np.float32, copy=False)
    if soxr is None:
        raise RuntimeError("soxr is not installed")
    return np.asarray(
        soxr.resample(
            samples.astype(np.float32, copy=False),
            input_rate,
            output_rate,
        ),
        dtype=np.float32,
    )
