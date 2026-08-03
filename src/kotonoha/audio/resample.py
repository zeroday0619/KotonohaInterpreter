"""soxr 리샘플링. 캡처 48k → 작업 16k, TTS 24k → 출력 장치 레이트."""

from __future__ import annotations

import numpy as np

try:  # soxr 는 aarch64 wheel 이 있다. 없으면 설치 단계에서 걸린다.
    import soxr
except ImportError:  # pragma: no cover
    soxr = None


class Resampler:
    """스트리밍 리샘플러. 블록 경계에서 위상이 끊기지 않도록 상태를 유지한다."""

    def __init__(self, in_rate: int, out_rate: int, quality: str = "HQ"):
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._passthrough = in_rate == out_rate
        self._st = None
        if not self._passthrough:
            if soxr is None:
                raise RuntimeError("soxr 가 설치되어 있지 않다: pip install soxr")
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
    """일회성 리샘플 (파일 로딩 등)."""
    if in_rate == out_rate:
        return x.astype(np.float32, copy=False)
    if soxr is None:
        raise RuntimeError("soxr 가 설치되어 있지 않다: pip install soxr")
    return np.asarray(
        soxr.resample(x.astype(np.float32, copy=False), in_rate, out_rate), dtype=np.float32
    )
