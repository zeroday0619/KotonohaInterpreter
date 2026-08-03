"""DeepFilterNet3 잡음 제거 (CPU).

DFN3 은 48kHz 전용이다. 캡처를 48k로 받아 여기서 처리하고, 그 다음에 16k로
내리는 순서를 지켜야 한다. 16k로 먼저 내리면 DFN3 을 못 쓴다.

실기(aarch64)에서 deepfilternet 설치가 실패할 수 있다. 그 경우 NoopDenoiser 로
떨어지되 **조용히 넘어가지 않고** 로그와 TUI에 상태를 남긴다.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..logging_setup import get_logger

log = get_logger(__name__)

DFN_RATE = 48000


class Denoiser(Protocol):
    rate: int
    name: str

    def __call__(self, x: np.ndarray) -> np.ndarray: ...


class NoopDenoiser:
    name = "none"

    def __init__(self, rate: int = DFN_RATE):
        self.rate = rate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x


class DeepFilterNet3:
    name = "deepfilternet3"
    rate = DFN_RATE

    def __init__(self, post_filter_beta: float = 0.02):
        from df.enhance import init_df  # type: ignore[import-not-found]

        self.model, self.state, _ = init_df(post_filter=True)
        self._beta = post_filter_beta
        if self.state.sr() != DFN_RATE:
            raise RuntimeError(f"DFN3 expects 48kHz, got {self.state.sr()}")
        log.info("denoiser.loaded", backend=self.name, sr=self.rate)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        import torch  # type: ignore[import-not-found]
        from df.enhance import enhance  # type: ignore[import-not-found]

        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            y = enhance(self.model, self.state, t)
        return y.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def build_denoiser(enabled: bool, backend: str, post_filter_beta: float) -> Denoiser:
    if not enabled or backend == "none":
        return NoopDenoiser()
    try:
        return DeepFilterNet3(post_filter_beta=post_filter_beta)
    except Exception as e:  # noqa: BLE001 — 어떤 실패든 통역은 계속돼야 한다
        log.warning("denoiser.fallback", requested=backend, error=repr(e))
        return NoopDenoiser()
