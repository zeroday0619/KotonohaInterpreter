"""DeepFilterNet3 noise suppression (CPU).

DFN3 is 48 kHz only. Capture has to arrive at 48k, get cleaned here, and only
then be resampled down to 16k. Downsample first and DFN3 is off the table.

On the device (aarch64) the deepfilternet install may fail. In that case we fall
back to NoopDenoiser, but never silently — the state is logged and shown in the TUI.
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
    except Exception as e:  # noqa: BLE001 - interpreting must survive any failure here
        log.warning("denoiser.fallback", requested=backend, error=repr(e))
        return NoopDenoiser()
