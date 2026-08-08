"""DeepFilterNet3 noise suppression (CPU).

DFN3 is 48 kHz only. Capture has to arrive at 48k, get cleaned here, and only
then be resampled down to 16k. Downsample first and DFN3 is off the table.

On the device (aarch64), the DeepFilterNet installation requires verification.
A failure activates `NoopDenoiser` and records the fallback in the log and Web UI.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final, Protocol

import numpy as np

from kotonoha._logging_setup import get_logger
from kotonoha._typing import override

log = get_logger(__name__)

DFN_RATE = 48000


class Denoiser(Protocol):
    __slots__: ClassVar[tuple[str, ...]] = ()
    rate: int
    name: str

    def __call__(
        self,
        /,
        samples: np.ndarray,
    ) -> np.ndarray: ...


class NoopDenoiser:
    __slots__: ClassVar[tuple[str, ...]] = (
        "rate",
    )
    name: Final = "none"
    rate: int

    @override
    def __init__(
        self,
        /,
        rate: int = DFN_RATE,
    ) -> None:
        self.rate = rate

    def __call__(
        self,
        /,
        samples: np.ndarray,
    ) -> np.ndarray:
        return samples


class DeepFilterNet3:
    __slots__: ClassVar[tuple[str, ...]] = (
        "_beta",
        "model",
        "state",
    )
    name: Final = "deepfilternet3"
    rate: Final = DFN_RATE
    model: Any
    state: Any
    _beta: float

    @override
    def __init__(
        self,
        /,
        post_filter_beta: float = 0.02,
    ) -> None:
        from df.enhance import init_df  # type: ignore[import-not-found]

        self.model, self.state, _ = init_df(post_filter=True)
        self._beta = post_filter_beta
        if self.state.sr() != DFN_RATE:
            raise RuntimeError(f"DFN3 expects 48kHz, got {self.state.sr()}")
        log.info("denoiser.loaded", backend=self.name, sr=self.rate)

    def __call__(
        self,
        /,
        samples: np.ndarray,
    ) -> np.ndarray:
        import torch  # type: ignore[import-not-found]
        from df.enhance import enhance  # type: ignore[import-not-found]

        tensor = torch.from_numpy(
            np.ascontiguousarray(samples, dtype=np.float32)
        ).unsqueeze(0)
        with torch.no_grad():
            enhanced = enhance(self.model, self.state, tensor)
        return enhanced.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def build_denoiser(
    enabled: bool,
    /,
    backend: str,
    post_filter_beta: float,
) -> Denoiser:
    if not enabled or backend == "none":
        return NoopDenoiser()
    try:
        return DeepFilterNet3(post_filter_beta=post_filter_beta)
    except Exception as error:  # noqa: BLE001 - interpreting must survive any failure here
        log.warning("denoiser.fallback", requested=backend, error=repr(error))
        return NoopDenoiser()
