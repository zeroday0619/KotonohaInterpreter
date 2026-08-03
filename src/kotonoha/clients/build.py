"""Build the four service roles from the settings, honouring placement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..config import ROLES, Settings
from ..logging_setup import get_logger
from .asr import AsrClient
from .asr_verify import AsrVerifyClient
from .base import remote_transport_kwargs
from .llm import LlmClient
from .router import FailoverClient
from .tts import TtsClient

log = get_logger(__name__)


@dataclass
class ServiceGroup:
    asr: FailoverClient
    asr_verify: FailoverClient
    llm: FailoverClient
    tts: FailoverClient

    def all(self) -> list[FailoverClient]:
        return [self.asr, self.asr_verify, self.llm, self.tts]

    def start_probes(self) -> None:
        for c in self.all():
            c.start_probe()

    def status(self) -> dict[str, Any]:
        return {c.role: c.status() for c in self.all()}

    @property
    def placement(self) -> dict[str, str]:
        return {c.role: c.side for c in self.all()}

    async def aclose(self) -> None:
        for c in self.all():
            await c.aclose()


_FACTORY = {
    "asr": lambda s, url, side, tk: AsrClient(
        url, s.asr, side=side, encoding=s.remote.audio_encoding, **tk
    ),
    "asr_verify": lambda s, url, side, tk: AsrVerifyClient(
        url, s.asr_verify, side=side, encoding=s.remote.audio_encoding, **tk
    ),
    "llm": lambda s, url, side, tk: LlmClient(url, s.llm, side=side, **tk),
    "tts": lambda s, url, side, tk: TtsClient(url, s.tts, side=side, **tk),
}


def build_service_group(
    s: Settings, on_change: Callable[[str, str, str], None] | None = None
) -> ServiceGroup:
    placement = s.resolved_placement()
    remote_kwargs = remote_transport_kwargs(s.remote)

    built = {}
    for role in ROLES:
        side = placement[role]
        make = _FACTORY[role]
        tk = remote_kwargs if side == "remote" else {}
        preferred = make(s, s.url_for(role, side), side, tk)
        # A remote role keeps its on-board twin ready so a link failure costs a
        # retry rather than the turn (§10).
        fallback = make(s, s.url_for(role, "local"), "local", {}) if side == "remote" else None
        built[role] = FailoverClient(role, preferred, fallback, s.remote, on_change)

    log.info(
        "services.built",
        perf_mode=s.perf_mode,
        placement=placement,
        audio_leaves_device=s.audio_leaves_device,
    )
    return ServiceGroup(**built)
