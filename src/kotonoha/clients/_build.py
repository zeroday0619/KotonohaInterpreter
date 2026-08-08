"""Build the four service roles from the settings, honouring placement."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kotonoha._config import ROLES, Settings
from kotonoha._logging_setup import get_logger
from kotonoha.clients._asr import AsrClient
from kotonoha.clients._asr_verify import AsrVerifyClient
from kotonoha.clients._base import remote_transport_kwargs
from kotonoha.clients._llm import LanguageModelClient
from kotonoha.clients._router import FailoverClient
from kotonoha.clients._tts import TextToSpeechClient

log = get_logger(__name__)


@dataclass(slots=True)
class ServiceGroup:
    asr: FailoverClient
    asr_verify: FailoverClient
    llm: FailoverClient
    tts: FailoverClient

    def all(
        self,
        /,
    ) -> list[FailoverClient]:
        return [self.asr, self.asr_verify, self.llm, self.tts]

    def start_probes(
        self,
        /,
    ) -> None:
        for client in self.all():
            client.start_probe()

    def status(
        self,
        /,
    ) -> dict[str, Any]:
        return {client.role: client.status() for client in self.all()}

    @property
    def placement(
        self,
        /,
    ) -> dict[str, str]:
        return {client.role: client.side for client in self.all()}

    async def aclose(
        self,
        /,
    ) -> None:
        clients = self.all()
        results = await asyncio.gather(
            *(client.aclose() for client in clients),
            return_exceptions=True,
        )
        for client, result in zip(clients, results, strict=True):
            if isinstance(result, BaseException):
                log.error(
                    "services.close_failed",
                    role=client.role,
                    error=repr(result),
                )


_FACTORY = {
    "asr": lambda settings, url, side, transport: AsrClient(
        url,
        settings.asr,
        side=side,
        encoding=settings.remote.audio_encoding,
        **transport,
    ),
    "asr_verify": lambda settings, url, side, transport: AsrVerifyClient(
        url,
        settings.asr_verify,
        side=side,
        encoding=settings.remote.audio_encoding,
        **transport,
    ),
    "llm": lambda settings, url, side, transport: LanguageModelClient(
        url,
        settings.llm,
        side=side,
        **transport,
    ),
    "tts": lambda settings, url, side, transport: TextToSpeechClient(
        url,
        settings.tts,
        side=side,
        **transport,
    ),
}


def build_service_group(
    settings: Settings,
    /,
    on_change: Callable[[str, str, str], None] | None = None,
) -> ServiceGroup:
    placement = settings.resolved_placement()
    remote_options = remote_transport_kwargs(settings.remote)

    built = {}
    for role in ROLES:
        side = placement[role]
        factory = _FACTORY[role]
        transport = remote_options if side == "remote" else {}
        preferred_url = settings.url_for(role, side)
        local_url = settings.url_for(role, "local")
        preferred = factory(settings, preferred_url, side, transport)
        # A remote role keeps its on-board twin ready so a link failure costs a
        # retry rather than the turn (§10). A deployment can intentionally map
        # both sides to one authenticated service, as the A6000 Web stack does.
        # Retrying that endpoint as a local shared-memory service cannot recover
        # the request and drops the bearer token, so it must not be registered.
        fallback = (
            factory(settings, local_url, "local", {})
            if side == "remote" and preferred_url.rstrip("/") != local_url.rstrip("/")
            else None
        )
        built[role] = FailoverClient(role, preferred, fallback, settings.remote, on_change)

    log.info(
        "services.built",
        perf_mode=settings.perf_mode,
        placement=placement,
        audio_leaves_device=settings.audio_leaves_device,
    )
    return ServiceGroup(**built)
