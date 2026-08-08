"""Build the four model-service clients selected by the hardware profile."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from kotonoha._config import ROLES, Settings
from kotonoha._logging_setup import get_logger
from kotonoha.clients._asr import AsrClient
from kotonoha.clients._asr_verify import AsrVerifyClient
from kotonoha.clients._base import service_transport_kwargs
from kotonoha.clients._llm import LanguageModelClient
from kotonoha.clients._managed import ManagedServiceClient
from kotonoha.clients._tts import TextToSpeechClient

log = get_logger(__name__)


@dataclass(slots=True)
class ServiceGroup:
    asr: ManagedServiceClient
    asr_verify: ManagedServiceClient
    llm: ManagedServiceClient
    tts: ManagedServiceClient
    hardware_profile: str

    def all(
        self,
        /,
    ) -> list[ManagedServiceClient]:
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
    "asr": lambda settings, url, transport: AsrClient(
        url,
        settings.asr,
        side="local",
        encoding=settings.transport.audio_encoding,
        **transport,
    ),
    "asr_verify": lambda settings, url, transport: AsrVerifyClient(
        url,
        settings.asr_verify,
        side="local",
        encoding=settings.transport.audio_encoding,
        **transport,
    ),
    "llm": lambda settings, url, transport: LanguageModelClient(
        url,
        settings.llm,
        side="local",
        **transport,
    ),
    "tts": lambda settings, url, transport: TextToSpeechClient(
        url,
        settings.tts,
        side="local",
        **transport,
    ),
}


def build_service_group(
    settings: Settings,
    /,
) -> ServiceGroup:
    transport = service_transport_kwargs(settings.transport)
    built: dict[str, ManagedServiceClient] = {}
    for role in ROLES:
        factory = _FACTORY[role]
        client = factory(
            settings,
            settings.service_url(role),
            transport,
        )
        built[role] = ManagedServiceClient(role, client)

    log.info(
        "services.built",
        hardware_profile=settings.accelerator.profile,
        endpoints={role: settings.service_url(role) for role in ROLES},
    )
    return ServiceGroup(**built, hardware_profile=settings.accelerator.profile)
