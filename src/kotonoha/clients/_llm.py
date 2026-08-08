"""TranslateGemma client using the resident service's WebSocket stream."""

from __future__ import annotations

import asyncio
import json
import ssl
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final
from urllib.parse import urlsplit, urlunsplit

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from kotonoha._config import LanguageModelConfig
from kotonoha._logging_setup import get_logger
from kotonoha._typing import override
from kotonoha.clients._base import (
    BaseClient,
    ServiceApplicationError,
    ServiceError,
    ServiceTimeout,
    read_json_object_response,
)

log = get_logger(__name__)
MAXIMUM_TRANSLATION_OUTPUT_CHARACTERS: Final[int] = 32768


@dataclass(slots=True)
class GenerationStatistics:
    token_count: int = 0
    first_token_at: float | None = None
    started_at: float = field(default_factory=time.perf_counter)
    finished_at: float | None = None

    @property
    def time_to_first_token_ms(
        self,
        /,
    ) -> float | None:
        if self.first_token_at is None:
            return None
        return round((self.first_token_at - self.started_at) * 1000, 1)

    @property
    def tokens_per_second(
        self,
        /,
    ) -> float | None:
        if self.finished_at is None or self.first_token_at is None or self.token_count <= 1:
            return None
        generation_seconds = self.finished_at - self.first_token_at
        if generation_seconds <= 0:
            return None
        return round((self.token_count - 1) / generation_seconds, 2)


class LanguageModelClient(BaseClient):
    __slots__: ClassVar[tuple[str, ...]] = (
        "config",
    )
    config: LanguageModelConfig

    @override
    def __init__(
        self,
        /,
        base_url: str,
        config: LanguageModelConfig,
        *,
        side: str = "local",
        **transport_options: Any,
    ) -> None:
        # A full stream can run long. This timeout applies to inactivity between
        # WebSocket events; the orchestrator owns the first-clause deadline.
        super().__init__(
            base_url,
            # This is an inactivity timeout for each WebSocket receive, not a
            # deadline for the complete translation stream.
            timeout=config.timeout_s,
            name="llm",
            side=side,
            **transport_options,
        )
        self.config = config

    @override
    async def health(
        self,
        /,
    ) -> dict:
        """Return the vLLM server health state."""
        try:
            async with self._client.stream("GET", "/health", timeout=2.0) as response:
                payload = await read_json_object_response(response, allow_empty=True)
                status_code = response.status_code
            if status_code == 200:
                return {
                    "ok": True,
                    "service": "llm",
                    "side": self.side,
                    **payload,
                }
            return {
                "ok": False,
                "service": "llm",
                "status": status_code,
                "side": self.side,
            }
        except Exception as error:  # noqa: BLE001
            return {
                "ok": False,
                "service": "llm",
                "error": repr(error),
                "side": self.side,
            }

    async def stream_chat(
        self,
        /,
        messages: list[dict[str, Any]],
        statistics: GenerationStatistics | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield content deltas one at a time."""
        generation_statistics = statistics or GenerationStatistics()
        completion_token_limit = (
            max_tokens if max_tokens is not None else self.config.max_tokens
        )
        payload = {
            "type": "translation.create",
            "model": self.config.served_model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "max_tokens": completion_token_limit,
        }
        loop = asyncio.get_running_loop()
        deadline = (
            loop.time()
            + self.config.timeout_s
            + completion_token_limit / self.config.min_tok_per_s
        )
        output_characters = 0
        try:
            async with connect(
                _websocket_url(self.base_url),
                additional_headers=_websocket_headers(self._headers),
                open_timeout=self._connect_timeout,
                ssl=_websocket_ssl(self.base_url, self._verify),
            ) as websocket:
                await websocket.send(json.dumps(payload, ensure_ascii=False))
                while True:
                    remaining_seconds = deadline - loop.time()
                    if remaining_seconds <= 0:
                        raise ServiceTimeout("llm stream exceeded the generation deadline")
                    raw_event = await asyncio.wait_for(
                        websocket.recv(),
                        min(self._timeout, remaining_seconds),
                    )
                    decoded_event = _decode_websocket_event(raw_event)
                    event_type = decoded_event.get("type")
                    if event_type == "session.created":
                        continue
                    if event_type == "error":
                        raise ServiceApplicationError(
                            f"llm application error: {decoded_event.get('error', 'unknown')}"
                        )
                    if event_type == "translation.done":
                        usage = decoded_event.get("usage")
                        completion_tokens = (
                            usage.get("completion_tokens")
                            if isinstance(usage, dict)
                            else None
                        )
                        if isinstance(completion_tokens, int):
                            generation_statistics.token_count = completion_tokens
                        break
                    if event_type != "translation.delta":
                        continue
                    delta = decoded_event.get("delta")
                    if not isinstance(delta, str) or not delta:
                        continue
                    output_characters += len(delta)
                    if output_characters > MAXIMUM_TRANSLATION_OUTPUT_CHARACTERS:
                        raise ServiceError(
                            "llm output exceeded "
                            f"{MAXIMUM_TRANSLATION_OUTPUT_CHARACTERS} characters"
                        )
                    generation_statistics.token_count += 1
                    if generation_statistics.first_token_at is None:
                        generation_statistics.first_token_at = time.perf_counter()
                    yield delta
        except asyncio.TimeoutError as error:
            raise ServiceTimeout("llm stream timeout") from error
        except (OSError, WebSocketException) as error:
            raise ServiceError(f"llm transport error: {error!r}") from error
        finally:
            generation_statistics.finished_at = time.perf_counter()
            if (
                generation_statistics.tokens_per_second is not None
                and generation_statistics.tokens_per_second < self.config.min_tok_per_s
            ):
                # Violates the §5.4 precondition: clause streaming can no
                # longer keep up with playback.
                log.warning(
                    "llm.too_slow",
                    tok_per_s=generation_statistics.tokens_per_second,
                    required=self.config.min_tok_per_s,
                    profile=self.config.profile,
                )


def _decode_websocket_event(
    raw_event: str | bytes,
    /,
) -> dict[str, Any]:
    """Decode one bounded transport frame and require the protocol object shape."""
    try:
        decoded_event = json.loads(raw_event)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ServiceError("llm returned invalid WebSocket JSON") from error
    if not isinstance(decoded_event, dict):
        raise ServiceError("llm returned a non-object WebSocket event")
    return decoded_event


def _websocket_url(
    base_url: str,
    /,
) -> str:
    parts = urlsplit(base_url)
    scheme = "wss" if parts.scheme == "https" else "ws"
    path = f"{parts.path.rstrip('/')}/v1/realtime"
    return urlunsplit((scheme, parts.netloc, path, "", ""))


def _websocket_headers(
    headers: dict[str, str],
    /,
) -> dict[str, str]:
    return {name: value for name, value in headers.items() if name.lower() != "connection"}


def _websocket_ssl(
    base_url: str,
    verify: bool | ssl.SSLContext,
    /,
) -> ssl.SSLContext | None:
    if not base_url.startswith("https://"):
        return None
    if isinstance(verify, ssl.SSLContext):
        return verify
    if verify:
        return ssl.create_default_context()
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context
