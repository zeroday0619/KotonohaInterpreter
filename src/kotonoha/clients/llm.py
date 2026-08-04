"""llama.cpp server client (OpenAI-compatible /v1/chat/completions, SSE streaming)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from kotonoha.clients.base import BaseClient, ServiceError, ServiceTimeout
from kotonoha.config import LanguageModelConfig
from kotonoha.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class GenerationStatistics:
    token_count: int = 0
    first_token_at: float | None = None
    started_at: float = field(default_factory=time.perf_counter)
    finished_at: float | None = None

    @property
    def time_to_first_token_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return round((self.first_token_at - self.started_at) * 1000, 1)

    @property
    def tokens_per_second(self) -> float | None:
        if self.finished_at is None or self.first_token_at is None or self.token_count <= 1:
            return None
        generation_seconds = self.finished_at - self.first_token_at
        if generation_seconds <= 0:
            return None
        return round((self.token_count - 1) / generation_seconds, 2)


class LanguageModelClient(BaseClient):
    config: LanguageModelConfig

    def __init__(
        self,
        base_url: str,
        config: LanguageModelConfig,
        *,
        side: str = "local",
        **transport_options,
    ):
        # A full stream can run long. The first-clause timeout is the
        # orchestrator's job, not this timeout.
        super().__init__(
            base_url,
            timeout=120.0,
            name="llm",
            side=side,
            **transport_options,
        )
        self.config = config

    async def health(self) -> dict:
        """llama.cpp server exposes /health."""
        try:
            response = await self._client.get("/health", timeout=2.0)
            if response.status_code == 200:
                return {
                    "ok": True,
                    "service": "llm",
                    "side": self.side,
                    **_safe_json(response),
                }
            return {
                "ok": False,
                "service": "llm",
                "status": response.status_code,
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
        messages: list[dict[str, str]],
        statistics: GenerationStatistics | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield content deltas one at a time."""
        generation_statistics = statistics or GenerationStatistics()
        payload = {
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repeat_penalty": self.config.repeat_penalty,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": True,
            "cache_prompt": True,
        }
        try:
            timeout = httpx.Timeout(120.0, connect=2.0)
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=payload, timeout=timeout
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    event_data = line[5:].strip()
                    if event_data == "[DONE]":
                        break
                    try:
                        decoded_event = json.loads(event_data)
                    except json.JSONDecodeError:
                        continue
                    choices = decoded_event.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content")
                    if not delta:
                        continue
                    generation_statistics.token_count += 1
                    if generation_statistics.first_token_at is None:
                        generation_statistics.first_token_at = time.perf_counter()
                    yield delta
        except httpx.TimeoutException as error:
            raise ServiceTimeout("llm stream timeout") from error
        except httpx.HTTPError as error:
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


def _safe_json(response: httpx.Response) -> dict:
    try:
        return response.json()
    except Exception:  # noqa: BLE001
        return {}
