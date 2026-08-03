"""llama.cpp server 클라이언트 (OpenAI 호환 /v1/chat/completions, SSE 스트리밍)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from ..config import LlmCfg
from ..logging_setup import get_logger
from .base import BaseClient, ServiceError, ServiceTimeout

log = get_logger(__name__)


@dataclass
class StreamStats:
    tokens: int = 0
    t_first_token: float | None = None
    t_start: float = field(default_factory=time.perf_counter)
    t_end: float | None = None

    @property
    def ttft_ms(self) -> float | None:
        if self.t_first_token is None:
            return None
        return round((self.t_first_token - self.t_start) * 1000, 1)

    @property
    def tok_per_s(self) -> float | None:
        if self.t_end is None or self.t_first_token is None or self.tokens <= 1:
            return None
        span = self.t_end - self.t_first_token
        return round((self.tokens - 1) / span, 2) if span > 0 else None


class LlmClient(BaseClient):
    def __init__(self, base_url: str, cfg: LlmCfg):
        # 스트리밍 전체 시간은 길 수 있다. 첫 절 타임아웃은 오케스트레이터가 건다.
        super().__init__(base_url, timeout=120.0, name="llm")
        self.cfg = cfg

    async def health(self) -> dict:
        """llama.cpp server 는 /health 를 준다."""
        try:
            r = await self._client.get("/health", timeout=2.0)
            if r.status_code == 200:
                return {"ok": True, "service": "llm", **_safe_json(r)}
            return {"ok": False, "service": "llm", "status": r.status_code}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "service": "llm", "error": repr(e)}

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        stats: StreamStats | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """콘텐츠 델타를 하나씩 흘린다."""
        st = stats or StreamStats()
        payload = {
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "repeat_penalty": self.cfg.repeat_penalty,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "stream": True,
            "cache_prompt": True,
        }
        try:
            timeout = httpx.Timeout(120.0, connect=2.0)
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=payload, timeout=timeout
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content")
                    if not delta:
                        continue
                    st.tokens += 1
                    if st.t_first_token is None:
                        st.t_first_token = time.perf_counter()
                    yield delta
        except httpx.TimeoutException as e:
            raise ServiceTimeout("llm stream timeout") from e
        except httpx.HTTPError as e:
            raise ServiceError(f"llm transport error: {e!r}") from e
        finally:
            st.t_end = time.perf_counter()
            if st.tok_per_s is not None and st.tok_per_s < self.cfg.min_tok_per_s:
                # §5.4 성립 조건 위반. 절 스트리밍이 재생을 못 따라간다.
                log.warning(
                    "llm.too_slow",
                    tok_per_s=st.tok_per_s,
                    required=self.cfg.min_tok_per_s,
                    profile=self.cfg.profile,
                )


def _safe_json(r: httpx.Response) -> dict:
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        return {}
