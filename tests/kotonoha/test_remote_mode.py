"""High-performance mode: role placement, audio transport, and link failover."""

from __future__ import annotations

import json
from typing import Any, ClassVar, Final

import httpx
import numpy as np
import pytest

from kotonoha._config import RemoteConfig, load_settings
from kotonoha._transport import AudioPayload, decode_pcm, encode_pcm, encoded_size
from kotonoha.clients._base import ServiceError, ServiceTimeout, remote_transport_kwargs
from kotonoha.clients._llm import LanguageModelClient
from kotonoha.clients._router import AllEndpointsFailed, FailoverClient
from kotonoha.clients._tts import TextToSpeechClient


# -- placement -------------------------------------------------------------
def _settings(
    _positional_only: object | None = None,
    /,
    **over: Any,
) -> Any:
    s = load_settings()
    for k, v in over.items():
        setattr(s, k, v)
    return s


def test_onboard_keeps_everything_local() -> None:
    s = _settings(perf_mode="onboard")
    s.remote.enabled = True
    assert s.resolved_placement() == {
        "asr": "local",
        "asr_verify": "local",
        "llm": "local",
        "tts": "local",
    }
    assert not s.audio_leaves_device


def test_hybrid_moves_only_the_llm_and_keeps_audio_on_device() -> None:
    """The point of hybrid: the biggest win, without shipping any audio."""
    s = _settings(perf_mode="hybrid")
    s.remote.enabled = True
    p = s.resolved_placement()
    assert p["llm"] == "remote"
    assert p["asr"] == p["asr_verify"] == p["tts"] == "local"
    assert not s.audio_leaves_device


def test_remote_moves_everything_and_flags_the_audio() -> None:
    s = _settings(perf_mode="remote")
    s.remote.enabled = True
    assert set(s.resolved_placement().values()) == {"remote"}
    assert s.audio_leaves_device


def test_custom_mode_selects_each_role_independently() -> None:
    s = _settings(
        perf_mode="custom",
        placement={"asr": "remote", "llm": "remote", "tts": "local"},
    )
    s.remote.enabled = True
    assert s.resolved_placement() == {
        "asr": "remote",
        "asr_verify": "local",
        "llm": "remote",
        "tts": "local",
    }
    assert s.audio_leaves_device


def test_disabled_remote_collapses_to_local() -> None:
    """A mode pointing at an unreachable box would just be a per-turn timeout."""
    s = _settings(perf_mode="remote")
    s.remote.enabled = False
    assert set(s.resolved_placement().values()) == {"local"}
    assert not s.audio_leaves_device


def test_explicit_placement_overrides_the_mode() -> None:
    s = _settings(perf_mode="onboard", placement={"llm": "remote"})
    s.remote.enabled = True
    p = s.resolved_placement()
    assert p["llm"] == "remote" and p["asr"] == "local"


def test_unknown_role_in_placement_is_rejected() -> None:
    s = _settings(placement={"vocoder": "remote"})
    s.remote.enabled = True
    with pytest.raises(ValueError, match="unknown role"):
        s.resolved_placement()


def test_url_selection_follows_the_side() -> None:
    s = _settings()
    s.remote.services.llm = "http://a6000.lan:8003"
    assert s.url_for("llm", "local").startswith("http://127.0.0.1")
    assert s.url_for("llm", "remote") == "http://a6000.lan:8003"


# -- transport --------------------------------------------------------------
def test_pcm_roundtrip_s16le_is_lossy_but_close() -> None:
    x = np.linspace(-1, 1, 4000, dtype=np.float32)
    y = decode_pcm(encode_pcm(x, "s16le"), "s16le")
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1e-4


def test_pcm_roundtrip_f32le_is_exact() -> None:
    x = np.linspace(-1, 1, 4000, dtype=np.float32)
    y = decode_pcm(encode_pcm(x, "f32le"), "f32le")
    assert np.array_equal(x, y)


def test_s16le_halves_the_bytes_on_the_wire() -> None:
    assert encoded_size(6.0, 16000, "s16le") == 192_000
    assert encoded_size(6.0, 16000, "f32le") == 384_000


def test_payload_carries_both_forms() -> None:
    pcm = np.zeros(16000, dtype=np.float32)
    payload = AudioPayload(pcm=pcm, audio_reference=None)
    assert payload.seconds == pytest.approx(1.0)
    assert len(payload.encoded("s16le")) == 32_000


def test_bearer_token_and_tls_flags_reach_httpx() -> None:
    tk = remote_transport_kwargs(RemoteConfig(token="secret", verify_tls=False))
    assert tk["headers"]["authorization"] == "Bearer secret"
    assert tk["verify"] is False


async def test_translation_request_uses_realtime_websocket_fields(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del _positional_only
    settings = load_settings()
    captured_payload: dict[str, Any] = {}
    captured_connection: dict[str, Any] = {}

    class FakeWebSocket:
        __slots__: ClassVar[tuple[str, ...]] = ("events",)

        def __init__(
            self,
            /,
        ) -> None:
            self.events = [
                json.dumps({"type": "session.created"}),
                json.dumps({"type": "translation.delta", "delta": "Hello"}),
                json.dumps(
                    {
                        "type": "translation.done",
                        "usage": {"completion_tokens": 3},
                    }
                ),
            ]

        async def send(
            self,
            payload: str,
            /,
        ) -> None:
            captured_payload.update(json.loads(payload))

        async def recv(
            self,
            /,
        ) -> str:
            return self.events.pop(0)

    class FakeConnection:
        __slots__: ClassVar[tuple[str, ...]] = ("websocket",)

        def __init__(
            self,
            /,
        ) -> None:
            self.websocket = FakeWebSocket()

        async def __aenter__(
            self,
            /,
        ) -> FakeWebSocket:
            return self.websocket

        async def __aexit__(
            self,
            error_type: type[BaseException] | None,
            error: BaseException | None,
            traceback: object | None,
            /,
        ) -> None:
            del error_type, error, traceback

    def fake_connect(
        uri: str,
        /,
        **options: Any,
    ) -> FakeConnection:
        captured_connection.update({"uri": uri, **options})
        return FakeConnection()

    monkeypatch.setattr("kotonoha.clients._llm.connect", fake_connect)
    client = LanguageModelClient("http://test", settings.llm)
    try:
        chunks = [
            chunk
            async for chunk in client.stream_chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "source_lang_code": "ko",
                                "target_lang_code": "en",
                                "text": "Translate this.",
                            }
                        ],
                    }
                ]
            )
        ]
    finally:
        await client.aclose()

    assert chunks == ["Hello"]
    assert captured_connection["uri"] == "ws://test/v1/realtime"
    assert captured_payload["type"] == "translation.create"
    assert captured_payload["model"] == "kotonoha-translation"
    assert captured_payload["repetition_penalty"] == 1.0
    assert captured_payload["messages"][0]["content"][0]["source_lang_code"] == "ko"
    assert "repeat_penalty" not in captured_payload
    assert "cache_prompt" not in captured_payload


async def test_vllm_omni_tts_request_streams_openai_compatible_pcm() -> None:
    settings = load_settings()
    captured_payload: dict[str, Any] = {}

    def handle_request(
        request: httpx.Request,
        /,
    ) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(200, content=b"\x00\x00\xff\x7f")

    client = TextToSpeechClient("http://test", settings.tts)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handle_request),
    )
    try:
        chunks = [chunk async for chunk in client.synthesize("안녕하세요.", "ko")]
    finally:
        await client.aclose()

    assert len(chunks) == 1
    assert chunks[0].dtype == np.float32
    assert chunks[0].tolist() == pytest.approx([0.0, 32767 / 32768])
    assert captured_payload == {
        "input": "안녕하세요.",
        "model": "kotonoha-tts",
        "voice": "Sohee",
        "language": "Korean",
        "task_type": "CustomVoice",
        "response_format": "pcm",
        "stream": True,
        "stream_format": "audio",
    }


async def test_vllm_omni_health_accepts_an_empty_success_response() -> None:
    settings = load_settings()

    def handle_request(
        request: httpx.Request,
        /,
    ) -> httpx.Response:
        return httpx.Response(200)

    client = TextToSpeechClient("http://test", settings.tts)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handle_request),
    )
    try:
        health = await client.health()
    finally:
        await client.aclose()

    assert health == {"ok": True, "service": "tts", "status": 200, "side": "local"}


# -- failover ---------------------------------------------------------------
class FakeClient:
    """Stands in for a service client; fails on demand."""
    __slots__: ClassVar[tuple[str, ...]] = (
        "calls",
        "fail",
        "healthy",
        "side",
    )

    name: Final = "asr"

    def __init__(
        self,
        /,
        side: str,
        fail: bool = False,
    ) -> None:
        self.side = side
        self.fail = fail
        self.calls = 0
        self.healthy = True

    @property
    def label(
        self,
        /,
    ) -> str:
        return f"{self.name}@{self.side}"

    async def work(
        self,
        /,
    ) -> str:
        self.calls += 1
        if self.fail:
            raise ServiceTimeout(f"{self.label} down")
        return self.side

    async def health(
        self,
        /,
    ) -> dict:
        return {"ok": self.healthy, "side": self.side}

    async def aclose(
        self,
        /,
    ) -> None:
        return None


def _route(
    remote_fails: bool,
    /,
    **config_overrides: Any,
) -> tuple[FailoverClient, FakeClient, FakeClient]:
    remote = FakeClient("remote", fail=remote_fails)
    local = FakeClient("local")
    failover_client = FailoverClient(
        "asr",
        remote,
        local,
        RemoteConfig(failover_after=2, **config_overrides),
    )
    return failover_client, remote, local


async def test_healthy_remote_is_used() -> None:
    failover_client, remote, local = _route(False)
    assert await failover_client.run(lambda client: client.work()) == "remote"
    assert local.calls == 0
    assert not failover_client.degraded


async def test_the_failing_turn_still_completes_on_the_fallback() -> None:
    """A link failure must cost a retry, not the turn (§10)."""
    failover_client, remote, local = _route(True)
    assert await failover_client.run(lambda client: client.work()) == "local"
    assert remote.calls == 1 and local.calls == 1
    assert failover_client.failover_count == 1


async def test_role_degrades_only_after_the_configured_streak() -> None:
    failover_client, remote, _ = _route(True)
    await failover_client.run(lambda client: client.work())
    assert not failover_client.degraded, "one failure should not move the placement"
    await failover_client.run(lambda client: client.work())
    assert failover_client.degraded and failover_client.side == "local"


async def test_a_success_resets_the_streak() -> None:
    failover_client, remote, _ = _route(True)
    await failover_client.run(lambda client: client.work())
    remote.fail = False
    await failover_client.run(lambda client: client.work())
    remote.fail = True
    await failover_client.run(lambda client: client.work())
    assert not failover_client.degraded


async def test_both_sides_down_raises_rather_than_hanging() -> None:
    failover_client, remote, local = _route(True)
    local.fail = True
    with pytest.raises(AllEndpointsFailed):
        await failover_client.run(lambda client: client.work())


async def test_no_fallback_propagates_the_error() -> None:
    remote = FakeClient("remote", fail=True)
    failover_client = FailoverClient("asr", remote, None, RemoteConfig())
    with pytest.raises(ServiceTimeout):
        await failover_client.run(lambda client: client.work())


async def test_stream_fails_over_before_the_first_chunk() -> None:
    async def generate(
        client: Any,
        /,
    ) -> Any:
        if client.fail:
            raise ServiceError("cold failure")
        for item in ("a", "b"):
            yield item

    failover_client, remote, local = _route(True)
    output = [item async for item in failover_client.stream(generate)]
    assert output == ["a", "b"]


async def test_stream_does_not_fail_over_midway() -> None:
    """Once audio is playing there is no clean way to rewind, so report it."""

    async def generate(
        client: Any,
        /,
    ) -> Any:
        yield "a"
        if client.fail:
            raise ServiceError("died mid-stream")
        yield "b"

    failover_client, remote, local = _route(True)
    with pytest.raises(ServiceError):
        _ = [item async for item in failover_client.stream(generate)]
    assert local.calls == 0


async def test_placement_change_is_reported() -> None:
    seen = []
    remote = FakeClient("remote", fail=True)
    local = FakeClient("local")
    failover_client = FailoverClient(
        "asr",
        remote,
        local,
        RemoteConfig(failover_after=1),
        on_change=lambda role, side, why: seen.append((role, side)),
    )
    await failover_client.run(lambda client: client.work())
    assert seen == [("asr", "local")]
