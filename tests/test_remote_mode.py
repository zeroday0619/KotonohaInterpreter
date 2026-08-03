"""High-performance mode: role placement, audio transport, and link failover."""

from __future__ import annotations

import numpy as np
import pytest

from kotonoha.clients.base import ServiceError, ServiceTimeout, remote_transport_kwargs
from kotonoha.clients.router import AllEndpointsFailed, FailoverClient
from kotonoha.config import RemoteCfg, load_settings
from kotonoha.transport import AudioPayload, decode_pcm, encode_pcm, encoded_size


# -- placement -------------------------------------------------------------
def _settings(**over):
    s = load_settings()
    for k, v in over.items():
        setattr(s, k, v)
    return s


def test_onboard_keeps_everything_local():
    s = _settings(perf_mode="onboard")
    s.remote.enabled = True
    assert s.resolved_placement() == {
        "asr": "local",
        "asr_verify": "local",
        "llm": "local",
        "tts": "local",
    }
    assert not s.audio_leaves_device


def test_hybrid_moves_only_the_llm_and_keeps_audio_on_device():
    """The point of hybrid: the biggest win, without shipping any audio."""
    s = _settings(perf_mode="hybrid")
    s.remote.enabled = True
    p = s.resolved_placement()
    assert p["llm"] == "remote"
    assert p["asr"] == p["asr_verify"] == p["tts"] == "local"
    assert not s.audio_leaves_device


def test_remote_moves_everything_and_flags_the_audio():
    s = _settings(perf_mode="remote")
    s.remote.enabled = True
    assert set(s.resolved_placement().values()) == {"remote"}
    assert s.audio_leaves_device


def test_disabled_remote_collapses_to_local():
    """A mode pointing at an unreachable box would just be a per-turn timeout."""
    s = _settings(perf_mode="remote")
    s.remote.enabled = False
    assert set(s.resolved_placement().values()) == {"local"}
    assert not s.audio_leaves_device


def test_explicit_placement_overrides_the_mode():
    s = _settings(perf_mode="onboard", placement={"llm": "remote"})
    s.remote.enabled = True
    p = s.resolved_placement()
    assert p["llm"] == "remote" and p["asr"] == "local"


def test_unknown_role_in_placement_is_rejected():
    s = _settings(placement={"vocoder": "remote"})
    s.remote.enabled = True
    with pytest.raises(ValueError, match="unknown role"):
        s.resolved_placement()


def test_url_selection_follows_the_side():
    s = _settings()
    s.remote.services.llm = "http://a6000.lan:8003"
    assert s.url_for("llm", "local").startswith("http://127.0.0.1")
    assert s.url_for("llm", "remote") == "http://a6000.lan:8003"


# -- transport --------------------------------------------------------------
def test_pcm_roundtrip_s16le_is_lossy_but_close():
    x = np.linspace(-1, 1, 4000, dtype=np.float32)
    y = decode_pcm(encode_pcm(x, "s16le"), "s16le")
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1e-4


def test_pcm_roundtrip_f32le_is_exact():
    x = np.linspace(-1, 1, 4000, dtype=np.float32)
    y = decode_pcm(encode_pcm(x, "f32le"), "f32le")
    assert np.array_equal(x, y)


def test_s16le_halves_the_bytes_on_the_wire():
    assert encoded_size(6.0, 16000, "s16le") == 192_000
    assert encoded_size(6.0, 16000, "f32le") == 384_000


def test_payload_carries_both_forms():
    pcm = np.zeros(16000, dtype=np.float32)
    p = AudioPayload(pcm=pcm, ref=None)
    assert p.seconds == pytest.approx(1.0)
    assert len(p.encoded("s16le")) == 32_000


def test_bearer_token_and_tls_flags_reach_httpx():
    tk = remote_transport_kwargs(RemoteCfg(token="secret", verify_tls=False))
    assert tk["headers"]["authorization"] == "Bearer secret"
    assert tk["verify"] is False


# -- failover ---------------------------------------------------------------
class FakeClient:
    """Stands in for a service client; fails on demand."""

    name = "asr"

    def __init__(self, side: str, fail: bool = False):
        self.side = side
        self.fail = fail
        self.calls = 0
        self.healthy = True

    @property
    def label(self) -> str:
        return f"{self.name}@{self.side}"

    async def work(self) -> str:
        self.calls += 1
        if self.fail:
            raise ServiceTimeout(f"{self.label} down")
        return self.side

    async def health(self) -> dict:
        return {"ok": self.healthy, "side": self.side}

    async def aclose(self) -> None:
        return None


def _route(remote_fails: bool, **cfg) -> tuple[FailoverClient, FakeClient, FakeClient]:
    remote = FakeClient("remote", fail=remote_fails)
    local = FakeClient("local")
    fc = FailoverClient("asr", remote, local, RemoteCfg(failover_after=2, **cfg))
    return fc, remote, local


async def test_healthy_remote_is_used():
    fc, remote, local = _route(remote_fails=False)
    assert await fc.run(lambda c: c.work()) == "remote"
    assert local.calls == 0
    assert not fc.degraded


async def test_the_failing_turn_still_completes_on_the_fallback():
    """A link failure must cost a retry, not the turn (§10)."""
    fc, remote, local = _route(remote_fails=True)
    assert await fc.run(lambda c: c.work()) == "local"
    assert remote.calls == 1 and local.calls == 1
    assert fc.failover_count == 1


async def test_role_degrades_only_after_the_configured_streak():
    fc, remote, _ = _route(remote_fails=True)
    await fc.run(lambda c: c.work())
    assert not fc.degraded, "one failure should not move the placement"
    await fc.run(lambda c: c.work())
    assert fc.degraded and fc.side == "local"


async def test_a_success_resets_the_streak():
    fc, remote, _ = _route(remote_fails=True)
    await fc.run(lambda c: c.work())
    remote.fail = False
    await fc.run(lambda c: c.work())
    remote.fail = True
    await fc.run(lambda c: c.work())
    assert not fc.degraded


async def test_both_sides_down_raises_rather_than_hanging():
    fc, remote, local = _route(remote_fails=True)
    local.fail = True
    with pytest.raises(AllEndpointsFailed):
        await fc.run(lambda c: c.work())


async def test_no_fallback_propagates_the_error():
    remote = FakeClient("remote", fail=True)
    fc = FailoverClient("asr", remote, None, RemoteCfg())
    with pytest.raises(ServiceTimeout):
        await fc.run(lambda c: c.work())


async def test_stream_fails_over_before_the_first_chunk():
    async def gen(client):
        if client.fail:
            raise ServiceError("cold failure")
        for x in ("a", "b"):
            yield x

    fc, remote, local = _route(remote_fails=True)
    got = [x async for x in fc.stream(gen)]
    assert got == ["a", "b"]


async def test_stream_does_not_fail_over_midway():
    """Once audio is playing there is no clean way to rewind, so report it."""

    async def gen(client):
        yield "a"
        if client.fail:
            raise ServiceError("died mid-stream")
        yield "b"

    fc, remote, local = _route(remote_fails=True)
    with pytest.raises(ServiceError):
        _ = [x async for x in fc.stream(gen)]
    assert local.calls == 0


async def test_placement_change_is_reported():
    seen = []
    remote = FakeClient("remote", fail=True)
    local = FakeClient("local")
    fc = FailoverClient(
        "asr",
        remote,
        local,
        RemoteCfg(failover_after=1),
        on_change=lambda role, side, why: seen.append((role, side)),
    )
    await fc.run(lambda c: c.work())
    assert seen == [("asr", "local")]
