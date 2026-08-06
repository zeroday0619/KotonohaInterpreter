"""Audio device discovery and stream probe behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

from kotonoha.audio import _devices
from kotonoha.audio._devices import (
    AudioDevice,
    AudioProbeResult,
    probe_audio_devices,
    query_audio_devices,
)


class FakeStream:
    __slots__: ClassVar[tuple[str, ...]] = ("calls",)

    def __init__(
        self,
        /,
        calls: list[str],
    ) -> None:
        self.calls = calls

    def start(
        self,
        /,
    ) -> None:
        self.calls.append("start")

    def stop(
        self,
        /,
    ) -> None:
        self.calls.append("stop")

    def close(
        self,
        /,
    ) -> None:
        self.calls.append("close")


def test_query_audio_devices_returns_channel_metadata(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        _devices,
        "_sounddevice",
        lambda: SimpleNamespace(
            query_devices=lambda: [
                {
                    "name": "Microphone",
                    "max_input_channels": 2,
                    "max_output_channels": 0,
                    "default_samplerate": 48000,
                },
            ],
        ),
    )

    devices = query_audio_devices()

    assert devices == (AudioDevice(0, "Microphone", 2, 0, 48000.0),)
    assert devices[0].label == "0: Microphone (2 in, 0 out)"


def test_probe_audio_devices_opens_both_streams(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: Any,
) -> None:
    input_calls: list[str] = []
    output_calls: list[str] = []
    fake_sounddevice = SimpleNamespace(
        check_input_settings=lambda **kwargs: input_calls.append(str(kwargs)),
        check_output_settings=lambda **kwargs: output_calls.append(str(kwargs)),
        InputStream=lambda **kwargs: FakeStream(input_calls),
        OutputStream=lambda **kwargs: FakeStream(output_calls),
    )
    monkeypatch.setattr(_devices, "_sounddevice", lambda: fake_sounddevice)
    monkeypatch.setattr(_devices.time, "sleep", lambda seconds: None)

    result = probe_audio_devices(
        1,
        2,
        capture_sample_rate=48000,
        playback_sample_rate=24000,
        channels=1,
    )

    assert result == AudioProbeResult(True, True)
    assert input_calls[1:] == ["start", "stop", "close"]
    assert output_calls[1:] == ["start", "stop", "close"]
