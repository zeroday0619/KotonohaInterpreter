"""Audio device discovery and stream probe behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np

from kotonoha.audio import _devices
from kotonoha.audio._devices import (
    AudioDevice,
    probe_audio_devices,
    query_audio_devices,
    resolve_audio_stream,
    select_mono_input,
)


class FakeStream:
    __slots__: ClassVar[tuple[str, ...]] = (
        "calls",
        "input_amplitude",
    )

    def __init__(
        self,
        /,
        calls: list[str],
        input_amplitude: float = 0.02,
    ) -> None:
        self.calls = calls
        self.input_amplitude = input_amplitude

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

    def read(
        self,
        frame_count: int,
        /,
    ) -> tuple[np.ndarray, bool]:
        self.calls.append(f"read:{frame_count}")
        return np.full((frame_count, 1), self.input_amplitude, dtype=np.float32), False

    def write(
        self,
        samples: np.ndarray,
        /,
    ) -> bool:
        self.calls.append(f"write:{samples.shape}")
        return False


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
                    "index": 0,
                    "name": "Microphone",
                    "hostapi": 0,
                    "max_input_channels": 2,
                    "max_output_channels": 0,
                    "default_samplerate": 48000,
                },
            ],
            query_hostapis=lambda: [{"name": "ALSA"}],
        ),
    )

    devices = query_audio_devices()

    assert devices == (AudioDevice(0, "Microphone", "ALSA", 2, 0, 48000.0),)
    assert devices[0].selector == "Microphone, ALSA"
    assert devices[0].label == "Microphone [ALSA] (2 in, 0 out, 48000 Hz)"


def test_probe_audio_devices_opens_both_streams(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: Any,
) -> None:
    input_calls: list[str] = []
    output_calls: list[str] = []
    device_information = {
        1: {
            "index": 1,
            "name": "Microphone",
            "hostapi": 0,
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 48000,
        },
        2: {
            "index": 2,
            "name": "Speaker",
            "hostapi": 0,
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 48000,
        },
    }
    fake_sounddevice = SimpleNamespace(
        query_devices=lambda device, kind: device_information[device],
        query_hostapis=lambda index: {"name": "ALSA"},
        check_input_settings=lambda **kwargs: input_calls.append(str(kwargs)),
        check_output_settings=lambda **kwargs: output_calls.append(str(kwargs)),
        InputStream=lambda **kwargs: FakeStream(input_calls),
        OutputStream=lambda **kwargs: FakeStream(output_calls),
    )
    monkeypatch.setattr(_devices, "_sounddevice", lambda: fake_sounddevice)

    result = probe_audio_devices(
        1,
        2,
        capture_sample_rate=48000,
        playback_sample_rate=24000,
        channels=1,
    )

    assert result.input_ok
    assert result.output_ok
    assert result.input_signal_detected
    assert result.input_sample_rate == 48000
    assert result.output_sample_rate == 24000
    assert result.input_device == "Microphone, ALSA"
    assert result.output_device == "Speaker, ALSA"
    assert input_calls[-4:] == ["start", "read:36000", "stop", "close"]
    assert output_calls[-4:] == ["start", "write:(6000, 1)", "stop", "close"]


def test_resolve_audio_stream_uses_native_stereo_when_requested_format_is_rejected(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: Any,
) -> None:
    attempts: list[tuple[int, int]] = []

    def check_output_settings(
        _positional_only: object | None = None,
        /,
        **settings: Any,
    ) -> None:
        del _positional_only
        attempt = (settings["samplerate"], settings["channels"])
        attempts.append(attempt)
        if attempt != (48000, 2):
            raise ValueError("unsupported format")

    fake_sounddevice = SimpleNamespace(
        query_devices=lambda device, kind: {
            "index": 7,
            "name": "USB Speaker",
            "hostapi": 0,
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 48000,
        },
        query_hostapis=lambda index: {"name": "ALSA"},
        check_output_settings=check_output_settings,
    )
    monkeypatch.setattr(_devices, "_sounddevice", lambda: fake_sounddevice)

    settings = resolve_audio_stream(
        "USB Speaker, ALSA",
        "output",
        requested_sample_rate=24000,
        requested_channels=1,
    )

    assert attempts == [(24000, 1), (24000, 2), (48000, 1), (48000, 2)]
    assert settings.device_index == 7
    assert settings.selector == "USB Speaker, ALSA"
    assert settings.sample_rate == 48000
    assert settings.channels == 2


def test_probe_reports_a_stream_that_only_returns_digital_silence(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: Any,
) -> None:
    devices = {
        1: {
            "index": 1,
            "name": "Silent Input",
            "hostapi": 0,
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 48000,
        },
        2: {
            "index": 2,
            "name": "Speaker",
            "hostapi": 0,
            "max_input_channels": 0,
            "max_output_channels": 1,
            "default_samplerate": 24000,
        },
    }
    fake_sounddevice = SimpleNamespace(
        query_devices=lambda device, kind: devices[device],
        query_hostapis=lambda index: {"name": "ALSA"},
        check_input_settings=lambda **settings: None,
        check_output_settings=lambda **settings: None,
        InputStream=lambda **settings: FakeStream([], input_amplitude=0.0),
        OutputStream=lambda **settings: FakeStream([]),
    )
    monkeypatch.setattr(_devices, "_sounddevice", lambda: fake_sounddevice)

    result = probe_audio_devices(
        1,
        2,
        capture_sample_rate=48000,
        playback_sample_rate=24000,
        channels=1,
    )

    assert result.input_ok
    assert result.output_ok
    assert result.input_signal_detected is False
    assert not result.ok
    assert result.input_peak_dbfs == -240.0


def test_mono_capture_selects_the_channel_that_contains_the_signal() -> None:
    input_data = np.column_stack(
        (
            np.zeros(4, dtype=np.float32),
            np.array([0.0, 0.25, -0.5, 0.0], dtype=np.float32),
        )
    )

    selected = select_mono_input(input_data)

    assert np.array_equal(selected, input_data[:, 1])
