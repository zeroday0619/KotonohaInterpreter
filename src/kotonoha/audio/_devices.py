"""Audio device discovery and non-destructive stream probes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

AudioDeviceIdentifier = int | str | None


@dataclass(frozen=True, slots=True)
class AudioDevice:
    """PortAudio device metadata used by the configuration editor."""

    index: int
    name: str
    input_channels: int
    output_channels: int
    default_sample_rate: float

    @property
    def label(
        self,
        /,
    ) -> str:
        return (
            f"{self.index}: {self.name} "
            f"({self.input_channels} in, {self.output_channels} out)"
        )


@dataclass(frozen=True, slots=True)
class AudioProbeResult:
    """Results from opening the configured input and output streams."""

    input_ok: bool
    output_ok: bool
    input_error: str | None = None
    output_error: str | None = None

    @property
    def ok(
        self,
        /,
    ) -> bool:
        return self.input_ok and self.output_ok


def _sounddevice() -> Any:
    import sounddevice

    return sounddevice


def query_audio_devices() -> tuple[AudioDevice, ...]:
    """Return the PortAudio devices visible to the current process."""
    sounddevice = _sounddevice()
    devices = sounddevice.query_devices()
    return tuple(
        AudioDevice(
            index=index,
            name=str(device["name"]),
            input_channels=int(device.get("max_input_channels", 0)),
            output_channels=int(device.get("max_output_channels", 0)),
            default_sample_rate=float(device.get("default_samplerate", 0.0)),
        )
        for index, device in enumerate(devices)
    )


def probe_audio_devices(
    input_device: AudioDeviceIdentifier,
    output_device: AudioDeviceIdentifier,
    /,
    *,
    capture_sample_rate: int,
    playback_sample_rate: int,
    channels: int,
) -> AudioProbeResult:
    """Open both configured streams briefly and report device-level failures."""
    sounddevice = _sounddevice()
    input_error: str | None = None
    output_error: str | None = None
    try:
        _probe_input(
            sounddevice,
            input_device,
            capture_sample_rate,
            channels,
        )
    except Exception as error:  # noqa: BLE001 - expose PortAudio diagnostics in the TUI
        input_error = str(error)
    try:
        _probe_output(
            sounddevice,
            output_device,
            playback_sample_rate,
            channels,
        )
    except Exception as error:  # noqa: BLE001 - expose PortAudio diagnostics in the TUI
        output_error = str(error)
    return AudioProbeResult(
        input_ok=input_error is None,
        output_ok=output_error is None,
        input_error=input_error,
        output_error=output_error,
    )


def _probe_input(
    sounddevice: Any,
    /,
    device: AudioDeviceIdentifier,
    sample_rate: int,
    channels: int,
) -> None:
    sounddevice.check_input_settings(
        device=device,
        samplerate=sample_rate,
        channels=channels,
    )
    stream = sounddevice.InputStream(
        device=device,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
    )
    try:
        stream.start()
        time.sleep(0.1)
    finally:
        _close_stream(stream)


def _probe_output(
    sounddevice: Any,
    /,
    device: AudioDeviceIdentifier,
    sample_rate: int,
    channels: int,
) -> None:
    sounddevice.check_output_settings(
        device=device,
        samplerate=sample_rate,
        channels=channels,
    )

    def output_callback(
        output_data: Any,
        /,
        frame_count: Any,
        time_info: Any,
        status: Any,
    ) -> None:
        del frame_count, time_info, status
        output_data.fill(0.0)

    stream = sounddevice.OutputStream(
        device=device,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        callback=output_callback,
    )
    try:
        stream.start()
        time.sleep(0.1)
    finally:
        _close_stream(stream)


def _close_stream(
    stream: Any,
    /,
) -> None:
    for method_name in ("stop", "close"):
        try:
            getattr(stream, method_name)()
        except Exception:  # noqa: BLE001 - preserve the original probe failure
            continue
