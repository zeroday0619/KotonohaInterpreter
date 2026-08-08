"""Audio device discovery, stable selection, and non-destructive stream probes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from structlog import get_logger

log = get_logger(__name__)

AudioDeviceIdentifier = int | str | None
AudioDirection = Literal["input", "output"]

_INPUT_PROBE_SECONDS = 0.75
_OUTPUT_PROBE_SECONDS = 0.25
_OUTPUT_PROBE_FREQUENCY_HZ = 440.0
_OUTPUT_PROBE_AMPLITUDE = 0.08
_MINIMUM_INPUT_PEAK = 1e-4


@dataclass(frozen=True, slots=True)
class AudioDevice:
    """PortAudio device metadata used by the configuration editor."""

    index: int
    name: str
    host_api: str
    input_channels: int
    output_channels: int
    default_sample_rate: float

    @property
    def selector(
        self,
        /,
    ) -> str:
        """Return a selector that survives PortAudio index changes."""
        return f"{self.name}, {self.host_api}"

    @property
    def label(
        self,
        /,
    ) -> str:
        return (
            f"{self.name} [{self.host_api}] "
            f"({self.input_channels} in, {self.output_channels} out, "
            f"{self.default_sample_rate:.0f} Hz)"
        )


@dataclass(frozen=True, slots=True)
class AudioStreamSettings:
    """Resolved stream settings accepted by the selected PortAudio device."""

    device_index: int
    selector: str
    name: str
    host_api: str
    sample_rate: int
    channels: int


@dataclass(frozen=True, slots=True)
class AudioProbeResult:
    """Results from reading the microphone and writing an audible test tone."""

    input_ok: bool
    output_ok: bool
    input_error: str | None = None
    output_error: str | None = None
    input_signal_detected: bool | None = None
    input_peak_dbfs: float | None = None
    input_rms_dbfs: float | None = None
    input_sample_rate: int | None = None
    output_sample_rate: int | None = None
    input_device: str | None = None
    output_device: str | None = None

    @property
    def ok(
        self,
        /,
    ) -> bool:
        signal_detected = self.input_signal_detected is not False
        return self.input_ok and self.output_ok and signal_detected


def _sounddevice() -> Any:
    import sounddevice

    return sounddevice


def query_audio_devices() -> tuple[AudioDevice, ...]:
    """Return the PortAudio devices visible to the current process."""
    sounddevice = _sounddevice()
    devices = sounddevice.query_devices()
    host_apis = sounddevice.query_hostapis()
    return tuple(
        AudioDevice(
            index=int(device.get("index", index)),
            name=str(device["name"]),
            host_api=str(host_apis[int(device["hostapi"])]["name"]),
            input_channels=int(device.get("max_input_channels", 0)),
            output_channels=int(device.get("max_output_channels", 0)),
            default_sample_rate=float(device.get("default_samplerate", 0.0)),
        )
        for index, device in enumerate(devices)
    )


def resolve_audio_stream(
    device: AudioDeviceIdentifier,
    direction: AudioDirection,
    /,
    *,
    requested_sample_rate: int,
    requested_channels: int,
) -> AudioStreamSettings:
    """Resolve a stable selector and negotiate settings accepted by PortAudio."""
    sounddevice = _sounddevice()
    device_information = sounddevice.query_devices(device, kind=direction)
    device_index = int(device_information["index"])
    host_api_information = sounddevice.query_hostapis(int(device_information["hostapi"]))
    name = str(device_information["name"])
    host_api = str(host_api_information["name"])
    maximum_channels = int(device_information[f"max_{direction}_channels"])
    if maximum_channels < 1:
        raise ValueError(f"Selected device has no {direction} channels: {name}, {host_api}")

    default_sample_rate = int(round(float(device_information["default_samplerate"])))
    sample_rates = _unique_positive((requested_sample_rate, default_sample_rate))
    channel_counts = _candidate_channel_counts(requested_channels, maximum_channels)
    check_settings = (
        sounddevice.check_input_settings
        if direction == "input"
        else sounddevice.check_output_settings
    )

    failures: list[str] = []
    last_error: Exception | None = None
    for sample_rate in sample_rates:
        for channels in channel_counts:
            try:
                check_settings(
                    device=device_index,
                    samplerate=sample_rate,
                    channels=channels,
                    dtype="float32",
                )
            except Exception as error:  # noqa: BLE001 - retain every PortAudio rejection
                last_error = error
                failures.append(f"{sample_rate} Hz/{channels} ch: {error}")
                continue
            return AudioStreamSettings(
                device_index=device_index,
                selector=f"{name}, {host_api}",
                name=name,
                host_api=host_api,
                sample_rate=sample_rate,
                channels=channels,
            )

    attempts = "; ".join(failures)
    message = f"No supported {direction} format for {name}, {host_api}: {attempts}"
    raise RuntimeError(message) from last_error


def probe_audio_devices(
    input_device: AudioDeviceIdentifier,
    output_device: AudioDeviceIdentifier,
    /,
    *,
    capture_sample_rate: int,
    playback_sample_rate: int,
    channels: int,
) -> AudioProbeResult:
    """Read microphone samples and submit an audible tone to the output stream."""
    sounddevice = _sounddevice()
    input_error: str | None = None
    output_error: str | None = None
    input_settings: AudioStreamSettings | None = None
    output_settings: AudioStreamSettings | None = None
    input_signal_detected: bool | None = None
    input_peak_dbfs: float | None = None
    input_rms_dbfs: float | None = None

    try:
        input_settings = resolve_audio_stream(
            input_device,
            "input",
            requested_sample_rate=capture_sample_rate,
            requested_channels=channels,
        )
        samples = _probe_input(sounddevice, input_settings)
        input_peak_dbfs, input_rms_dbfs = _signal_levels(samples)
        input_signal_detected = bool(np.max(np.abs(samples), initial=0.0) >= _MINIMUM_INPUT_PEAK)
    except Exception as error:  # noqa: BLE001 - expose complete PortAudio diagnostics
        input_error = str(error)

    try:
        output_settings = resolve_audio_stream(
            output_device,
            "output",
            requested_sample_rate=playback_sample_rate,
            requested_channels=1,
        )
        _probe_output(sounddevice, output_settings)
    except Exception as error:  # noqa: BLE001 - expose complete PortAudio diagnostics
        output_error = str(error)

    return AudioProbeResult(
        input_ok=input_error is None,
        output_ok=output_error is None,
        input_error=input_error,
        output_error=output_error,
        input_signal_detected=input_signal_detected,
        input_peak_dbfs=input_peak_dbfs,
        input_rms_dbfs=input_rms_dbfs,
        input_sample_rate=input_settings.sample_rate if input_settings is not None else None,
        output_sample_rate=output_settings.sample_rate if output_settings is not None else None,
        input_device=input_settings.selector if input_settings is not None else None,
        output_device=output_settings.selector if output_settings is not None else None,
    )


def _candidate_channel_counts(
    requested_channels: int,
    maximum_channels: int,
    /,
) -> tuple[int, ...]:
    if requested_channels < 1:
        raise ValueError(f"Channel count must be positive: {requested_channels}")
    if requested_channels > maximum_channels:
        raise ValueError(
            f"Requested {requested_channels} channels but the device supports {maximum_channels}"
        )
    # Some ALSA hardware endpoints reject mono streams even though PortAudio reports
    # two available channels. A stereo fallback keeps direct hardware devices usable.
    fallback_channels = 2 if requested_channels == 1 and maximum_channels >= 2 else None
    return _unique_positive((requested_channels, fallback_channels))


def _unique_positive(
    values: tuple[int | None, ...],
    /,
) -> tuple[int, ...]:
    output: list[int] = []
    for value in values:
        if value is not None and value > 0 and value not in output:
            output.append(value)
    return tuple(output)


def _probe_input(
    sounddevice: Any,
    settings: AudioStreamSettings,
    /,
) -> np.ndarray:
    stream = sounddevice.InputStream(
        device=settings.device_index,
        samplerate=settings.sample_rate,
        channels=settings.channels,
        dtype="float32",
        blocksize=0,
    )
    try:
        stream.start()
        frame_count = max(1, int(settings.sample_rate * _INPUT_PROBE_SECONDS))
        input_data, overflowed = stream.read(frame_count)
        if overflowed:
            raise RuntimeError("Input overflow occurred during the microphone probe")
        return select_mono_input(input_data)
    finally:
        _close_stream(stream)


def _probe_output(
    sounddevice: Any,
    settings: AudioStreamSettings,
    /,
) -> None:
    frame_count = max(1, int(settings.sample_rate * _OUTPUT_PROBE_SECONDS))
    phase = np.arange(frame_count, dtype=np.float32) / float(settings.sample_rate)
    tone = np.sin(2.0 * math.pi * _OUTPUT_PROBE_FREQUENCY_HZ * phase).astype(np.float32)
    fade_frames = min(frame_count // 2, max(1, int(settings.sample_rate * 0.01)))
    envelope = np.ones(frame_count, dtype=np.float32)
    envelope[:fade_frames] = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
    envelope[-fade_frames:] = np.linspace(1.0, 0.0, fade_frames, dtype=np.float32)
    samples = (_OUTPUT_PROBE_AMPLITUDE * tone * envelope)[:, np.newaxis]
    if settings.channels > 1:
        samples = np.repeat(samples, settings.channels, axis=1)

    stream = sounddevice.OutputStream(
        device=settings.device_index,
        samplerate=settings.sample_rate,
        channels=settings.channels,
        dtype="float32",
        blocksize=0,
    )
    try:
        stream.start()
        underflowed = stream.write(samples)
        if underflowed:
            raise RuntimeError("Output underflow occurred during the speaker probe")
    finally:
        _close_stream(stream)


def _signal_levels(
    samples: np.ndarray,
    /,
) -> tuple[float, float]:
    if samples.size == 0:
        return _to_dbfs(0.0), _to_dbfs(0.0)
    absolute_peak = float(np.max(np.abs(samples), initial=0.0))
    root_mean_square = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    return _to_dbfs(absolute_peak), _to_dbfs(root_mean_square)


def select_mono_input(
    samples: Any,
    /,
) -> np.ndarray:
    """Select the active input channel instead of assuming channel zero carries speech."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        return audio.reshape(-1)
    if audio.ndim != 2 or audio.shape[1] == 0:
        raise ValueError(f"Expected audio frames by channels, got shape {audio.shape}")
    if audio.shape[1] == 1:
        return audio[:, 0].reshape(-1)
    if audio.shape[1] == 2:
        first_energy = float(np.dot(audio[:, 0], audio[:, 0]))
        second_energy = float(np.dot(audio[:, 1], audio[:, 1]))
        active_channel = 0 if first_energy >= second_energy else 1
        return np.asarray(audio[:, active_channel], dtype=np.float32).reshape(-1)
    # einsum avoids allocating a block-sized temporary in the capture worker.
    channel_energy = np.einsum("ij,ij->j", audio, audio, optimize=False)
    active_channel = int(np.argmax(channel_energy))
    return np.asarray(audio[:, active_channel], dtype=np.float32).reshape(-1)


def _to_dbfs(
    amplitude: float,
    /,
) -> float:
    return round(20.0 * math.log10(max(amplitude, 1e-12)), 1)


def _close_stream(
    stream: Any,
    /,
) -> None:
    for method_name in ("stop", "close"):
        try:
            getattr(stream, method_name)()
        except Exception as error:  # noqa: BLE001 - preserve the original probe failure
            log.debug(
                "audio.probe_stream_close_failed",
                method=method_name,
                error=repr(error),
            )
