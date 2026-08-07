#!/usr/bin/env python3
"""Measure deployed TTS intelligibility through an ASR round trip."""

from __future__ import annotations

import argparse
import asyncio
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from kotonoha._config import Settings, load_settings
from kotonoha._transport import AudioPayload, encode_pcm
from kotonoha.audio._resample import resample_once
from kotonoha.clients._asr import AsrClient
from kotonoha.clients._base import remote_transport_kwargs
from kotonoha.clients._tts import TextToSpeechClient
from kotonoha.core._quality import character_error_rate


@dataclass(frozen=True, slots=True)
class SpeechProbe:
    """One native-language phrase and its expected TTS voice."""

    language: str
    text: str


PROBES: tuple[SpeechProbe, ...] = (
    SpeechProbe("ko", "안녕하세요. 지금 통역기의 음성 품질을 확인하고 있습니다."),
    SpeechProbe("en", "Hello. This test verifies the interpreter speech quality."),
    SpeechProbe("ja", "こんにちは。通訳機の音声品質を確認しています。"),
    SpeechProbe("zh-TW", "您好，我們正在確認口譯裝置的語音品質。"),
)


class SpeechQualityRunner:
    """Synthesize fixed phrases, save them, and transcribe the generated audio."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "asr_client",
        "max_character_error_rate",
        "output_directory",
        "settings",
        "tts_client",
    )

    settings: Settings
    output_directory: Path
    max_character_error_rate: float
    asr_client: AsrClient
    tts_client: TextToSpeechClient

    def __init__(
        self,
        settings: Settings,
        output_directory: Path,
        max_character_error_rate: float,
        /,
    ) -> None:
        self.settings = settings
        self.output_directory = output_directory
        self.max_character_error_rate = max_character_error_rate
        placement = settings.resolved_placement()
        asr_transport = (
            remote_transport_kwargs(settings.remote)
            if placement["asr"] == "remote"
            else {}
        )
        tts_transport = (
            remote_transport_kwargs(settings.remote)
            if placement["tts"] == "remote"
            else {}
        )
        # The upload endpoint works on both hosts and avoids requiring shared memory
        # in this standalone diagnostic process.
        self.asr_client = AsrClient(
            settings.url_for("asr", placement["asr"]),
            settings.asr,
            side="remote",
            encoding=settings.remote.audio_encoding,
            **asr_transport,
        )
        self.tts_client = TextToSpeechClient(
            settings.url_for("tts", placement["tts"]),
            settings.tts,
            side=placement["tts"],
            **tts_transport,
        )

    async def run(
        self,
        /,
    ) -> dict[str, Any]:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        try:
            for probe in PROBES:
                results.append(await self._run_probe(probe))
        finally:
            await self.asr_client.aclose()
            await self.tts_client.aclose()
        return {
            "ok": all(result["ok"] for result in results),
            "max_character_error_rate": self.max_character_error_rate,
            "output_directory": str(self.output_directory),
            "results": results,
        }

    async def _run_probe(
        self,
        probe: SpeechProbe,
        /,
    ) -> dict[str, Any]:
        try:
            chunks = [
                chunk
                async for chunk in self.tts_client.synthesize(
                    probe.text,
                    probe.language,
                )
            ]
            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            if audio.size == 0:
                raise RuntimeError("TTS returned no audio")
            output_path = self.output_directory / f"{probe.language.lower()}-roundtrip.wav"
            await asyncio.to_thread(
                _write_wave,
                output_path,
                audio,
                self.settings.tts.sample_rate,
            )
            asr_audio = resample_once(
                audio,
                self.settings.tts.sample_rate,
                self.settings.shm.sample_rate,
            )
            transcription = await self.asr_client.transcribe(
                AudioPayload(
                    pcm=asr_audio,
                    sample_rate=self.settings.shm.sample_rate,
                ),
                language_hint=probe.language,
            )
            error_rate = character_error_rate(probe.text, transcription.best)
            return {
                "ok": bool(
                    transcription.best.strip()
                    and error_rate <= self.max_character_error_rate
                ),
                "language": probe.language,
                "reference": probe.text,
                "transcription": transcription.best,
                "character_error_rate": round(error_rate, 4),
                "audio_seconds": round(audio.size / self.settings.tts.sample_rate, 3),
                "path": str(output_path),
            }
        except Exception as error:  # noqa: BLE001 - report every language in one run
            return {
                "ok": False,
                "language": probe.language,
                "reference": probe.text,
                "error": repr(error),
            }


def _write_wave(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
    /,
) -> None:
    with wave.open(str(path), "wb") as wave_writer:
        wave_writer.setnchannels(1)
        wave_writer.setsampwidth(2)
        wave_writer.setframerate(sample_rate)
        wave_writer.writeframes(encode_pcm(audio, "s16le"))


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description="Measure deployed TTS intelligibility through ASR round-trip CER."
    )
    argument_parser.add_argument("--config", type=Path)
    argument_parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("spikes/out/speech-quality"),
    )
    argument_parser.add_argument("--max-cer", type=float, default=0.35)
    arguments = argument_parser.parse_args()
    settings = load_settings(arguments.config)
    runner = SpeechQualityRunner(
        settings,
        arguments.output_directory,
        arguments.max_cer,
    )
    result = asyncio.run(runner.run())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
