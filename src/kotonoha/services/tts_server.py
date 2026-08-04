"""TTS server — Qwen3-TTS 0.6B with a MeloTTS fallback (§7, §10).

The Qwen3-TTS model card uses `attn_implementation="flash_attention_2"`.
Spike 2 determines whether flash-attn builds for sm_87. The server attempts
flash_attention_2, then sdpa, then eager, and exposes the selected backend on
the `/health` endpoint.

The model card documents no streaming synthesis API. The server synthesizes one
clause at a time and emits the result in `chunk_ms` slices. First-packet latency
therefore includes complete synthesis of the first clause. Phase 2 measures this
latency against the 0.3-second budget in §6.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from kotonoha.config import load_settings
from kotonoha.logging_setup import setup_logging
from kotonoha.services.auth import install_auth
from kotonoha.transport import encode_pcm

log = setup_logging(service="tts", console=True)

QWEN_LANG = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh-TW": "Chinese"}
MELO_LANG = {"ko": "KR", "en": "EN", "ja": "JP", "zh-TW": "ZH"}


class SynthesisRequest(BaseModel):
    text: str
    lang: str
    voice: str | None = None
    speaker: str | None = None
    sample_rate: int = 24000
    # f32le for a local hop; s16le halves the bytes when the client is remote.
    encoding: Literal["s16le", "f32le"] = "f32le"


class Qwen3TtsBackend:
    name = "qwen3"
    model: Any
    attention_implementation: str

    def __init__(self, model_id: str):
        import torch
        from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]

        last: Exception | None = None
        attention_implementations = ["sdpa", "eager"]
        if importlib.util.find_spec("flash_attn") is not None:
            attention_implementations.insert(0, "flash_attention_2")
        else:
            log.warning("tts.flash_attn_unavailable")

        for attention_implementation in attention_implementations:
            try:
                start_time = time.perf_counter()
                self.model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map="cuda:0",
                    dtype=torch.bfloat16,
                    attn_implementation=attention_implementation,
                )
                self.attention_implementation = attention_implementation
                log.info(
                    "tts.loaded",
                    backend=self.name,
                    attn=attention_implementation,
                    load_s=round(time.perf_counter() - start_time, 2),
                )
                return
            except Exception as error:  # noqa: BLE001
                last = error
                log.warning(
                    "tts.attn_failed",
                    attn=attention_implementation,
                    error=repr(error),
                )
        raise RuntimeError(f"Qwen3-TTS load failed for all attn impls: {last!r}")

    def synth(self, request: SynthesisRequest) -> tuple[np.ndarray, int]:
        waveforms, sample_rate = self.model.generate_custom_voice(
            text=request.text,
            language=QWEN_LANG.get(request.lang, "English"),
            speaker=request.voice or "Vivian",
        )
        waveform = np.asarray(waveforms[0], dtype=np.float32).reshape(-1)
        return waveform, int(sample_rate)


class MeloBackend:
    name = "melo"
    _text_to_speech_class: Any
    _device: str
    _models: dict[str, Any]

    def __init__(self, device: str = "cuda:0"):
        from melo.api import TTS  # type: ignore[import-not-found]

        self._text_to_speech_class = TTS
        self._device = device
        self._models: dict[str, Any] = {}
        log.info("tts.loaded", backend=self.name, device=device)

    def _model(self, lang: str):
        code = MELO_LANG.get(lang, "EN")
        if code not in self._models:
            start_time = time.perf_counter()
            self._models[code] = self._text_to_speech_class(
                language=code,
                device=self._device,
            )
            log.info(
                "melo.lang_loaded",
                lang=code,
                load_s=round(time.perf_counter() - start_time, 2),
            )
        return self._models[code]

    def synth(self, request: SynthesisRequest) -> tuple[np.ndarray, int]:
        model = self._model(request.lang)
        speaker_map = model.hps.data.spk2id
        speaker = (
            request.speaker
            if request.speaker in speaker_map
            else next(iter(speaker_map))
        )
        waveform = model.tts_to_file(
            request.text,
            speaker_map[speaker],
            None,
            speed=1.0,
        )
        return (
            np.asarray(waveform, dtype=np.float32).reshape(-1),
            int(model.hps.data.sampling_rate),
        )


STATE: dict[str, Any] = {"primary": None, "fallback": None, "error": None, "chunk_ms": 200}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = await asyncio.to_thread(
        load_settings,
        os.environ.get("KOTONOHA_CONFIG"),
    )
    STATE["chunk_ms"] = settings.tts.chunk_ms

    if settings.tts.backend == "qwen3":
        try:
            STATE["primary"] = await asyncio.to_thread(
                Qwen3TtsBackend,
                settings.tts.model_id,
            )
        except Exception as error:  # noqa: BLE001
            STATE["error"] = repr(error)
            log.error("tts.qwen3_failed", error=repr(error))

    # §10 TTS failure -> MeloTTS fallback. Load it up front: loading only after
    # the first failure would lose that entire turn.
    if settings.tts.backend == "melo" or settings.tts.fallback == "melo":
        try:
            STATE["fallback"] = await asyncio.to_thread(MeloBackend)
        except Exception as error:  # noqa: BLE001
            log.error("tts.melo_failed", error=repr(error))
            STATE["error"] = (STATE["error"] or "") + f" | melo: {error!r}"

    if STATE["primary"] is None and STATE["fallback"] is not None:
        STATE["primary"], STATE["fallback"] = STATE["fallback"], None
    yield


app = FastAPI(title="kotonoha-tts", lifespan=lifespan)
install_auth(app, "tts")


@app.get("/health")
def health() -> dict:
    primary, fallback = STATE["primary"], STATE["fallback"]
    return {
        "ok": primary is not None,
        "service": "tts",
        "backend": getattr(primary, "name", None),
        "attn": getattr(primary, "attention_implementation", None),
        "fallback": getattr(fallback, "name", None),
        "error": STATE["error"],
    }


def _resample(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    import soxr

    return np.asarray(
        soxr.resample(samples, source_rate, target_rate),
        dtype=np.float32,
    )


@app.post("/synthesize")
async def synthesize(request: SynthesisRequest) -> StreamingResponse:
    primary, fallback = STATE["primary"], STATE["fallback"]
    if primary is None:
        raise HTTPException(503, f"tts not loaded: {STATE['error']}")
    if not request.text.strip():
        raise HTTPException(400, "empty text")

    start_time = time.perf_counter()
    try:
        waveform, sample_rate = await asyncio.to_thread(primary.synth, request)
        used_backend = primary.name
    except Exception as error:  # noqa: BLE001
        log.warning(
            "tts.primary_failed",
            error=repr(error),
            text=request.text[:40],
        )
        if fallback is None:
            raise HTTPException(500, f"tts failed: {error!r}") from error
        waveform, sample_rate = await asyncio.to_thread(fallback.synth, request)
        used_backend = fallback.name

    waveform = _resample(waveform, sample_rate, request.sample_rate)
    synthesis_ms = round((time.perf_counter() - start_time) * 1000, 1)
    log.info(
        "tts.synth",
        backend=used_backend,
        lang=request.lang,
        chars=len(request.text),
        audio_s=round(waveform.size / request.sample_rate, 2),
        synth_ms=synthesis_ms,
    )

    chunk_size = max(
        1,
        int(request.sample_rate * STATE["chunk_ms"] / 1000),
    )

    def generate_chunks():
        for start in range(0, waveform.size, chunk_size):
            yield encode_pcm(
                waveform[start : start + chunk_size],
                request.encoding,
            )

    return StreamingResponse(
        generate_chunks(),
        media_type="application/octet-stream",
        headers={
            "X-TTS-Backend": used_backend,
            "X-Synth-Ms": str(synthesis_ms),
            "X-Sample-Rate": str(request.sample_rate),
            # The client trusts this over what it asked for, so an older or
            # misconfigured service cannot silently corrupt the stream.
            "X-Encoding": request.encoding,
        },
    )
