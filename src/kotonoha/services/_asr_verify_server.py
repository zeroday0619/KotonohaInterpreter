"""Cross-verification ASR server using faster-whisper large-v3.

The Jetson AArch64 binary uses CPU INT8 because its CTranslate2 build lacks CUDA.
The A6000 overlay selects CUDA FP16 independently.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, ClassVar, Final

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from kotonoha._call_compatibility import keyword_compatible
from kotonoha._config import load_settings
from kotonoha._logging_setup import setup_logging
from kotonoha._prometheus import install_metrics, observe_service_health
from kotonoha._shmring import AudioRef, StaleSlotError, attach_cached
from kotonoha._transport import decode_pcm
from kotonoha._typing import override
from kotonoha.services._auth import install_auth
from kotonoha.services._resources import resource_report

log = setup_logging(service="asr-verify", console=True)


class VerificationRequest(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    audio: dict[str, Any] | None = None
    language: str | None = None
    beam_size: int = 5


class FasterWhisperBackend:
    __slots__: ClassVar[tuple[str, ...]] = (
        "compute_type",
        "device",
        "model",
    )
    name: Final = "faster_whisper"
    compute_type: str
    device: str
    model: Any

    @override
    def __init__(
        self,
        /,
        model_id: str,
        device: str,
        compute_type: str,
    ) -> None:
        from faster_whisper import WhisperModel

        start_time = time.perf_counter()
        self.compute_type = compute_type
        self.device = device
        self.model = WhisperModel(model_id, device=device, compute_type=compute_type)
        log.info(
            "verify.loaded",
            model=model_id,
            device=device,
            compute_type=compute_type,
            load_s=round(time.perf_counter() - start_time, 2),
        )

    def transcribe(
        self,
        /,
        audio: np.ndarray,
        request: VerificationRequest,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language=request.language,
            beam_size=request.beam_size,
            vad_filter=False,  # the frontend already segmented this
            condition_on_previous_text=False,
        )
        parts, log_probabilities = [], []
        for segment in segments:
            parts.append(segment.text)
            if segment.avg_logprob is not None:
                log_probabilities.append(segment.avg_logprob)
        return {
            "text": "".join(parts).strip(),
            "avg_logprob": (
                float(np.mean(log_probabilities)) if log_probabilities else -99.0
            ),
            "language": getattr(info, "language", None),
            "infer_ms": round((time.perf_counter() - start_time) * 1000, 1),
        }


class WhisperCppBackend:
    """Proxy fallback to a whisper.cpp CUDA server.

    Run whisper.cpp's server example (`--port`) and point this at it via the
    WHISPER_CPP_URL environment variable.
    """
    __slots__: ClassVar[tuple[str, ...]] = (
        "client",
        "url",
    )

    name: Final = "whisper_cpp"
    url: str
    client: Any

    @override
    def __init__(
        self,
        /,
        url: str,
    ) -> None:
        import httpx

        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=10.0)
        log.info("verify.whisper_cpp", url=self.url)

    def transcribe(
        self,
        /,
        audio: np.ndarray,
        request: VerificationRequest,
    ) -> dict[str, Any]:
        import io
        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wave_file:
            wave_file.setnchannels(1)
            wave_file.setsampwidth(2)
            wave_file.setframerate(16000)
            wave_file.writeframes(
                (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
            )
        buffer.seek(0)
        start_time = time.perf_counter()
        response = self.client.post(
            f"{self.url}/inference",
            files={"file": ("audio.wav", buffer, "audio/wav")},
            data={
                "language": request.language or "auto",
                "response_format": "json",
            },
        )
        response.raise_for_status()
        result = response.json()
        return {
            "text": (result.get("text") or "").strip(),
            "avg_logprob": -1.0,  # the whisper.cpp server does not expose log-probs
            "language": result.get("language"),
            "infer_ms": round((time.perf_counter() - start_time) * 1000, 1),
        }


STATE: dict[str, Any] = {"backend": None, "error": None}


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    /,
) -> Any:
    settings = await asyncio.to_thread(
        load_settings,
        os.environ.get("KOTONOHA_CONFIG"),
    )
    config = settings.asr_verify
    try:
        if config.backend == "whisper_cpp":
            STATE["backend"] = await asyncio.to_thread(
                WhisperCppBackend,
                os.environ.get("WHISPER_CPP_URL", "http://127.0.0.1:8082"),
            )
        else:
            STATE["backend"] = await asyncio.to_thread(
                FasterWhisperBackend,
                config.model_id,
                config.device,
                config.compute_type,
            )
    except Exception as error:  # noqa: BLE001
        STATE["error"] = repr(error)
        log.error("verify.load_failed", error=repr(error))
    yield


app = FastAPI(title="kotonoha-asr-verify", lifespan=lifespan)
install_auth(app, "asr-verify")
install_metrics(app, "asr-verify")


@app.get("/health")
@keyword_compatible
def health() -> dict:
    backend = STATE["backend"]
    result = {
        "ok": backend is not None,
        "service": "asr-verify",
        "backend": getattr(backend, "name", None),
        "device": getattr(backend, "device", None),
        "compute_type": getattr(backend, "compute_type", None),
        "error": STATE["error"],
        "resources": resource_report("asr_verify"),
    }
    observe_service_health("asr-verify", bool(result["ok"]), result["resources"])
    return result


def _backend() -> Any:
    backend = STATE["backend"]
    if backend is None:
        raise HTTPException(503, f"verify backend not loaded: {STATE['error']}")
    return backend


@app.post("/transcribe")
@keyword_compatible
def transcribe(
    request: VerificationRequest,
    /,
) -> dict:
    backend = _backend()
    if request.audio is None:
        raise HTTPException(400, "missing audio reference; use /transcribe/upload instead")
    audio_reference = AudioRef.from_json(request.audio)
    try:
        audio = attach_cached(audio_reference.name).read(audio_reference)
    except StaleSlotError as error:
        raise HTTPException(409, str(error)) from error
    return backend.transcribe(audio, request)


@app.post("/transcribe/upload")
@keyword_compatible
async def transcribe_upload(
    params: str = Form("{}"),
    /,
    audio: UploadFile = File(...),
) -> dict:
    """Upload path, for an orchestrator running on another machine."""
    backend = _backend()
    try:
        data = json.loads(params or "{}")
    except json.JSONDecodeError as error:
        raise HTTPException(400, f"bad params json: {error}") from error

    encoding = data.pop("encoding", "s16le")
    data.pop("sample_rate", None)
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty audio")
    pcm = decode_pcm(raw, encoding)

    known_fields = set(VerificationRequest.model_fields)
    request = VerificationRequest(
        **{
            key: value
            for key, value in data.items()
            if key in known_fields and key != "audio"
        }
    )
    return await asyncio.to_thread(backend.transcribe, pcm, request)


@app.post("/echo")
@keyword_compatible
async def echo(
    audio: UploadFile = File(...),
    /,
) -> dict:
    """Transport probe for `kotonoha netcheck`."""
    raw = await audio.read()
    return {"bytes": len(raw)}
