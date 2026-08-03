"""Cross-verification ASR server — faster-whisper large-v3 (CTranslate2).

The aarch64 CUDA build of CTranslate2 may not work. If it does not, we fall
back to whisper.cpp with CUDA (§7): run the whisper.cpp server separately and
put this thin proxy in front of it.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..config import load_settings
from ..logging_setup import setup_logging
from ..shmring import AudioRef, StaleSlotError, attach_cached
from ..transport import decode_pcm
from .auth import install_auth

log = setup_logging(service="asr-verify", console=True)


class VerifyReq(BaseModel):
    audio: dict[str, Any] | None = None
    language: str | None = None
    beam_size: int = 5


class FasterWhisperBackend:
    name = "faster_whisper"

    def __init__(self, model_id: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        self.model = WhisperModel(model_id, device=device, compute_type=compute_type)
        log.info(
            "verify.loaded",
            model=model_id,
            device=device,
            compute_type=compute_type,
            load_s=round(time.perf_counter() - t0, 2),
        )

    def transcribe(self, audio: np.ndarray, req: VerifyReq) -> dict[str, Any]:
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language=req.language,
            beam_size=req.beam_size,
            vad_filter=False,  # the frontend already segmented this
            condition_on_previous_text=False,
        )
        parts, lps = [], []
        for seg in segments:
            parts.append(seg.text)
            if seg.avg_logprob is not None:
                lps.append(seg.avg_logprob)
        return {
            "text": "".join(parts).strip(),
            "avg_logprob": float(np.mean(lps)) if lps else -99.0,
            "language": getattr(info, "language", None),
            "infer_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


class WhisperCppBackend:
    """Proxy fallback to a whisper.cpp CUDA server.

    Run whisper.cpp's server example (`--port`) and point this at it via the
    WHISPER_CPP_URL environment variable.
    """

    name = "whisper_cpp"

    def __init__(self, url: str):
        import httpx

        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=10.0)
        log.info("verify.whisper_cpp", url=self.url)

    def transcribe(self, audio: np.ndarray, req: VerifyReq) -> dict[str, Any]:
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        buf.seek(0)
        t0 = time.perf_counter()
        r = self.client.post(
            f"{self.url}/inference",
            files={"file": ("a.wav", buf, "audio/wav")},
            data={"language": req.language or "auto", "response_format": "json"},
        )
        r.raise_for_status()
        d = r.json()
        return {
            "text": (d.get("text") or "").strip(),
            "avg_logprob": -1.0,  # the whisper.cpp server does not expose log-probs
            "language": d.get("language"),
            "infer_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


STATE: dict[str, Any] = {"backend": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = load_settings(os.environ.get("KOTONOHA_CONFIG"))
    c = s.asr_verify
    try:
        if c.backend == "whisper_cpp":
            STATE["backend"] = WhisperCppBackend(
                os.environ.get("WHISPER_CPP_URL", "http://127.0.0.1:8082")
            )
        else:
            STATE["backend"] = FasterWhisperBackend(c.model_id, c.device, c.compute_type)
    except Exception as e:  # noqa: BLE001
        STATE["error"] = repr(e)
        log.error("verify.load_failed", error=repr(e))
    yield


app = FastAPI(title="kotonoha-asr-verify", lifespan=lifespan)
install_auth(app, "asr-verify")


@app.get("/health")
def health() -> dict:
    b = STATE["backend"]
    return {
        "ok": b is not None,
        "service": "asr-verify",
        "backend": getattr(b, "name", None),
        "error": STATE["error"],
    }


def _backend():
    b = STATE["backend"]
    if b is None:
        raise HTTPException(503, f"verify backend not loaded: {STATE['error']}")
    return b


@app.post("/transcribe")
def transcribe(req: VerifyReq) -> dict:
    b = _backend()
    if req.audio is None:
        raise HTTPException(400, "missing audio reference; use /transcribe/upload instead")
    ref = AudioRef.from_json(req.audio)
    try:
        audio = attach_cached(ref.name).read(ref)
    except StaleSlotError as e:
        raise HTTPException(409, str(e)) from e
    return b.transcribe(audio, req)


@app.post("/transcribe/upload")
async def transcribe_upload(params: str = Form("{}"), audio: UploadFile = File(...)) -> dict:
    """Upload path, for an orchestrator running on another machine."""
    b = _backend()
    try:
        d = json.loads(params or "{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad params json: {e}") from e

    encoding = d.pop("encoding", "s16le")
    d.pop("sample_rate", None)
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty audio")
    pcm = decode_pcm(raw, encoding)

    known = set(VerifyReq.model_fields)
    req = VerifyReq(**{k: v for k, v in d.items() if k in known and k != "audio"})
    return b.transcribe(pcm, req)


@app.post("/echo")
async def echo(audio: UploadFile = File(...)) -> dict:
    """Transport probe for `kotonoha netcheck`."""
    raw = await audio.read()
    return {"bytes": len(raw)}
