"""교차 검증 ASR 상주 서버 — faster-whisper large-v3 (CTranslate2).

CTranslate2 의 aarch64 CUDA 빌드가 실패할 수 있다. 그 경우 whisper.cpp CUDA 로
폴백한다(§7). 폴백 경로는 whisper.cpp 서버를 별도로 띄우고 그 앞에 얇게 붙인다.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..config import load_settings
from ..logging_setup import setup_logging
from ..shmring import AudioRef, StaleSlotError, attach_cached

log = setup_logging(service="asr-verify", console=True)


class VerifyReq(BaseModel):
    audio: dict[str, Any]
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
            vad_filter=False,  # 프런트엔드에서 이미 잘라 보냈다
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
    """whisper.cpp CUDA 서버 프록시 폴백.

    whisper.cpp 의 server 예제(`--port`)를 띄워두고 그 앞에 붙인다.
    환경변수 WHISPER_CPP_URL 로 주소를 준다.
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
            "avg_logprob": -1.0,  # whisper.cpp 서버는 로그확률을 주지 않는다
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


@app.get("/health")
def health() -> dict:
    b = STATE["backend"]
    return {
        "ok": b is not None,
        "service": "asr-verify",
        "backend": getattr(b, "name", None),
        "error": STATE["error"],
    }


@app.post("/transcribe")
def transcribe(req: VerifyReq) -> dict:
    b = STATE["backend"]
    if b is None:
        raise HTTPException(503, f"verify backend not loaded: {STATE['error']}")
    ref = AudioRef.from_json(req.audio)
    try:
        audio = attach_cached(ref.name).read(ref)
    except StaleSlotError as e:
        raise HTTPException(409, str(e)) from e
    return b.transcribe(audio, req)
