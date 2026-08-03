"""TTS 상주 서버 — Qwen3-TTS 0.6B, MeloTTS 폴백 (§7, §10).

Qwen3-TTS 모델 카드의 예시는 `attn_implementation="flash_attention_2"` 를 쓴다.
flash-attn 이 sm_87 에서 빌드되는지가 Spike 2 의 질문이다. 여기서는
flash_attention_2 → sdpa → eager 순으로 시도하고 무엇으로 떴는지 /health 에
노출한다. 실기에서 어느 경로로 떴는지 추측하지 않아도 되게 하기 위함이다.

스트리밍: 모델 카드에 스트리밍 합성 API 가 공개되어 있지 않다. 대신 절 단위로
짧게 합성하고(§5.4 절 핸드오프 덕에 절은 원래 짧다) 결과를 chunk_ms 조각으로
잘라 흘려보낸다. 첫 패킷 지연은 '절 하나의 합성 시간'이 되며, 그 값은 Phase 2
에서 실측해 §6 의 0.3초 예산과 대조한다.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..config import load_settings
from ..logging_setup import setup_logging

log = setup_logging(service="tts", console=True)

QWEN_LANG = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh-TW": "Chinese"}
MELO_LANG = {"ko": "KR", "en": "EN", "ja": "JP", "zh-TW": "ZH"}


class SynthReq(BaseModel):
    text: str
    lang: str
    voice: str | None = None
    speaker: str | None = None
    sample_rate: int = 24000


class Qwen3TtsBackend:
    name = "qwen3"

    def __init__(self, model_id: str):
        import torch
        from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]

        last: Exception | None = None
        for attn in ("flash_attention_2", "sdpa", "eager"):
            try:
                t0 = time.perf_counter()
                self.model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map="cuda:0",
                    dtype=torch.bfloat16,
                    attn_implementation=attn,
                )
                self.attn = attn
                log.info(
                    "tts.loaded",
                    backend=self.name,
                    attn=attn,
                    load_s=round(time.perf_counter() - t0, 2),
                )
                return
            except Exception as e:  # noqa: BLE001
                last = e
                log.warning("tts.attn_failed", attn=attn, error=repr(e))
        raise RuntimeError(f"Qwen3-TTS load failed for all attn impls: {last!r}")

    def synth(self, req: SynthReq) -> tuple[np.ndarray, int]:
        wavs, sr = self.model.generate_custom_voice(
            text=req.text,
            language=QWEN_LANG.get(req.lang, "English"),
            speaker=req.voice or "Vivian",
        )
        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        return wav, int(sr)


class MeloBackend:
    name = "melo"

    def __init__(self, device: str = "cuda:0"):
        from melo.api import TTS  # type: ignore[import-not-found]

        self._TTS = TTS
        self._device = device
        self._models: dict[str, Any] = {}
        log.info("tts.loaded", backend=self.name, device=device)

    def _model(self, lang: str):
        code = MELO_LANG.get(lang, "EN")
        if code not in self._models:
            t0 = time.perf_counter()
            self._models[code] = self._TTS(language=code, device=self._device)
            log.info("melo.lang_loaded", lang=code, load_s=round(time.perf_counter() - t0, 2))
        return self._models[code]

    def synth(self, req: SynthReq) -> tuple[np.ndarray, int]:
        m = self._model(req.lang)
        spk_map = m.hps.data.spk2id
        key = req.speaker if req.speaker in spk_map else next(iter(spk_map))
        wav = m.tts_to_file(req.text, spk_map[key], None, speed=1.0)
        return np.asarray(wav, dtype=np.float32).reshape(-1), int(m.hps.data.sampling_rate)


STATE: dict[str, Any] = {"primary": None, "fallback": None, "error": None, "chunk_ms": 200}


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = load_settings(os.environ.get("KOTONOHA_CONFIG"))
    STATE["chunk_ms"] = s.tts.chunk_ms

    if s.tts.backend == "qwen3":
        try:
            STATE["primary"] = Qwen3TtsBackend(s.tts.model_id)
        except Exception as e:  # noqa: BLE001
            STATE["error"] = repr(e)
            log.error("tts.qwen3_failed", error=repr(e))

    # §10 TTS 실패 → MeloTTS 폴백. 실패한 뒤에 로드하면 첫 폴백 턴이 통째로 날아가므로
    # 미리 올려둔다.
    if s.tts.backend == "melo" or s.tts.fallback == "melo":
        try:
            STATE["fallback"] = MeloBackend()
        except Exception as e:  # noqa: BLE001
            log.error("tts.melo_failed", error=repr(e))
            STATE["error"] = (STATE["error"] or "") + f" | melo: {e!r}"

    if STATE["primary"] is None and STATE["fallback"] is not None:
        STATE["primary"], STATE["fallback"] = STATE["fallback"], None
    yield


app = FastAPI(title="kotonoha-tts", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    p, f = STATE["primary"], STATE["fallback"]
    return {
        "ok": p is not None,
        "service": "tts",
        "backend": getattr(p, "name", None),
        "attn": getattr(p, "attn", None),
        "fallback": getattr(f, "name", None),
        "error": STATE["error"],
    }


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    import soxr

    return np.asarray(soxr.resample(x, src, dst), dtype=np.float32)


@app.post("/synthesize")
async def synthesize(req: SynthReq) -> StreamingResponse:
    p, f = STATE["primary"], STATE["fallback"]
    if p is None:
        raise HTTPException(503, f"tts not loaded: {STATE['error']}")
    if not req.text.strip():
        raise HTTPException(400, "empty text")

    t0 = time.perf_counter()
    try:
        wav, sr = await asyncio.to_thread(p.synth, req)
        used = p.name
    except Exception as e:  # noqa: BLE001
        log.warning("tts.primary_failed", error=repr(e), text=req.text[:40])
        if f is None:
            raise HTTPException(500, f"tts failed: {e!r}") from e
        wav, sr = await asyncio.to_thread(f.synth, req)
        used = f.name

    wav = _resample(wav, sr, req.sample_rate)
    synth_ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info(
        "tts.synth",
        backend=used,
        lang=req.lang,
        chars=len(req.text),
        audio_s=round(wav.size / req.sample_rate, 2),
        synth_ms=synth_ms,
    )

    chunk = max(1, int(req.sample_rate * STATE["chunk_ms"] / 1000))

    def gen():
        for i in range(0, wav.size, chunk):
            yield wav[i : i + chunk].astype("<f4").tobytes()

    return StreamingResponse(
        gen(),
        media_type="application/octet-stream",
        headers={
            "X-TTS-Backend": used,
            "X-Synth-Ms": str(synth_ms),
            "X-Sample-Rate": str(req.sample_rate),
        },
    )
