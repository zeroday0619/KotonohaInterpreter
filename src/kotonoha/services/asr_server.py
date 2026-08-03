"""Primary ASR server — Qwen3-ASR 1.7B, N-best 5 with LID.

The API follows the model card:
    processor.apply_transcription_request(audio=..., prompt=..., language=...)
    processor.decode(ids, return_format="parsed") -> {"language", "transcription"}

N-best comes from beam search via num_return_sequences (§5.2). This is
consecutive interpreting, so there is no reason to decode greedily.

On LID confidence: the model does not hand back a language probability. What we
use instead is the fraction of the five candidates that agree on a language.
It is a defensible proxy, and it catches precisely the case where candidates
split over the language on a short utterance — which is exactly when the
fallback is needed. Phase 1 should check how well this tracks real LID accuracy.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
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
from .config_admin import router as config_admin_router

log = setup_logging(service="asr", console=True)

# Our language codes to the names Qwen3-ASR expects.
QWEN_LANG = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh-TW": "Chinese"}


class TranscribeReq(BaseModel):
    # Present on the shared-memory path, absent on the upload path.
    audio: dict[str, Any] | None = None
    n_best: int = 5
    num_beams: int = 5
    max_new_tokens: int = 256
    context: str = ""
    language_hint: str | None = None


class TransformersBackend:
    name = "transformers"

    def __init__(self, model_id: str, dtype: str = "float16"):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        td = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id, dtype=td, device_map="auto"
        )
        self.model.eval()
        self.load_s = round(time.perf_counter() - t0, 2)
        log.info("asr.loaded", model=model_id, dtype=dtype, load_s=self.load_s)

    def _build_inputs(self, audio: np.ndarray, prompt: str, language: str | None):
        kwargs: dict[str, Any] = {"audio": audio}
        if prompt:
            kwargs["prompt"] = prompt
        if language:
            kwargs["language"] = language
        # Some versions accept sampling_rate; others assume 16k.
        try:
            return self.processor.apply_transcription_request(sampling_rate=16000, **kwargs)
        except TypeError:
            return self.processor.apply_transcription_request(**kwargs)

    def transcribe(self, audio: np.ndarray, req: TranscribeReq) -> dict[str, Any]:
        torch = self.torch
        lang = QWEN_LANG.get(req.language_hint or "", None)
        inputs = self._build_inputs(audio, req.context, lang)
        inputs = inputs.to(self.model.device, self.model.dtype)

        n = max(1, req.n_best)
        beams = max(n, req.num_beams)

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                num_beams=beams,
                num_return_sequences=n,
                length_penalty=1.0,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        infer_ms = (time.perf_counter() - t0) * 1000

        prompt_len = inputs["input_ids"].shape[1]
        seqs = out.sequences[:, prompt_len:]
        parsed = self.processor.decode(seqs, return_format="parsed")
        if isinstance(parsed, dict):
            parsed = [parsed]

        # Beam search sequences_scores is a length-normalised log-probability,
        # i.e. the average log-probability.
        if getattr(out, "sequences_scores", None) is not None:
            scores = [float(s) for s in out.sequences_scores.detach().cpu().tolist()]
        else:
            scores = [-99.0] * len(parsed)

        hyps = []
        langs = []
        for i, p in enumerate(parsed):
            if isinstance(p, dict):
                text = (p.get("transcription") or "").strip()
                langs.append(p.get("language"))
            else:
                text = str(p).strip()
                langs.append(None)
            hyps.append({"text": text, "avg_logprob": scores[i] if i < len(scores) else -99.0})

        language, confidence = _vote_language(langs)
        return {
            "hypotheses": hyps,
            "language": language,
            "language_confidence": confidence,
            "duration_s": round(len(audio) / 16000.0, 3),
            "infer_ms": round(infer_ms, 1),
        }


class VllmBackend:
    """[Blocked on Spike 1] the vLLM path.

    Whether the Jetson vLLM container loads Qwen3-ASR at all, and whether it can
    produce N-best, has to be established on the device
    (spikes/spike1_asr_load.py). Guessing at an implementation before that would
    invalidate the entire latency budget, so this fails loudly instead.
    """

    name = "vllm"

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "The vLLM ASR backend is implemented once Spike 1 settles it. "
            "Run spikes/spike1_asr_load.py on the Jetson, confirm how N-best and "
            "log-probabilities are obtained, then fill in this class. "
            "Until then use asr.backend: transformers in the config."
        )


def _vote_language(langs: list[str | None]) -> tuple[str | None, float | None]:
    """Use the candidates' agreement rate on a language as the confidence."""
    vals = [x for x in langs if x]
    if not vals:
        return None, None
    top, count = Counter(vals).most_common(1)[0]
    return top, round(count / len(vals), 3)


STATE: dict[str, Any] = {"backend": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = load_settings(os.environ.get("KOTONOHA_CONFIG"))
    try:
        if s.asr.backend == "vllm":
            STATE["backend"] = VllmBackend(s.asr.vllm_model_id)
        else:
            STATE["backend"] = TransformersBackend(s.asr.model_id, s.asr.dtype)
    except Exception as e:  # noqa: BLE001
        STATE["error"] = repr(e)
        log.error("asr.load_failed", error=repr(e))
    yield


app = FastAPI(title="kotonoha-asr", lifespan=lifespan)
install_auth(app, "asr")
app.include_router(config_admin_router)


@app.get("/health")
def health() -> dict:
    b = STATE["backend"]
    return {
        "ok": b is not None,
        "service": "asr",
        "backend": getattr(b, "name", None),
        "error": STATE["error"],
    }


def _backend():
    b = STATE["backend"]
    if b is None:
        raise HTTPException(503, f"asr backend not loaded: {STATE['error']}")
    return b


@app.post("/transcribe")
def transcribe(req: TranscribeReq) -> dict:
    """Shared-memory path, used when the orchestrator is on the same box."""
    b = _backend()
    if req.audio is None:
        raise HTTPException(400, "missing audio reference; use /transcribe/upload instead")
    ref = AudioRef.from_json(req.audio)
    try:
        audio = attach_cached(ref.name).read(ref)
    except StaleSlotError as e:
        raise HTTPException(409, str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(503, f"shm not available: {e}") from e
    return b.transcribe(audio, req)


@app.post("/transcribe/upload")
async def transcribe_upload(params: str = Form("{}"), audio: UploadFile = File(...)) -> dict:
    """Upload path, for an orchestrator running on another machine.

    `params` carries the JSON that would otherwise be the request body; `audio`
    is raw PCM in the encoding named there. No base64 anywhere (§3).
    """
    b = _backend()
    try:
        d = json.loads(params or "{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad params json: {e}") from e

    encoding = d.pop("encoding", "s16le")
    sample_rate = int(d.pop("sample_rate", 16000))
    if sample_rate != 16000:
        raise HTTPException(400, f"expected 16 kHz audio, got {sample_rate}")

    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty audio")
    pcm = decode_pcm(raw, encoding)

    known = set(TranscribeReq.model_fields)
    req = TranscribeReq(**{k: v for k, v in d.items() if k in known and k != "audio"})
    out = b.transcribe(pcm, req)
    out["received_bytes"] = len(raw)
    return out


@app.post("/echo")
async def echo(audio: UploadFile = File(...)) -> dict:
    """Transport probe for `kotonoha netcheck`.

    Reads the body and reports its size, deliberately running no inference, so
    the number measures the link and nothing else.
    """
    t0 = time.perf_counter()
    raw = await audio.read()
    return {"bytes": len(raw), "read_ms": round((time.perf_counter() - t0) * 1000, 3)}
