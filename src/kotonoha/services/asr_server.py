"""1차 ASR 상주 서버 — Qwen3-ASR 1.7B, N-best 5 + LID.

API 는 모델 카드 기준이다:
    processor.apply_transcription_request(audio=..., prompt=..., language=...)
    processor.decode(ids, return_format="parsed") -> {"language", "transcription"}

N-best 는 beam search 의 num_return_sequences 로 얻는다(§5.2). 순차식이라
그리디를 쓸 이유가 없다.

LID 신뢰도에 대해: 모델이 언어 확률을 직접 주지 않는다. 여기서는 5개 후보가
같은 언어에 동의한 비율을 신뢰도로 쓴다. 근거 있는 대리 지표이고, 짧은 발화에서
후보들이 언어를 두고 갈리는 상황을 정확히 잡아낸다 — 그게 폴백이 필요한 바로
그 경우다. Phase 1 에서 실제 LID 정확도와 이 지표의 상관을 확인할 것.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..config import load_settings
from ..logging_setup import setup_logging
from ..shmring import AudioRef, StaleSlotError, attach_cached

log = setup_logging(service="asr", console=True)

# 내부 코드 → Qwen3-ASR 언어명
QWEN_LANG = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh-TW": "Chinese"}


class TranscribeReq(BaseModel):
    audio: dict[str, Any]
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
        # 일부 버전은 sampling_rate 를 받고, 일부는 16k 를 가정한다.
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

        # beam search 의 sequences_scores 는 길이 정규화된 로그확률 = 평균 로그확률
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
    """[SPIKE-1 대기] vLLM 경로.

    Jetson 용 vLLM 컨테이너가 Qwen3-ASR 을 로드하는지, N-best 를 낼 수 있는지는
    실기에서 확인해야 한다(spikes/spike1_asr_load.py). 확인 전에 추측으로 구현하면
    지연 예산 계산이 통째로 틀어지므로, 여기서는 명시적으로 실패시킨다.
    """

    name = "vllm"

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "vLLM ASR 백엔드는 Spike 1 결과 확정 후 구현한다. "
            "spikes/spike1_asr_load.py 를 Jetson 에서 실행하고, "
            "N-best/로그확률 획득 방법을 확인한 뒤 이 클래스를 채울 것. "
            "그 전까지는 config asr.backend: transformers 를 쓴다."
        )


def _vote_language(langs: list[str | None]) -> tuple[str | None, float | None]:
    """후보들의 언어 합의율을 신뢰도로 쓴다."""
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


@app.get("/health")
def health() -> dict:
    b = STATE["backend"]
    return {
        "ok": b is not None,
        "service": "asr",
        "backend": getattr(b, "name", None),
        "error": STATE["error"],
    }


@app.post("/transcribe")
def transcribe(req: TranscribeReq) -> dict:
    b = STATE["backend"]
    if b is None:
        raise HTTPException(503, f"asr backend not loaded: {STATE['error']}")
    ref = AudioRef.from_json(req.audio)
    try:
        audio = attach_cached(ref.name).read(ref)
    except StaleSlotError as e:
        raise HTTPException(409, str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(503, f"shm not available: {e}") from e
    return b.transcribe(audio, req)
