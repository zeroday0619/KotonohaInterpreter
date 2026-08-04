"""Primary ASR server — Qwen3-ASR 1.7B, N-best 5 with LID.

The API follows the model card:
    processor.apply_transcription_request(audio=..., prompt=..., language=...)
    processor.decode(ids, return_format="parsed") -> {"language", "transcription"}

N-best comes from beam search via num_return_sequences (§5.2). This is
consecutive interpreting, so there is no reason to decode greedily.

The model does not return a language probability. The implementation uses the
fraction of the five candidates that agree on a language. Candidate disagreement
activates the configured low-confidence fallback. Phase 1 must measure how this
proxy correlates with language-identification accuracy.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import Counter
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any, ClassVar, Final

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from kotonoha._call_compatibility import keyword_compatible
from kotonoha._config import load_settings
from kotonoha._logging_setup import setup_logging
from kotonoha._shmring import AudioRef, StaleSlotError, attach_cached
from kotonoha._transport import decode_pcm
from kotonoha._typing import override
from kotonoha.services._auth import install_auth
from kotonoha.services._config_admin import router as config_admin_router

log = setup_logging(service="asr", console=True)

# Map application language codes to the names expected by Qwen3-ASR.
QWEN_LANG = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh-TW": "Chinese"}
QWEN_AUDIO_PLACEHOLDER: Final = "<|audio_start|><|audio_pad|><|audio_end|>"
QWEN_ASR_TEXT_TAG: Final = "<asr_text>"
MINIMUM_VLLM_VERSION: Final = (0, 19, 0)
_CHATML_TOKEN = re.compile(r"<\|[^<>|]*\|>")


class TranscribeRequest(BaseModel):
    # Present on the shared-memory path, absent on the upload path.
    __slots__: ClassVar[tuple[str, ...]] = ()
    audio: dict[str, Any] | None = None
    n_best: int = 5
    num_beams: int = 5
    max_new_tokens: int = 256
    context: str = ""
    language_hint: str | None = None


class TransformersBackend:
    __slots__: ClassVar[tuple[str, ...]] = (
        "load_seconds",
        "model",
        "processor",
        "torch",
    )
    name: Final = "transformers"
    torch: Any
    processor: Any
    model: Any
    load_seconds: float

    @override
    def __init__(
        self,
        /,
        model_id: str,
        dtype: str = "float16",
    ) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        start_time = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map="auto",
        )
        self.model.eval()
        self.load_seconds = round(time.perf_counter() - start_time, 2)
        log.info(
            "asr.loaded",
            model=model_id,
            dtype=dtype,
            load_s=self.load_seconds,
        )

    def _build_inputs(
        self,
        /,
        audio: np.ndarray,
        prompt: str,
        language: str | None,
    ) -> Any:
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

    def transcribe(
        self,
        /,
        audio: np.ndarray,
        request: TranscribeRequest,
    ) -> dict[str, Any]:
        torch = self.torch
        language = QWEN_LANG.get(request.language_hint or "", None)
        inputs = self._build_inputs(audio, request.context, language)
        inputs = inputs.to(self.model.device, self.model.dtype)

        candidate_count = max(1, request.n_best)
        beams = max(candidate_count, request.num_beams)

        start_time = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                num_beams=beams,
                num_return_sequences=candidate_count,
                length_penalty=1.0,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        inference_ms = (time.perf_counter() - start_time) * 1000

        prompt_length = inputs["input_ids"].shape[1]
        sequences = output.sequences[:, prompt_length:]
        parsed = self.processor.decode(sequences, return_format="parsed")
        if isinstance(parsed, dict):
            parsed = [parsed]

        # Beam search sequences_scores is a length-normalised log-probability,
        # i.e. the average log-probability.
        if getattr(output, "sequences_scores", None) is not None:
            scores = [
                float(score)
                for score in output.sequences_scores.detach().cpu().tolist()
            ]
        else:
            scores = [-99.0] * len(parsed)

        hypotheses = []
        languages = []
        for index, candidate in enumerate(parsed):
            if isinstance(candidate, dict):
                text = (candidate.get("transcription") or "").strip()
                languages.append(candidate.get("language"))
            else:
                text = str(candidate).strip()
                languages.append(None)
            hypotheses.append(
                {
                    "text": text,
                    "avg_logprob": (
                        scores[index] if index < len(scores) else -99.0
                    ),
                }
            )

        language, confidence = _vote_language(languages)
        return {
            "hypotheses": hypotheses,
            "language": language,
            "language_confidence": confidence,
            "duration_s": round(len(audio) / 16000.0, 3),
            "infer_ms": round(inference_ms, 1),
        }


class VllmBackend:
    """Qwen3-ASR inference through vLLM multimodal beam search."""
    __slots__: ClassVar[tuple[str, ...]] = (
        "beam_search_parameters_type",
        "llm",
        "load_seconds",
    )

    name: Final = "vllm"
    beam_search_parameters_type: Any
    llm: Any
    load_seconds: float

    @override
    def __init__(
        self,
        /,
        model_id: str,
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.80,
        max_model_len: int = 4096,
        enforce_eager: bool = True,
    ) -> None:
        try:
            runtime_version = version("vllm")
        except PackageNotFoundError as error:
            raise RuntimeError("vLLM is not installed in the ASR service image") from error
        if _numeric_version(runtime_version) < MINIMUM_VLLM_VERSION:
            raise RuntimeError(
                f"Qwen3-ASR requires vLLM >= 0.19.0; found {runtime_version}"
            )

        from vllm import LLM
        from vllm.sampling_params import BeamSearchParams

        start_time = time.perf_counter()
        self.beam_search_parameters_type = BeamSearchParams
        self.llm = LLM(
            model=model_id,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"audio": 1},
            enforce_eager=enforce_eager,
        )
        self.load_seconds = round(time.perf_counter() - start_time, 2)
        log.info(
            "asr.loaded",
            model=model_id,
            dtype=dtype,
            load_s=self.load_seconds,
            vllm_version=runtime_version,
        )

    def transcribe(
        self,
        /,
        audio: np.ndarray,
        request: TranscribeRequest,
    ) -> dict[str, Any]:
        candidate_count = max(1, request.n_best)
        beam_width = max(candidate_count, request.num_beams)
        language_hint = QWEN_LANG.get(request.language_hint or "")
        prompt = _vllm_prompt(audio, request.context, language_hint)
        parameters = self.beam_search_parameters_type(
            beam_width=beam_width,
            max_tokens=request.max_new_tokens,
            temperature=0.0,
            length_penalty=1.0,
        )

        start_time = time.perf_counter()
        outputs = self.llm.beam_search(
            [prompt],
            parameters,
            use_tqdm=False,
        )
        inference_ms = (time.perf_counter() - start_time) * 1000
        if not outputs:
            raise RuntimeError("vLLM returned no ASR output")

        hypotheses: list[dict[str, Any]] = []
        languages: list[str | None] = []
        for sequence in outputs[0].sequences[:candidate_count]:
            text, language = _parse_vllm_output(sequence.text or "", language_hint)
            generated_tokens = max(1, len(sequence.logprobs))
            hypotheses.append(
                {
                    "text": text,
                    "avg_logprob": float(sequence.cum_logprob) / generated_tokens,
                }
            )
            languages.append(language)

        if len(hypotheses) != candidate_count:
            raise RuntimeError(
                f"vLLM returned {len(hypotheses)} hypotheses; expected {candidate_count}"
            )

        language, confidence = _vote_language(languages)
        return {
            "hypotheses": hypotheses,
            "language": language,
            "language_confidence": confidence,
            "duration_s": round(len(audio) / 16000.0, 3),
            "infer_ms": round(inference_ms, 1),
        }


def _numeric_version(
    raw_version: str,
    /,
) -> tuple[int, ...]:
    components = re.match(r"^(\d+)\.(\d+)\.(\d+)", raw_version)
    if components is None:
        return (0, 0, 0)
    return tuple(int(component) for component in components.groups())


def _sanitize_transcription_context(
    context: str,
    /,
) -> str:
    previous = None
    while previous != context:
        previous = context
        context = _CHATML_TOKEN.sub("", context).replace(QWEN_ASR_TEXT_TAG, "")
    return context.strip()


def _vllm_prompt(
    audio: np.ndarray,
    /,
    context: str,
    language: str | None,
) -> dict[str, Any]:
    sanitized_context = _sanitize_transcription_context(context)
    system_turn = (
        f"<|im_start|>system\n{sanitized_context}<|im_end|>\n"
        if sanitized_context
        else ""
    )
    prompt = (
        f"{system_turn}<|im_start|>user\n{QWEN_AUDIO_PLACEHOLDER}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if language:
        prompt += f"language {language}{QWEN_ASR_TEXT_TAG}"
    return {
        "prompt": prompt,
        "multi_modal_data": {
            "audio": (np.asarray(audio, dtype=np.float32), 16000),
        },
    }


def _parse_vllm_output(
    raw_text: str,
    /,
    language_hint: str | None,
) -> tuple[str, str | None]:
    language = language_hint
    text = raw_text
    if QWEN_ASR_TEXT_TAG in raw_text:
        prefix, text = raw_text.rsplit(QWEN_ASR_TEXT_TAG, 1)
        language_match = re.search(r"language\s+([^<\n]+)$", prefix, re.IGNORECASE)
        if language_match is not None:
            language = language_match.group(1).strip()
    text = text.split("<|im_end|>", 1)[0].strip()
    return text, language


def _vote_language(
    languages: list[str | None],
    /,
) -> tuple[str | None, float | None]:
    """Use the candidates' agreement rate on a language as the confidence."""
    available = [language for language in languages if language]
    if not available:
        return None, None
    most_common, count = Counter(available).most_common(1)[0]
    return most_common, round(count / len(available), 3)


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
    try:
        if settings.asr.backend == "vllm":
            STATE["backend"] = await asyncio.to_thread(
                VllmBackend,
                settings.asr.vllm_model_id,
                settings.asr.dtype,
                settings.asr.vllm_gpu_memory_utilization,
                settings.asr.vllm_max_model_len,
                settings.asr.vllm_enforce_eager,
            )
        else:
            STATE["backend"] = await asyncio.to_thread(
                TransformersBackend,
                settings.asr.model_id,
                settings.asr.dtype,
            )
    except Exception as error:  # noqa: BLE001
        STATE["error"] = repr(error)
        log.error("asr.load_failed", error=repr(error))
    yield


app = FastAPI(title="kotonoha-asr", lifespan=lifespan)
install_auth(app, "asr")
app.include_router(config_admin_router)


@app.get("/health")
@keyword_compatible
def health() -> dict:
    backend = STATE["backend"]
    return {
        "ok": backend is not None,
        "service": "asr",
        "backend": getattr(backend, "name", None),
        "error": STATE["error"],
    }


def _backend() -> Any:
    backend = STATE["backend"]
    if backend is None:
        raise HTTPException(503, f"asr backend not loaded: {STATE['error']}")
    return backend


@app.post("/transcribe")
@keyword_compatible
def transcribe(
    request: TranscribeRequest,
    /,
) -> dict:
    """Shared-memory path, used when the orchestrator is on the same box."""
    backend = _backend()
    if request.audio is None:
        raise HTTPException(400, "missing audio reference; use /transcribe/upload instead")
    audio_reference = AudioRef.from_json(request.audio)
    try:
        audio = attach_cached(audio_reference.name).read(audio_reference)
    except StaleSlotError as error:
        raise HTTPException(409, str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(503, f"shm not available: {error}") from error
    return backend.transcribe(audio, request)


@app.post("/transcribe/upload")
@keyword_compatible
async def transcribe_upload(
    params: str = Form("{}"),
    /,
    audio: UploadFile = File(...),
) -> dict:
    """Upload path, for an orchestrator running on another machine.

    `params` carries the JSON that would otherwise be the request body; `audio`
    is raw PCM in the encoding named there. No base64 anywhere (§3).
    """
    backend = _backend()
    try:
        data = json.loads(params or "{}")
    except json.JSONDecodeError as error:
        raise HTTPException(400, f"bad params json: {error}") from error

    encoding = data.pop("encoding", "s16le")
    sample_rate = int(data.pop("sample_rate", 16000))
    if sample_rate != 16000:
        raise HTTPException(400, f"expected 16 kHz audio, got {sample_rate}")

    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty audio")
    pcm = decode_pcm(raw, encoding)

    known_fields = set(TranscribeRequest.model_fields)
    request = TranscribeRequest(
        **{
            key: value
            for key, value in data.items()
            if key in known_fields and key != "audio"
        }
    )
    result = await asyncio.to_thread(backend.transcribe, pcm, request)
    result["received_bytes"] = len(raw)
    return result


@app.post("/echo")
@keyword_compatible
async def echo(
    audio: UploadFile = File(...),
    /,
) -> dict:
    """Transport probe for `kotonoha netcheck`.

    Reads the body and reports its size, deliberately running no inference, so
    the number measures the link and nothing else.
    """
    start_time = time.perf_counter()
    raw = await audio.read()
    return {
        "bytes": len(raw),
        "read_ms": round((time.perf_counter() - start_time) * 1000, 3),
    }
