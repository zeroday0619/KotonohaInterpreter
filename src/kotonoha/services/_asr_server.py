"""Primary ASR server with in-process vLLM batch and realtime transcription.

The primary backend owns vLLM's asynchronous engine and speech-serving classes in this
FastAPI process. Batch requests reuse vLLM's transcription preprocessing and beam-search
implementation. WebSocket requests reuse its realtime connection implementation.

The explicit Transformers fallback follows the Qwen model card:
    processor.apply_transcription_request(audio=..., prompt=..., language=...)
    processor.decode(ids, return_format="parsed") -> {"language", "transcription"}

Both batch backends preserve N-best five beam search for correction evidence.

The model does not return a language probability. The implementation uses the
fraction of the five candidates that agree on a language. Candidate disagreement
activates the configured low-confidence fallback. Phase 1 must measure how this
proxy correlates with language-identification accuracy.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import os
import re
import time
import wave
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar, Final, Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from pydantic import BaseModel

from kotonoha._call_compatibility import keyword_compatible
from kotonoha._config import load_settings
from kotonoha._logging_setup import setup_logging
from kotonoha._shmring import AudioRef, StaleSlotError, attach_cached
from kotonoha._transport import decode_pcm
from kotonoha._typing import override
from kotonoha.core._lid import detect_script
from kotonoha.services._auth import install_auth, websocket_authorized
from kotonoha.services._config_admin import router as config_admin_router

log = setup_logging(service="asr", console=True)

# Map application language codes to the names expected by Qwen3-ASR.
QWEN_LANG = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh-TW": "Chinese"}
QWEN_ASR_TEXT_TAG: Final = "<asr_text>"
MINIMUM_VLLM_VERSION: Final = (0, 19, 0)
QWEN_REALTIME_ARCHITECTURE: Final = "Qwen3ASRRealtimeGeneration"
_CHATML_TOKEN = re.compile(r"<\|[^<>|]*\|>")
_QWEN_STREAM_PREFIX = re.compile(r"language\s+[^<\n]{1,50}<asr_text>", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class VllmRuntimeBindings:
    engine_arguments_type: Any
    engine_context_type: Any
    model_path_type: Any
    models_type: Any
    realtime_connection_type: Any
    realtime_serving_type: Any
    transcription_request_type: Any
    transcription_serving_type: Any


def _runtime_symbol(
    module_names: tuple[str, ...],
    symbol_name: str,
    /,
) -> Any:
    failures: list[str] = []
    for module_name in module_names:
        try:
            return getattr(import_module(module_name), symbol_name)
        except (AttributeError, ImportError) as error:
            failures.append(f"{module_name}: {error!r}")
    detail = "; ".join(failures)
    raise RuntimeError(f"vLLM runtime symbol {symbol_name} is unavailable: {detail}")


def _load_vllm_runtime_bindings() -> VllmRuntimeBindings:
    """Load both the vLLM 0.19 and current module layouts lazily."""
    return VllmRuntimeBindings(
        engine_arguments_type=_runtime_symbol(
            ("vllm.engine.arg_utils",),
            "AsyncEngineArgs",
        ),
        engine_context_type=_runtime_symbol(
            ("vllm.entrypoints.openai.api_server",),
            "build_async_engine_client_from_engine_args",
        ),
        model_path_type=_runtime_symbol(
            ("vllm.entrypoints.openai.models.protocol",),
            "BaseModelPath",
        ),
        models_type=_runtime_symbol(
            ("vllm.entrypoints.openai.models.serving",),
            "OpenAIServingModels",
        ),
        realtime_connection_type=_runtime_symbol(
            (
                "vllm.entrypoints.speech_to_text.realtime.connection",
                "vllm.entrypoints.openai.realtime.connection",
            ),
            "RealtimeConnection",
        ),
        realtime_serving_type=_runtime_symbol(
            (
                "vllm.entrypoints.speech_to_text.realtime.serving",
                "vllm.entrypoints.openai.realtime.serving",
            ),
            "OpenAIServingRealtime",
        ),
        transcription_request_type=_runtime_symbol(
            (
                "vllm.entrypoints.speech_to_text.transcription.protocol",
                "vllm.entrypoints.openai.speech_to_text.protocol",
            ),
            "TranscriptionRequest",
        ),
        transcription_serving_type=_runtime_symbol(
            (
                "vllm.entrypoints.speech_to_text.transcription.serving",
                "vllm.entrypoints.openai.speech_to_text.serving",
            ),
            "OpenAIServingTranscription",
        ),
    )


def _vllm_engine_arguments(
    model_id: str,
    served_model_name: str,
    architecture: Literal["qwen3_asr", "voxtral"],
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool,
    /,
) -> dict[str, Any]:
    return {
        "model": model_id,
        "served_model_name": [served_model_name],
        "dtype": dtype,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "trust_remote_code": True,
        "limit_mm_per_prompt": {"audio": 1},
        "hf_overrides": (
            {"architectures": [QWEN_REALTIME_ARCHITECTURE]}
            if architecture == "qwen3_asr"
            else {}
        ),
        "tokenizer_mode": "mistral" if architecture == "voxtral" else "auto",
    }


def _validate_absolute_model_directory(
    model_id: str,
    /,
) -> None:
    model_path = Path(model_id)
    if not model_path.is_absolute() or model_path.is_dir():
        return
    raise FileNotFoundError(
        f"Absolute ASR model path is missing or is not a directory: {model_path}. "
        "Update asr.vllm_model_id or mount the offline model directory at that path."
    )


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
    """Own one in-process vLLM engine for beam search and realtime ASR."""
    __slots__: ClassVar[tuple[str, ...]] = (
        "architecture",
        "bindings",
        "engine",
        "engine_context",
        "error",
        "_engine_arguments",
        "load_seconds",
        "model_id",
        "models",
        "realtime",
        "served_model_name",
        "transcription",
    )

    name: Final = "vllm_in_process"
    architecture: Literal["qwen3_asr", "voxtral"]
    bindings: VllmRuntimeBindings | None
    engine: Any
    engine_context: Any
    error: str | None
    load_seconds: float
    model_id: str
    models: Any
    realtime: Any
    served_model_name: str
    transcription: Any
    _engine_arguments: dict[str, Any]

    @override
    def __init__(
        self,
        /,
        model_id: str,
        served_model_name: str,
        architecture: Literal["qwen3_asr", "voxtral"],
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.80,
        max_model_len: int = 4096,
        enforce_eager: bool = True,
    ) -> None:
        self.architecture = architecture
        self.bindings = None
        self.engine = None
        self.engine_context = None
        self.error = None
        self.load_seconds = 0.0
        self.model_id = model_id
        self.models = None
        self.realtime = None
        self.served_model_name = served_model_name
        self.transcription = None

        try:
            runtime_version = version("vllm")
        except PackageNotFoundError as error:
            raise RuntimeError("vLLM is not installed in the ASR service image") from error
        if _numeric_version(runtime_version) < MINIMUM_VLLM_VERSION:
            raise RuntimeError(
                f"Realtime ASR requires vLLM >= 0.19.0; found {runtime_version}"
            )
        self.bindings = _load_vllm_runtime_bindings()
        self._engine_arguments = _vllm_engine_arguments(
            model_id,
            served_model_name,
            architecture,
            dtype,
            gpu_memory_utilization,
            max_model_len,
            enforce_eager,
        )

    async def start(
        self,
        /,
    ) -> None:
        if self.bindings is None:
            raise RuntimeError("vLLM runtime bindings are unavailable")

        _validate_absolute_model_directory(self.model_id)
        start_time = time.perf_counter()
        try:
            arguments = self.bindings.engine_arguments_type(**self._engine_arguments)
            engine_context = self.bindings.engine_context_type(arguments)
            self.engine = await engine_context.__aenter__()
            self.engine_context = engine_context
            model_paths = [
                self.bindings.model_path_type(
                    name=self.served_model_name,
                    model_path=self.model_id,
                )
            ]
            self.models = self.bindings.models_type(
                engine_client=self.engine,
                base_model_paths=model_paths,
                lora_modules=None,
            )
            await self.models.init_static_loras()
            self.transcription = self.bindings.transcription_serving_type(
                self.engine,
                self.models,
                request_logger=None,
            )
            self.realtime = self.bindings.realtime_serving_type(
                self.engine,
                self.models,
                request_logger=None,
            )
        except Exception:
            await self.shutdown()
            raise
        self.load_seconds = round(time.perf_counter() - start_time, 2)
        log.info(
            "asr.loaded",
            backend=self.name,
            model=self.model_id,
            served_model_name=self.served_model_name,
            realtime_architecture=self.architecture,
            load_s=self.load_seconds,
            vllm_version=version("vllm"),
        )

    async def shutdown(
        self,
        /,
    ) -> None:
        if self.transcription is not None:
            shutdown = getattr(self.transcription, "shutdown", None)
            if callable(shutdown):
                shutdown_result = shutdown()
                if inspect.isawaitable(shutdown_result):
                    await shutdown_result
            self.transcription = None
        self.realtime = None
        self.models = None
        if self.engine_context is not None:
            await self.engine_context.__aexit__(None, None, None)
            self.engine_context = None
        self.engine = None

    async def health(
        self,
        /,
    ) -> dict[str, Any]:
        ready = self.engine is not None and self.realtime is not None
        if ready:
            try:
                await self.engine.check_health()
                self.error = None
            except Exception as error:  # noqa: BLE001
                ready = False
                self.error = repr(error)
        return {
            "ok": ready,
            "service": "asr",
            "backend": self.name if ready else None,
            "model": self.model_id,
            "served_model_name": self.served_model_name,
            "realtime_architecture": self.architecture,
            "vllm": version("vllm"),
            "error": self.error,
        }

    async def transcribe(
        self,
        /,
        audio: np.ndarray,
        request: TranscribeRequest,
    ) -> dict[str, Any]:
        if self.transcription is None or self.bindings is None:
            raise RuntimeError("vLLM transcription service is not ready")
        candidate_count = max(1, request.n_best)
        beam_width = max(candidate_count, request.num_beams)
        language_hint = _iso_language(request.language_hint)
        audio_bytes = _wav_bytes(audio)
        upload = UploadFile(filename="utterance.wav", file=io.BytesIO(audio_bytes))
        transcription_request = self.bindings.transcription_request_type(
            file=upload,
            model=self.served_model_name,
            language=language_hint,
            prompt=_sanitize_transcription_context(request.context),
            response_format="json",
            use_beam_search=True,
            n=beam_width,
            temperature=0.0,
            max_completion_tokens=request.max_new_tokens,
            length_penalty=1.0,
        )

        start_time = time.perf_counter()
        request_id = f"kotonoha-{uuid4()}"
        preprocessed = await self.transcription._preprocess_speech_to_text(
            request=transcription_request,
            audio_data=audio_bytes,
            request_id=request_id,
        )
        engine_inputs = preprocessed[0]
        parameters = transcription_request.to_beam_search_params(
            request.max_new_tokens,
            self.transcription.default_sampling_params,
        )
        hypotheses_by_rank: list[list[str]] = [[] for _ in range(candidate_count)]
        score_totals = [0.0] * candidate_count
        token_totals = [0] * candidate_count
        languages_by_rank: list[list[str | None]] = [[] for _ in range(candidate_count)]
        for chunk_index, engine_input in enumerate(engine_inputs):
            final_output = None
            generator = self.transcription.beam_search(
                prompt=engine_input,
                params=parameters,
                request_id=f"{request_id}-{chunk_index}",
            )
            async for output in generator:
                final_output = output
            if final_output is None:
                raise RuntimeError("vLLM returned no ASR output")
            if len(final_output.outputs) < candidate_count:
                raise RuntimeError(
                    f"vLLM returned {len(final_output.outputs)} hypotheses; "
                    f"expected {candidate_count}"
                )
            for rank, output in enumerate(final_output.outputs[:candidate_count]):
                raw_text = output.text or ""
                text, language = _parse_vllm_output(
                    raw_text,
                    QWEN_LANG.get(request.language_hint or ""),
                )
                processed_text = self.transcription.model_cls.post_process_output(text)
                hypotheses_by_rank[rank].append(processed_text.strip())
                score_totals[rank] += float(output.cumulative_logprob or 0.0)
                token_count = max(1, len(output.token_ids))
                token_totals[rank] += token_count
                languages_by_rank[rank].append(language)
        inference_ms = (time.perf_counter() - start_time) * 1000

        hypotheses: list[dict[str, Any]] = []
        languages: list[str | None] = []
        for rank in range(candidate_count):
            available_languages = [item for item in languages_by_rank[rank] if item]
            rank_language = available_languages[0] if available_languages else None
            separator = "" if rank_language in {"Chinese", "Japanese"} else " "
            text = separator.join(part for part in hypotheses_by_rank[rank] if part).strip()
            hypotheses.append(
                {
                    "text": text,
                    "avg_logprob": score_totals[rank] / max(1, token_totals[rank]),
                }
            )
            languages.append(rank_language)

        language, confidence = _vote_language(languages)
        if language is None and hypotheses:
            language, script_confidence = detect_script(hypotheses[0]["text"])
            confidence = script_confidence if language is not None else None
        return {
            "hypotheses": hypotheses,
            "language": language,
            "language_confidence": confidence,
            "duration_s": round(len(audio) / 16000.0, 3),
            "infer_ms": round(inference_ms, 1),
        }

    async def handle_websocket(
        self,
        websocket: WebSocket,
        /,
    ) -> None:
        if self.realtime is None or self.bindings is None:
            await websocket.accept()
            await websocket.send_json(
                {
                    "type": "error",
                    "error": "vLLM realtime transcription is not ready",
                    "code": "service_unavailable",
                }
            )
            await websocket.close(code=1013)
            return
        filtered_websocket = RealtimeWebSocketAdapter(websocket)
        connection = self.bindings.realtime_connection_type(
            filtered_websocket,
            self.realtime,
        )
        await connection.handle_connection()


def _iso_language(
    language: str | None,
    /,
) -> str | None:
    return {
        "ko": "ko",
        "en": "en",
        "ja": "ja",
        "zh-TW": "zh",
    }.get(language or "")


def _wav_bytes(
    audio: np.ndarray,
    /,
) -> bytes:
    samples = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples.tobytes())
    return stream.getvalue()


def _safe_realtime_text(
    raw_text: str,
    /,
    *,
    final: bool,
) -> str:
    cleaned = _QWEN_STREAM_PREFIX.sub("", raw_text)
    if final or not cleaned:
        return cleaned

    lowered = cleaned.lower()
    possible_prefix = "language "
    maximum_candidate_length = 50 + len(QWEN_ASR_TEXT_TAG)
    for start in range(len(cleaned) - 1, -1, -1):
        suffix = lowered[start:]
        partial_label = possible_prefix.startswith(suffix)
        incomplete_prefix = (
            suffix.startswith(possible_prefix)
            and "\n" not in suffix
            and len(suffix) < maximum_candidate_length
            and QWEN_ASR_TEXT_TAG not in suffix
        )
        if partial_label or incomplete_prefix:
            return cleaned[:start]
    return cleaned


class RealtimeWebSocketAdapter:
    """Delegate vLLM's connection while filtering Qwen control prefixes."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "_emitted_text",
        "_raw_text",
        "_websocket",
    )

    _emitted_text: str
    _raw_text: str
    _websocket: WebSocket

    def __init__(
        self,
        /,
        websocket: WebSocket,
    ) -> None:
        self._emitted_text = ""
        self._raw_text = ""
        self._websocket = websocket

    async def accept(
        self,
        /,
        *arguments: Any,
        **keywords: Any,
    ) -> None:
        await self._websocket.accept(*arguments, **keywords)

    async def receive_text(
        self,
        /,
    ) -> str:
        return await self._websocket.receive_text()

    async def send_text(
        self,
        data: str,
        /,
    ) -> None:
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            await self._websocket.send_text(data)
            return
        event_type = event.get("type")
        if event_type == "transcription.delta":
            self._raw_text += str(event.get("delta", ""))
            safe_text = _safe_realtime_text(self._raw_text, final=False)
            delta = safe_text[len(self._emitted_text) :]
            if not delta:
                return
            self._emitted_text = safe_text
            event["delta"] = delta
        elif event_type == "transcription.done":
            self._raw_text = str(event.get("text", self._raw_text))
            safe_text = _safe_realtime_text(self._raw_text, final=True)
            delta = safe_text[len(self._emitted_text) :]
            if delta:
                await self._websocket.send_text(
                    json.dumps(
                        {"type": "transcription.delta", "delta": delta},
                        ensure_ascii=False,
                    )
                )
            self._emitted_text = safe_text
            event["text"] = safe_text
        await self._websocket.send_text(json.dumps(event, ensure_ascii=False))


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
) -> AsyncIterator[None]:
    del app
    STATE["backend"] = None
    STATE["error"] = None
    settings = await asyncio.to_thread(
        load_settings,
        os.environ.get("KOTONOHA_CONFIG"),
    )
    try:
        if settings.asr.backend == "vllm":
            backend = VllmBackend(
                settings.asr.vllm_model_id,
                settings.asr.vllm_served_model_name,
                settings.asr.vllm_realtime_architecture,
                settings.asr.dtype,
                settings.asr.vllm_gpu_memory_utilization,
                settings.asr.vllm_max_model_len,
                settings.asr.vllm_enforce_eager,
            )
            await backend.start()
            STATE["backend"] = backend
        else:
            STATE["backend"] = await asyncio.to_thread(
                TransformersBackend,
                settings.asr.model_id,
                settings.asr.dtype,
            )
    except Exception as error:  # noqa: BLE001
        STATE["error"] = repr(error)
        log.exception("asr.load_failed", error=repr(error))
    try:
        yield
    finally:
        backend = STATE["backend"]
        if isinstance(backend, VllmBackend):
            await backend.shutdown()
        STATE["backend"] = None


app = FastAPI(title="kotonoha-asr", lifespan=lifespan)
install_auth(app, "asr")
app.include_router(config_admin_router)


@app.get("/health")
@keyword_compatible
async def health() -> dict:
    backend = STATE["backend"]
    if isinstance(backend, VllmBackend):
        return await backend.health()
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
async def transcribe(
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
    if isinstance(backend, VllmBackend):
        return await backend.transcribe(audio, request)
    return await asyncio.to_thread(backend.transcribe, audio, request)


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
    if isinstance(backend, VllmBackend):
        result = await backend.transcribe(pcm, request)
    else:
        result = await asyncio.to_thread(backend.transcribe, pcm, request)
    result["received_bytes"] = len(raw)
    return result


@app.websocket("/v1/realtime")
async def realtime_transcription(
    websocket: WebSocket,
    /,
) -> None:
    """Expose vLLM's realtime protocol from this resident FastAPI process."""
    if not websocket_authorized(websocket, "asr"):
        await websocket.close(code=4401, reason="unauthorized")
        return
    backend = STATE["backend"]
    if not isinstance(backend, VllmBackend):
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "error": "vLLM realtime transcription is unavailable",
                "code": "service_unavailable",
            }
        )
        await websocket.close(code=1013)
        return
    await backend.handle_websocket(websocket)


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
