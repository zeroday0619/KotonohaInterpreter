"""FastAPI service owning an in-process vLLM TranslateGemma engine."""

from __future__ import annotations

import inspect
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, ClassVar, Final, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from kotonoha._call_compatibility import keyword_compatible
from kotonoha._config import LanguageModelConfig, load_settings
from kotonoha._logging_setup import setup_logging
from kotonoha.services._auth import install_auth, websocket_authorized
from kotonoha.services._resources import resource_report

log = setup_logging(service="llm", console=True)

MINIMUM_VLLM_VERSION: Final = (0, 19, 0)


@dataclass(frozen=True, slots=True)
class VllmRuntimeBindings:
    engine_arguments_type: Any
    engine_context_type: Any
    sampling_parameters_type: Any


class TranslateGemmaTextContent(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    type: Literal["text"] = "text"
    source_lang_code: Literal["ko", "en", "ja", "zh-TW"]
    target_lang_code: Literal["ko", "en", "ja", "zh-TW"]
    text: str = Field(min_length=1)


class TranslateGemmaMessage(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    role: Literal["user"] = "user"
    content: list[TranslateGemmaTextContent] = Field(min_length=1, max_length=1)


class TranslationRequest(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    type: Literal["translation.create"] = "translation.create"
    model: str
    messages: list[TranslateGemmaMessage] = Field(min_length=1, max_length=1)
    temperature: float = Field(0.0, ge=0.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, gt=0.0)
    max_tokens: int = Field(512, ge=1)


class ChatCompletionRequest(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    model: str
    messages: list[TranslateGemmaMessage] = Field(min_length=1, max_length=1)
    temperature: float = Field(0.0, ge=0.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, gt=0.0)
    max_tokens: int = Field(512, ge=1)
    stream: bool = True

    def translation_request(
        self,
        /,
    ) -> TranslationRequest:
        return TranslationRequest(**self.model_dump(exclude={"stream"}))


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
    return VllmRuntimeBindings(
        engine_arguments_type=_runtime_symbol(
            ("vllm.engine.arg_utils",),
            "AsyncEngineArgs",
        ),
        engine_context_type=_runtime_symbol(
            ("vllm.entrypoints.openai.api_server",),
            "build_async_engine_client_from_engine_args",
        ),
        sampling_parameters_type=_runtime_symbol(
            ("vllm", "vllm.sampling_params"),
            "SamplingParams",
        ),
    )


def _numeric_version(
    raw_version: str,
    /,
) -> tuple[int, ...]:
    components = raw_version.split("+", 1)[0].split(".")
    parsed: list[int] = []
    for component in components[:3]:
        digits = "".join(character for character in component if character.isdigit())
        parsed.append(int(digits or "0"))
    return tuple(parsed)


def _engine_arguments(
    config: LanguageModelConfig,
    /,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "model": str(config.model_path),
        "served_model_name": [config.served_model_name],
        "dtype": config.active.dtype,
        "max_model_len": config.max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "enforce_eager": config.enforce_eager,
        "trust_remote_code": True,
    }
    if config.kv_cache_dtype != "auto":
        arguments["kv_cache_dtype"] = config.kv_cache_dtype
    arguments["max_num_seqs"] = config.max_num_seqs
    arguments["enable_prefix_caching"] = config.enable_prefix_caching
    if config.limit_mm_per_prompt is not None:
        arguments["limit_mm_per_prompt"] = config.limit_mm_per_prompt
    if config.max_num_batched_tokens is not None:
        arguments["max_num_batched_tokens"] = config.max_num_batched_tokens
    if config.compilation_mode is not None:
        compilation_config: dict[str, Any] = {"mode": config.compilation_mode}
        if config.compilation_cudagraph_capture_sizes:
            compilation_config["cudagraph_capture_sizes"] = list(
                config.compilation_cudagraph_capture_sizes
            )
        if config.compilation_cache_dir is not None:
            compilation_config["cache_dir"] = str(config.compilation_cache_dir)
        arguments["compilation_config"] = compilation_config
    if config.active.quantization is not None:
        arguments["quantization"] = config.active.quantization
    return arguments


class VllmTranslationBackend:
    """Own one asynchronous engine and expose cumulative output as deltas."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "bindings",
        "config",
        "engine",
        "engine_context",
        "error",
        "load_seconds",
        "tokenizer",
    )

    name: Final = "vllm_in_process"
    bindings: VllmRuntimeBindings
    config: LanguageModelConfig
    engine: Any
    engine_context: Any
    error: str | None
    load_seconds: float
    tokenizer: Any

    def __init__(
        self,
        config: LanguageModelConfig,
        /,
    ) -> None:
        self.config = config
        self.engine = None
        self.engine_context = None
        self.error = None
        self.load_seconds = 0.0
        self.tokenizer = None
        try:
            runtime_version = version("vllm")
        except PackageNotFoundError as error:
            raise RuntimeError("vLLM is not installed in the translation service image") from error
        if _numeric_version(runtime_version) < MINIMUM_VLLM_VERSION:
            raise RuntimeError(
                f"In-process translation requires vLLM >= 0.19.0; found {runtime_version}"
            )
        self.bindings = _load_vllm_runtime_bindings()

    async def start(
        self,
        /,
    ) -> None:
        start_time = time.perf_counter()
        try:
            arguments = self.bindings.engine_arguments_type(**_engine_arguments(self.config))
            engine_context = self.bindings.engine_context_type(arguments)
            self.engine = await engine_context.__aenter__()
            self.engine_context = engine_context
            tokenizer = self.engine.get_tokenizer()
            self.tokenizer = await tokenizer if inspect.isawaitable(tokenizer) else tokenizer
        except Exception:
            await self.shutdown()
            raise
        self.load_seconds = round(time.perf_counter() - start_time, 2)
        log.info(
            "llm.loaded",
            backend=self.name,
            model=str(self.config.model_path),
            served_model_name=self.config.served_model_name,
            load_s=self.load_seconds,
            vllm_version=version("vllm"),
        )

    async def shutdown(
        self,
        /,
    ) -> None:
        self.tokenizer = None
        if self.engine_context is not None:
            await self.engine_context.__aexit__(None, None, None)
            self.engine_context = None
        self.engine = None

    async def health(
        self,
        /,
    ) -> dict[str, Any]:
        ready = self.engine is not None and self.tokenizer is not None
        if ready:
            try:
                await self.engine.check_health()
                self.error = None
            except Exception as error:  # noqa: BLE001
                ready = False
                self.error = repr(error)
        result = {
            "ok": ready,
            "service": "llm",
            "backend": self.name if ready else None,
            "model": str(self.config.model_path),
            "served_model_name": self.config.served_model_name,
            "vllm": version("vllm"),
            "error": self.error,
            "resources": resource_report(
                "llm",
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_num_seqs=self.config.max_num_seqs,
                prefix_caching=self.config.enable_prefix_caching,
            ),
        }
        try:
            import torch

            if torch.cuda.is_available():
                result["gpu_memory_allocated_mib"] = round(
                    torch.cuda.memory_allocated() / 1024**2,
                    1,
                )
                result["gpu_memory_reserved_mib"] = round(
                    torch.cuda.memory_reserved() / 1024**2,
                    1,
                )
                result["gpu_max_memory_reserved_mib"] = round(
                    torch.cuda.max_memory_reserved() / 1024**2,
                    1,
                )
        except ImportError:
            pass
        return result

    def render_prompt(
        self,
        messages: list[TranslateGemmaMessage],
        /,
    ) -> str:
        if self.tokenizer is None:
            raise RuntimeError("TranslateGemma tokenizer is not ready")
        return self.tokenizer.apply_chat_template(
            [message.model_dump() for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )

    async def stream(
        self,
        request: TranslationRequest,
        /,
    ) -> AsyncIterator[tuple[str, int]]:
        if self.engine is None:
            raise RuntimeError("vLLM translation engine is not ready")
        if request.model != self.config.served_model_name:
            raise ValueError(f"unknown served model: {request.model}")
        prompt = self.render_prompt(request.messages)
        parameters = self.bindings.sampling_parameters_type(
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            max_tokens=request.max_tokens,
        )
        request_id = f"translation-{uuid4()}"
        emitted_text = ""
        completed = False
        try:
            async for output in self.engine.generate(prompt, parameters, request_id):
                if not output.outputs:
                    continue
                candidate = output.outputs[0]
                text = candidate.text or ""
                delta = text[len(emitted_text) :]
                emitted_text = text
                if delta:
                    yield delta, len(candidate.token_ids)
            completed = True
        finally:
            if not completed:
                abort_result = self.engine.abort(request_id)
                if inspect.isawaitable(abort_result):
                    await abort_result

    async def handle_websocket(
        self,
        websocket: WebSocket,
        /,
    ) -> None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "session.created",
                "model": self.config.served_model_name,
            }
        )
        try:
            while True:
                event = await websocket.receive_json()
                if event.get("type") != "translation.create":
                    await websocket.send_json(
                        {"type": "error", "error": "expected translation.create"}
                    )
                    continue
                request = TranslationRequest.model_validate(event)
                completion_tokens = 0
                complete_text = ""
                async for delta, current_completion_tokens in self.stream(request):
                    completion_tokens = current_completion_tokens
                    complete_text += delta
                    await websocket.send_json({"type": "translation.delta", "delta": delta})
                await websocket.send_json(
                    {
                        "type": "translation.done",
                        "text": complete_text,
                        "usage": {"completion_tokens": completion_tokens},
                    }
                )
        except WebSocketDisconnect:
            return
        except Exception as error:  # noqa: BLE001
            log.exception("llm.websocket_failed", error=repr(error))
            await websocket.send_json({"type": "error", "error": repr(error)})


STATE: dict[str, Any] = {"backend": None, "error": None}


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    /,
) -> AsyncIterator[None]:
    del app
    STATE["backend"] = None
    STATE["error"] = None
    settings = load_settings(os.environ.get("KOTONOHA_CONFIG"))
    try:
        backend = VllmTranslationBackend(settings.llm)
        await backend.start()
        STATE["backend"] = backend
    except Exception as error:  # noqa: BLE001
        STATE["error"] = repr(error)
        log.exception("llm.load_failed", error=repr(error))
    try:
        yield
    finally:
        backend = STATE["backend"]
        if isinstance(backend, VllmTranslationBackend):
            await backend.shutdown()
        STATE["backend"] = None


app = FastAPI(title="kotonoha-translation", lifespan=lifespan)
install_auth(app, "llm")


def _backend() -> VllmTranslationBackend:
    backend = STATE["backend"]
    if not isinstance(backend, VllmTranslationBackend):
        raise HTTPException(503, f"llm backend not loaded: {STATE['error']}")
    return backend


@app.get("/health")
@keyword_compatible
async def health() -> dict[str, Any]:
    backend = STATE["backend"]
    if isinstance(backend, VllmTranslationBackend):
        return await backend.health()
    return {
        "ok": False,
        "service": "llm",
        "backend": None,
        "error": STATE["error"],
    }


@app.websocket("/v1/realtime")
@keyword_compatible
async def realtime_translation(
    websocket: WebSocket,
    /,
) -> None:
    if not websocket_authorized(websocket, "llm"):
        await websocket.close(code=4401, reason="unauthorized")
        return
    backend = STATE["backend"]
    if not isinstance(backend, VllmTranslationBackend):
        await websocket.accept()
        await websocket.send_json(
            {"type": "error", "error": f"llm backend not loaded: {STATE['error']}"}
        )
        await websocket.close(code=1013)
        return
    await backend.handle_websocket(websocket)


@app.post("/v1/chat/completions")
@keyword_compatible
async def chat_completions(
    request: ChatCompletionRequest,
    /,
) -> StreamingResponse:
    if not request.stream:
        raise HTTPException(400, "only streaming chat completions are supported")
    backend = _backend()
    translation_request = request.translation_request()

    async def events() -> AsyncIterator[str]:
        completion_tokens = 0
        async for delta, current_completion_tokens in backend.stream(translation_request):
            completion_tokens = current_completion_tokens
            event = {"choices": [{"delta": {"content": delta}}]}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        usage = {"choices": [], "usage": {"completion_tokens": completion_tokens}}
        yield f"data: {json.dumps(usage)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
