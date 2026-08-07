"""FastAPI service wrapping the in-process vLLM-Omni Qwen3-TTS runtime."""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from http import HTTPStatus
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator, model_validator

from kotonoha._call_compatibility import keyword_compatible
from kotonoha._config import QWEN_LANGUAGE_VOICES, QWEN_VOICE_NAMES, QwenVoice
from kotonoha._logging_setup import setup_logging
from kotonoha._prometheus import install_metrics, observe_service_health
from kotonoha.services._auth import install_auth
from kotonoha.services._resources import resource_report

log = setup_logging(service="tts", console=True)

EXPECTED_RELEASE: Final = (0, 26)
DEFAULT_MODEL: Final = "/models/Qwen3-TTS-0.6B"
DEFAULT_SERVED_MODEL_NAME: Final = "kotonoha-tts"
DEFAULT_GPU_MEMORY_UTILIZATION: Final = 0.30
DEFAULT_STARTUP_TIMEOUT_SECONDS: Final = 600.0


@dataclass(frozen=True, slots=True)
class RuntimeConfiguration:
    model: Path
    served_model_name: str
    gpu_memory_utilization: float
    enforce_eager: bool
    startup_timeout_seconds: float
    deploy_config: Path
    vllm_version: str
    omni_version: str


@dataclass(frozen=True, slots=True)
class RuntimeBindings:
    engine_type: Any
    model_path_type: Any
    models_type: Any
    request_type: Any
    error_type: Any
    speech_type: Any


class SpeechRequest(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    input: str = Field(min_length=1)
    model: str | None = None
    voice: QwenVoice = "Vivian"
    language: Literal["Auto", "Chinese", "English", "Japanese", "Korean"] = "Auto"
    task_type: Literal["CustomVoice"] = "CustomVoice"
    response_format: Literal["pcm"] = "pcm"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    stream: Literal[True] = True
    stream_format: Literal["audio"] = "audio"
    max_new_tokens: int = Field(2048, ge=1, le=4096)

    @field_validator("voice", mode="before")
    @classmethod
    def _normalize_voice(
        cls,
        value: Any,
        /,
    ) -> Any:
        del cls
        if not isinstance(value, str):
            return value
        return QWEN_VOICE_NAMES.get(value.strip().lower(), value)

    @model_validator(mode="after")
    def _validate_language_voice(
        self,
        /,
    ) -> SpeechRequest:
        allowed = QWEN_LANGUAGE_VOICES.get(self.language)
        if allowed is not None and self.voice not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(
                f"voice {self.voice} is not native to {self.language}; choose one of: {choices}"
            )
        return self


def _parse_release(
    package_name: str,
    package_version: str,
    /,
) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", package_version)
    if match is None:
        raise RuntimeError(f"Cannot parse {package_name} version: {package_version}")
    return int(match.group(1)), int(match.group(2))


def _validate_runtime_versions() -> tuple[str, str]:
    try:
        vllm_version = version("vllm")
        omni_version = version("vllm-omni")
    except PackageNotFoundError as error:
        raise RuntimeError(f"Required runtime package is not installed: {error.name}") from error

    vllm_release = _parse_release("vLLM", vllm_version)
    omni_release = _parse_release("vLLM-Omni", omni_version)
    if vllm_release != omni_release:
        raise RuntimeError(
            "vLLM and vLLM-Omni major/minor versions must match: "
            f"vLLM {vllm_version}, vLLM-Omni {omni_version}"
        )
    if omni_release != EXPECTED_RELEASE:
        expected = ".".join(str(component) for component in EXPECTED_RELEASE)
        raise RuntimeError(
            f"This service requires vLLM and vLLM-Omni {expected}.x: "
            f"found {vllm_version} and {omni_version}"
        )
    return vllm_version, omni_version


def _parse_positive_float(
    name: str,
    value: str,
    /,
    *,
    maximum: float | None = None,
) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number: {value}") from error
    if parsed <= 0.0 or maximum is not None and parsed > maximum:
        range_description = f" and at most {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be greater than 0{range_description}: {parsed}")
    return parsed


def _parse_boolean(
    name: str,
    value: str,
    /,
) -> bool:
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"{name} must be 0 or 1: {value}")


def _default_deploy_config() -> Path:
    module_specification = find_spec("vllm_omni")
    if module_specification is None or not module_specification.submodule_search_locations:
        raise RuntimeError("Cannot locate the installed vLLM-Omni package")
    package_directory = Path(next(iter(module_specification.submodule_search_locations)))
    return package_directory / "deploy" / "qwen3_tts.yaml"


def runtime_configuration() -> RuntimeConfiguration:
    model = Path(os.environ.get("TTS_MODEL", DEFAULT_MODEL))
    configuration_path = model / "config.json"
    if not configuration_path.is_file() or configuration_path.stat().st_size == 0:
        raise RuntimeError(f"vLLM-Omni TTS model snapshot is incomplete: {model}")
    vllm_version, omni_version = _validate_runtime_versions()
    configured_deploy_config = os.environ.get("TTS_DEPLOY_CONFIG")
    deploy_config = (
        Path(configured_deploy_config)
        if configured_deploy_config is not None
        else _default_deploy_config()
    )
    if not deploy_config.is_file() or deploy_config.stat().st_size == 0:
        raise RuntimeError(f"vLLM-Omni Qwen3-TTS deploy config is missing: {deploy_config}")

    return RuntimeConfiguration(
        model=model,
        served_model_name=os.environ.get(
            "TTS_SERVED_MODEL_NAME",
            DEFAULT_SERVED_MODEL_NAME,
        ),
        gpu_memory_utilization=_parse_positive_float(
            "TTS_GPU_MEMORY_UTILIZATION",
            os.environ.get(
                "TTS_GPU_MEMORY_UTILIZATION",
                str(DEFAULT_GPU_MEMORY_UTILIZATION),
            ),
            maximum=1.0,
        ),
        enforce_eager=_parse_boolean(
            "TTS_ENFORCE_EAGER",
            os.environ.get("TTS_ENFORCE_EAGER", "1"),
        ),
        startup_timeout_seconds=_parse_positive_float(
            "TTS_STARTUP_TIMEOUT_SECONDS",
            os.environ.get(
                "TTS_STARTUP_TIMEOUT_SECONDS",
                str(DEFAULT_STARTUP_TIMEOUT_SECONDS),
            ),
        ),
        deploy_config=deploy_config,
        vllm_version=vllm_version,
        omni_version=omni_version,
    )


def _load_runtime_bindings() -> RuntimeBindings:
    # These imports remain lazy so workstation tests do not require target GPU packages.
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

    return RuntimeBindings(
        engine_type=AsyncOmni,
        model_path_type=BaseModelPath,
        models_type=OpenAIServingModels,
        request_type=OpenAICreateSpeechRequest,
        error_type=ErrorResponse,
        speech_type=OmniOpenAIServingSpeech,
    )


def _configure_multiprocessing() -> None:
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "forkserver":
        return

    import multiprocessing
    import multiprocessing.forkserver as forkserver

    multiprocessing.set_start_method("forkserver")
    multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
    forkserver.ensure_running()


class VllmOmniRuntime:
    """Own the in-process vLLM-Omni engine and speech-serving adapter."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "bindings",
        "configuration",
        "engine",
        "error",
        "models",
        "speech",
        "_ready",
    )

    bindings: RuntimeBindings | None
    configuration: RuntimeConfiguration | None
    engine: Any
    error: str | None
    models: Any
    speech: Any
    _ready: bool

    def __init__(
        self,
        /,
    ) -> None:
        self.bindings = None
        self.configuration = None
        self.engine = None
        self.error = None
        self.models = None
        self.speech = None
        self._ready = False

    @property
    def ready(
        self,
        /,
    ) -> bool:
        return self._ready and self.engine is not None and self.speech is not None

    def _create_engine(
        self,
        configuration: RuntimeConfiguration,
        bindings: RuntimeBindings,
        /,
    ) -> Any:
        del self
        _configure_multiprocessing()
        return bindings.engine_type(
            model=str(configuration.model),
            deploy_config=str(configuration.deploy_config),
            gpu_memory_utilization=configuration.gpu_memory_utilization,
            enforce_eager=configuration.enforce_eager,
            trust_remote_code=True,
            init_timeout=configuration.startup_timeout_seconds,
            stage_init_timeout=configuration.startup_timeout_seconds,
            log_stats=True,
        )

    async def _create_speech_service(
        self,
        engine: Any,
        configuration: RuntimeConfiguration,
        bindings: RuntimeBindings,
        /,
    ) -> tuple[Any, Any]:
        del self
        vllm_configuration = await engine.get_vllm_config()
        if vllm_configuration is None:
            raise RuntimeError("vLLM-Omni did not expose a vLLM stage configuration")
        if getattr(engine, "model_config", None) is None:
            engine.model_config = vllm_configuration.model_config
        if getattr(engine, "input_processor", None) is None:
            raise RuntimeError("vLLM-Omni did not initialize its input processor")

        model_paths = [
            bindings.model_path_type(
                name=configuration.served_model_name,
                model_path=str(configuration.model),
            )
        ]
        models = bindings.models_type(
            engine_client=engine,
            base_model_paths=model_paths,
            lora_modules=None,
        )
        await models.init_static_loras()
        speech = bindings.speech_type(
            engine,
            models,
            request_logger=None,
            model_name=configuration.served_model_name,
            forced_aligner_config=None,
        )
        await speech.warmup()
        return models, speech

    async def start(
        self,
        /,
    ) -> None:
        self.shutdown()
        self.bindings = None
        self.configuration = None
        self.error = None
        self.models = None
        try:
            self.configuration = runtime_configuration()
            self.bindings = _load_runtime_bindings()
            self.engine = self._create_engine(self.configuration, self.bindings)
            self.models, self.speech = await self._create_speech_service(
                self.engine,
                self.configuration,
                self.bindings,
            )
            self._ready = True
            log.info(
                "tts.loaded",
                backend="vllm_omni_in_process",
                model=str(self.configuration.model),
                vllm=self.configuration.vllm_version,
                vllm_omni=self.configuration.omni_version,
            )
        except Exception as error:  # noqa: BLE001
            self.error = repr(error)
            log.exception("tts.load_failed", error=self.error)
            self.shutdown()

    def shutdown(
        self,
        /,
    ) -> None:
        self._ready = False
        if self.speech is not None:
            try:
                self.speech.shutdown()
            except Exception as error:  # noqa: BLE001
                log.warning("tts.speech_shutdown_failed", error=repr(error))
            self.speech = None
        if self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception as error:  # noqa: BLE001
                log.warning("tts.engine_shutdown_failed", error=repr(error))
            self.engine = None

    async def health(
        self,
        /,
    ) -> dict:
        ready = self.ready
        if ready:
            try:
                await self.engine.check_health()
            except Exception as error:  # noqa: BLE001
                ready = False
                self._ready = False
                self.error = repr(error)
        result = {
            "ok": ready,
            "service": "tts",
            "backend": "vllm_omni_in_process" if ready else None,
            "model": str(self.configuration.model) if self.configuration is not None else None,
            "served_model_name": (
                self.configuration.served_model_name
                if self.configuration is not None
                else None
            ),
            "vllm": self.configuration.vllm_version if self.configuration is not None else None,
            "vllm_omni": (
                self.configuration.omni_version if self.configuration is not None else None
            ),
            "error": self.error,
            "resources": resource_report(
                "tts",
                gpu_memory_utilization=(
                    self.configuration.gpu_memory_utilization
                    if self.configuration is not None
                    else None
                ),
                max_num_seqs=1,
                prefix_caching=False,
            ),
        }
        observe_service_health("tts", ready, result["resources"])
        return result

    def _error_response(
        self,
        error: Any,
        /,
    ) -> JSONResponse:
        del self
        status_code = HTTPStatus.BAD_REQUEST
        nested_error = getattr(error, "error", None)
        if nested_error is not None and nested_error.code is not None:
            status_code = nested_error.code
        payload = error.model_dump()
        if nested_error is not None:
            payload["error"]["code"] = int(status_code)
        return JSONResponse(content=payload, status_code=int(status_code))

    async def synthesize(
        self,
        request: SpeechRequest,
        raw_request: Request,
        /,
    ) -> Response:
        if not request.input.strip():
            raise HTTPException(400, "empty text")
        if not self.ready or self.bindings is None or self.configuration is None:
            raise HTTPException(503, f"tts backend not loaded: {self.error}")

        payload = request.model_dump(exclude_none=True)
        if payload.get("model") is None:
            payload["model"] = self.configuration.served_model_name
        omni_request = self.bindings.request_type(**payload)
        try:
            response = await self.speech.create_speech(omni_request, raw_request)
        except Exception as error:  # noqa: BLE001
            log.error("tts.synthesis_failed", error=repr(error))
            raise HTTPException(500, "vLLM-Omni synthesis failed") from error
        if isinstance(response, self.bindings.error_type):
            return self._error_response(response)
        if not isinstance(response, Response):
            raise HTTPException(500, "vLLM-Omni returned an unsupported speech response")
        media_type = (response.media_type or response.headers.get("content-type", ""))
        media_type = media_type.split(";", 1)[0].lower()
        if media_type != "audio/pcm":
            raise HTTPException(
                500,
                f"vLLM-Omni returned {media_type or 'no media type'} instead of audio/pcm",
            )
        body_iterator = getattr(response, "body_iterator", None)
        if body_iterator is not None:
            response.body_iterator = self._observe_audio_stream(body_iterator, request)
        response.headers["x-kotonoha-audio-format"] = "s16le"
        response.headers["x-kotonoha-sample-rate"] = "24000"
        log.info(
            "tts.synthesis_started",
            language=request.language,
            voice=request.voice,
            characters=len(request.input),
            max_new_tokens=request.max_new_tokens,
        )
        return response

    async def _observe_audio_stream(
        self,
        body_iterator: AsyncIterator[Any],
        request: SpeechRequest,
        /,
    ) -> AsyncIterator[Any]:
        del self
        byte_count = 0
        started_at = time.perf_counter()
        try:
            async for chunk in body_iterator:
                encoded_chunk = chunk.encode() if isinstance(chunk, str) else bytes(chunk)
                byte_count += len(encoded_chunk)
                yield chunk
        except Exception as error:
            log.exception(
                "tts.synthesis_stream_failed",
                language=request.language,
                voice=request.voice,
                bytes=byte_count,
                error=repr(error),
            )
            raise
        log.info(
            "tts.synthesis_finished",
            language=request.language,
            voice=request.voice,
            bytes=byte_count,
            audio_seconds=round(byte_count / (24000 * 2), 3),
            elapsed_ms=round((time.perf_counter() - started_at) * 1000, 1),
        )


RUNTIME = VllmOmniRuntime()


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    /,
) -> AsyncIterator[None]:
    app.state.tts_runtime = RUNTIME
    await RUNTIME.start()
    try:
        yield
    finally:
        await asyncio.to_thread(RUNTIME.shutdown)


app = FastAPI(title="kotonoha-tts", lifespan=lifespan)
install_auth(app, "tts")
install_metrics(app, "tts")


@app.get("/health")
@keyword_compatible
async def health() -> dict:
    return await RUNTIME.health()


@app.post("/v1/audio/speech")
@keyword_compatible
async def create_speech(
    request: SpeechRequest,
    raw_request: Request,
    /,
) -> Response:
    return await RUNTIME.synthesize(request, raw_request)
