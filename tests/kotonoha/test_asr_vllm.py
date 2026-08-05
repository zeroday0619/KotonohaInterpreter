"""vLLM ASR backend contracts that do not require a model or GPU."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import Mock

import numpy as np
import pytest

from kotonoha._config import DEFAULT_CONFIG, AsrConfig, read_yaml
from kotonoha.services._asr_server import (
    TranscribeRequest,
    VllmBackend,
    VllmRuntimeBindings,
    _parse_vllm_output,
    _safe_realtime_text,
    _validate_absolute_model_directory,
    _vllm_engine_arguments,
    _wav_bytes,
)
from kotonoha.services._auth import websocket_authorized


class FakeTranscriptionRequest:
    __slots__: ClassVar[tuple[str, ...]] = ("fields",)
    fields: dict[str, Any]

    def __init__(
        self,
        /,
        **fields: Any,
    ) -> None:
        self.fields = fields

    def to_beam_search_params(
        self,
        /,
        default_max_tokens: int,
        default_sampling_params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "max_tokens": default_max_tokens,
            "defaults": default_sampling_params,
        }


class FakeModelClass:
    __slots__: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def post_process_output(
        cls,
        text: str,
        /,
    ) -> str:
        del cls
        return text


class FakeTranscriptionServing:
    __slots__: ClassVar[tuple[str, ...]] = (
        "default_sampling_params",
        "model_cls",
    )

    default_sampling_params: dict[str, Any]
    model_cls: type[FakeModelClass]

    def __init__(
        self,
        /,
    ) -> None:
        self.default_sampling_params = {}
        self.model_cls = FakeModelClass

    async def _preprocess_speech_to_text(
        self,
        /,
        **arguments: Any,
    ) -> tuple[list[dict[str, str]], float]:
        assert arguments["audio_data"][:4] == b"RIFF"
        return ([{"type": "tokens"}], 2.0)

    async def beam_search(
        self,
        /,
        **arguments: Any,
    ) -> Any:
        assert arguments["params"]["max_tokens"] == 256
        outputs = [
            SimpleNamespace(
                text=f"language Korean<asr_text>candidate {index}<|im_end|>",
                cumulative_logprob=-float(index + 1),
                token_ids=[1, 2],
            )
            for index in range(5)
        ]
        yield SimpleNamespace(outputs=outputs)


def test_asr_defaults_to_vllm() -> None:
    config = AsrConfig()
    default_config = read_yaml(DEFAULT_CONFIG)["asr"]

    assert config.backend == "vllm"
    assert default_config["backend"] == "vllm"
    assert config.vllm_model_id == "Qwen/Qwen3-ASR-0.6B"
    assert default_config["vllm_model_id"] == config.vllm_model_id
    assert config.vllm_realtime_architecture == "qwen3_asr"
    assert config.n_best == 5


def test_vllm_wav_adapter_preserves_sixteen_kilohertz_pcm() -> None:
    audio = np.zeros(16000, dtype=np.float32)

    encoded = _wav_bytes(audio)

    assert encoded[:4] == b"RIFF"
    assert encoded[8:12] == b"WAVE"
    assert len(encoded) == 44 + 32000


def test_vllm_engine_arguments_select_target_realtime_architectures() -> None:
    qwen = _vllm_engine_arguments(
        "/models/Qwen3-ASR-0.6B",
        "kotonoha-asr",
        "qwen3_asr",
        "float16",
        0.8,
        4096,
        True,
    )
    voxtral = _vllm_engine_arguments(
        "/models/Voxtral-Mini-4B-Realtime-2602",
        "kotonoha-asr",
        "voxtral",
        "bfloat16",
        0.20,
        4096,
        True,
    )

    assert qwen["hf_overrides"] == {
        "architectures": ["Qwen3ASRRealtimeGeneration"]
    }
    assert qwen["tokenizer_mode"] == "auto"
    assert voxtral["hf_overrides"] == {}
    assert voxtral["tokenizer_mode"] == "mistral"


def test_vllm_model_path_validation_allows_non_absolute_identifiers() -> None:
    _validate_absolute_model_directory("Qwen/Qwen3-ASR-0.6B")
    _validate_absolute_model_directory("models/Qwen3-ASR-0.6B")
    _validate_absolute_model_directory(str(DEFAULT_CONFIG.parent))


async def test_vllm_backend_rejects_a_missing_absolute_model_before_engine_arguments(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    engine_arguments_type = Mock()
    backend = object.__new__(VllmBackend)
    backend.bindings = SimpleNamespace(engine_arguments_type=engine_arguments_type)
    backend.model_id = str(tmp_path / "missing-model")

    with pytest.raises(FileNotFoundError, match="Absolute ASR model path is missing"):
        await backend.start()

    engine_arguments_type.assert_not_called()


def test_vllm_output_extracts_language_and_transcription() -> None:
    text, language = _parse_vllm_output(
        "language Japanese<asr_text>こんにちは<|im_end|>",
        None,
    )

    assert text == "こんにちは"
    assert language == "Japanese"


async def test_vllm_backend_returns_five_scored_hypotheses() -> None:
    backend = object.__new__(VllmBackend)
    backend.bindings = VllmRuntimeBindings(
        engine_arguments_type=None,
        engine_context_type=None,
        model_path_type=None,
        models_type=None,
        realtime_connection_type=None,
        realtime_serving_type=None,
        transcription_request_type=FakeTranscriptionRequest,
        transcription_serving_type=None,
    )
    backend.transcription = FakeTranscriptionServing()
    backend.load_seconds = 0.0
    backend.served_model_name = "kotonoha-asr"

    result = await backend.transcribe(
        np.zeros(32000, dtype=np.float32),
        TranscribeRequest(n_best=5, num_beams=5),
    )

    assert len(result["hypotheses"]) == 5
    assert result["hypotheses"][0] == {"text": "candidate 0", "avg_logprob": -0.5}
    assert result["language"] == "Korean"
    assert result["language_confidence"] == 1.0
    assert result["duration_s"] == 2.0


def test_realtime_filter_removes_qwen_control_prefixes_across_chunks() -> None:
    raw = "language Korean<asr_text>안녕language Korean<asr_text>하세요"

    assert _safe_realtime_text("lang", final=False) == ""
    assert _safe_realtime_text("language Korean<asr_", final=False) == ""
    assert _safe_realtime_text(raw, final=False) == "안녕하세요"
    assert _safe_realtime_text(raw, final=True) == "안녕하세요"


def test_realtime_websocket_requires_the_shared_service_token(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("KOTONOHA_SERVICE_TOKEN", "secret")
    unauthorized = SimpleNamespace(
        headers={},
        url=SimpleNamespace(path="/v1/realtime"),
    )
    authorized = SimpleNamespace(
        headers={"authorization": "Bearer secret"},
        url=SimpleNamespace(path="/v1/realtime"),
    )

    assert websocket_authorized(unauthorized, "asr") is False
    assert websocket_authorized(authorized, "asr") is True


def test_asr_service_embeds_vllm_without_launching_an_internal_server() -> None:
    from kotonoha.services import _asr_server

    source = Path(_asr_server.__file__).read_text(encoding="utf-8")

    assert "build_async_engine_client_from_engine_args" in source
    assert "OpenAIServingRealtime" in source
    assert "RealtimeConnection" in source
    assert "create_subprocess_exec" not in source
