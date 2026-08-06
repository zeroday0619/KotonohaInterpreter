"""FastAPI in-process vLLM-Omni TTS service contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from kotonoha._config import TextToSpeechVoices
from kotonoha.services import _tts_server


class FakeSpeechService:
    __slots__: ClassVar[tuple[str, ...]] = ("request", "raw_request")

    request: dict[str, Any] | None
    raw_request: Request | None

    def __init__(
        self,
        /,
    ) -> None:
        self.request = None
        self.raw_request = None

    async def create_speech(
        self,
        request: dict[str, Any],
        raw_request: Request,
        /,
    ) -> StreamingResponse:
        self.request = request
        self.raw_request = raw_request

        async def pcm_stream() -> AsyncIterator[bytes]:
            yield b"\x00\x00\x01\x00"

        return StreamingResponse(pcm_stream(), media_type="audio/pcm")


class FakeErrorResponse:
    __slots__: ClassVar[tuple[str, ...]] = ()


def test_vllm_omni_service_wraps_the_engine_without_an_internal_server() -> None:
    source = Path(_tts_server.__file__).read_text(encoding="utf-8")

    assert "AsyncOmni" in source
    assert "OmniOpenAIServingSpeech" in source
    assert "class VllmOmniRuntime" in source
    assert "create_subprocess_exec" not in source
    assert "httpx2.AsyncClient" not in source
    assert "TTS_UPSTREAM_PORT" not in source


def test_vllm_omni_service_enforces_matching_release_versions(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {"vllm": "0.25.1", "vllm-omni": "0.26.0"}
    monkeypatch.setattr(_tts_server, "version", versions.__getitem__)

    with pytest.raises(RuntimeError, match="major/minor versions must match"):
        _tts_server._validate_runtime_versions()


def test_qwen_custom_voice_defaults_and_language_constraints() -> None:
    voices = TextToSpeechVoices()

    assert voices.for_language("ko") == "Sohee"
    assert voices.for_language("en") == "Ryan"
    assert voices.for_language("ja") == "Ono_Anna"
    assert voices.for_language("zh-TW") == "Vivian"
    assert TextToSpeechVoices(en="Aiden", zh_tw="Serena").en == "Aiden"
    with pytest.raises(ValidationError):
        TextToSpeechVoices(ko="Vivian")


def test_speech_request_normalizes_and_checks_the_native_voice() -> None:
    request = _tts_server.SpeechRequest(
        input="안녕하세요",
        voice="sohee",
        language="Korean",
    )

    assert request.voice == "Sohee"
    with pytest.raises(ValidationError, match="not native to Korean"):
        _tts_server.SpeechRequest(
            input="안녕하세요",
            voice="Vivian",
            language="Korean",
        )


@pytest.mark.asyncio
async def test_fastapi_speech_endpoint_returns_the_direct_runtime_stream(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    speech = FakeSpeechService()
    bindings = _tts_server.RuntimeBindings(
        engine_type=None,
        model_path_type=None,
        models_type=None,
        request_type=dict,
        error_type=FakeErrorResponse,
        speech_type=None,
    )
    configuration = _tts_server.RuntimeConfiguration(
        model=tmp_path / "model",
        served_model_name="kotonoha-tts",
        gpu_memory_utilization=0.25,
        enforce_eager=True,
        startup_timeout_seconds=600.0,
        deploy_config=tmp_path / "qwen3_tts.yaml",
        vllm_version="0.26.0",
        omni_version="0.26.0",
    )
    monkeypatch.setattr(_tts_server.RUNTIME, "bindings", bindings)
    monkeypatch.setattr(_tts_server.RUNTIME, "configuration", configuration)
    monkeypatch.setattr(_tts_server.RUNTIME, "engine", object())
    monkeypatch.setattr(_tts_server.RUNTIME, "_ready", True)
    monkeypatch.setattr(_tts_server.RUNTIME, "speech", speech)
    raw_request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/speech",
            "headers": [],
        }
    )

    response = await _tts_server.create_speech(
        _tts_server.SpeechRequest(
            input="안녕하세요",
            voice="Sohee",
            language="Korean",
        ),
        raw_request,
    )
    chunks = [chunk async for chunk in response.body_iterator]

    assert b"".join(chunks) == b"\x00\x00\x01\x00"
    assert response.media_type == "audio/pcm"
    assert speech.request is not None
    assert speech.request["model"] == "kotonoha-tts"
    assert speech.request["voice"] == "Sohee"
    assert speech.request["stream_format"] == "audio"
    assert speech.raw_request is raw_request
