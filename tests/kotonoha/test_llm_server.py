"""TranslateGemma prompt and in-process vLLM translation tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from fastapi import WebSocketDisconnect
from pydantic import ValidationError

from kotonoha._config import load_settings
from kotonoha.prompts._translate import SRC_MARKER, build_translate_messages
from kotonoha.services._llm_server import (
    TranslationRequest,
    VllmRuntimeBindings,
    VllmTranslationBackend,
    _engine_arguments,
)


class FakeSamplingParameters:
    __slots__: ClassVar[tuple[str, ...]] = ("values",)

    def __init__(
        self,
        /,
        **values: Any,
    ) -> None:
        self.values = values


class FakeTokenizer:
    __slots__: ClassVar[tuple[str, ...]] = ("messages",)

    def __init__(
        self,
        /,
    ) -> None:
        self.messages: list[dict[str, Any]] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        /,
        **options: Any,
    ) -> str:
        assert options == {"tokenize": False, "add_generation_prompt": True}
        self.messages = messages
        return "rendered prompt"


class FakeEngine:
    __slots__: ClassVar[tuple[str, ...]] = ("aborted", "parameters", "prompt")

    def __init__(
        self,
        /,
    ) -> None:
        self.aborted: list[str] = []
        self.parameters: FakeSamplingParameters | None = None
        self.prompt: str | None = None

    async def generate(
        self,
        prompt: str,
        parameters: FakeSamplingParameters,
        request_id: str,
        /,
    ) -> Any:
        del request_id
        self.prompt = prompt
        self.parameters = parameters
        yield SimpleNamespace(
            outputs=[SimpleNamespace(text="Hello", token_ids=[1])],
        )
        yield SimpleNamespace(
            outputs=[SimpleNamespace(text="Hello world", token_ids=[1, 2])],
        )

    async def abort(
        self,
        request_id: str,
        /,
    ) -> None:
        self.aborted.append(request_id)


class FakeWebSocket:
    __slots__: ClassVar[tuple[str, ...]] = ("accepted", "events", "sent")

    def __init__(
        self,
        request: dict[str, Any],
        /,
    ) -> None:
        self.accepted = False
        self.events = [request]
        self.sent: list[dict[str, Any]] = []

    async def accept(
        self,
        /,
    ) -> None:
        self.accepted = True

    async def send_json(
        self,
        event: dict[str, Any],
        /,
    ) -> None:
        self.sent.append(event)

    async def receive_json(
        self,
        /,
    ) -> dict[str, Any]:
        if self.events:
            return self.events.pop(0)
        raise WebSocketDisconnect()


def _backend() -> VllmTranslationBackend:
    backend = object.__new__(VllmTranslationBackend)
    backend.config = load_settings().llm
    backend.bindings = VllmRuntimeBindings(
        engine_arguments_type=object,
        engine_context_type=object,
        sampling_parameters_type=FakeSamplingParameters,
    )
    backend.engine = FakeEngine()
    backend.engine_context = None
    backend.error = None
    backend.load_seconds = 0.0
    backend.tokenizer = FakeTokenizer()
    return backend


def test_translate_messages_follow_translategemma_content_contract() -> None:
    messages = build_translate_messages(
        ["안녕하세요", "안녕 하세요"],
        source_language="ko",
        target_language="en",
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert content[0]["source_lang_code"] == "ko"
    assert content[0]["target_lang_code"] == "en"
    assert "1. 안녕하세요" in content[0]["text"]
    assert SRC_MARKER in content[0]["text"]


def test_engine_arguments_select_local_bfloat16_snapshot() -> None:
    settings = load_settings()
    arguments = _engine_arguments(settings.llm)

    assert arguments["model"].endswith("models/llm/translategemma-4b-it")
    assert arguments["dtype"] == "bfloat16"
    assert settings.accelerator.profile == "nvidia.jetson.agx-orin"
    assert arguments["kv_cache_dtype"] == "fp8"
    assert arguments["max_num_seqs"] == 1
    assert arguments["max_num_batched_tokens"] == 2048
    assert arguments["enable_prefix_caching"] is False
    assert "compilation_config" not in arguments
    assert arguments["limit_mm_per_prompt"] == {"image": 0, "audio": 0, "video": 0}
    assert "quantization" not in arguments
    assert arguments["served_model_name"] == ["kotonoha-translation"]


def test_remote_engine_arguments_select_twelve_billion_parameter_snapshot() -> None:
    settings = load_settings("config/remote-server.yaml")
    arguments = _engine_arguments(settings.llm)

    assert arguments["model"] == "/models/llm/translategemma-12b-it"
    assert arguments["dtype"] == "bfloat16"
    assert settings.accelerator.profile == "nvidia.rtx.a6000"
    assert settings.llm.enforce_eager is False
    assert settings.llm.gpu_memory_utilization == 0.90
    assert "kv_cache_dtype" not in arguments
    assert arguments["max_num_seqs"] == 1
    assert arguments["max_num_batched_tokens"] == 4096
    assert arguments["enable_prefix_caching"] is True
    assert arguments["compilation_config"] == {
        "mode": 2,
        "cudagraph_capture_sizes": [1, 2, 4],
        "cache_dir": "/models/vllm-compile-cache",
    }
    assert "limit_mm_per_prompt" not in arguments
    assert "quantization" not in arguments


def test_translation_request_rejects_openai_system_messages() -> None:
    with pytest.raises(ValidationError):
        TranslationRequest(
            model="kotonoha-translation",
            messages=[{"role": "system", "content": "Translate."}],
        )


async def test_backend_streams_only_new_text() -> None:
    backend = _backend()
    messages = build_translate_messages(
        ["안녕하세요"],
        source_language="ko",
        target_language="en",
    )
    request = TranslationRequest(
        model="kotonoha-translation",
        messages=messages,
        max_tokens=32,
    )

    chunks = [item async for item in backend.stream(request)]

    assert chunks == [("Hello", 1), (" world", 2)]
    assert backend.engine.prompt == "rendered prompt"
    assert backend.engine.parameters.values["max_tokens"] == 32
    assert backend.engine.aborted == []


async def test_websocket_streams_translation_events() -> None:
    backend = _backend()
    messages = build_translate_messages(
        ["안녕하세요"],
        source_language="ko",
        target_language="en",
    )
    websocket = FakeWebSocket(
        {
            "type": "translation.create",
            "model": "kotonoha-translation",
            "messages": messages,
        }
    )

    await backend.handle_websocket(websocket)

    assert websocket.accepted
    assert [event["type"] for event in websocket.sent] == [
        "session.created",
        "translation.delta",
        "translation.delta",
        "translation.done",
    ]
    assert websocket.sent[-1]["text"] == "Hello world"
    assert websocket.sent[-1]["usage"] == {"completion_tokens": 2}
