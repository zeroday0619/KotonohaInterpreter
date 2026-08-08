"""Cross-verification ASR target configuration contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx2
import numpy as np
import pytest

from kotonoha._config import DEFAULT_CONFIG, AsrVerificationConfig, load_settings, read_yaml
from kotonoha.services._asr_verify_server import (
    MAXIMUM_WHISPER_CPP_RESPONSE_BYTES,
    STATE,
    VerificationRequest,
    WhisperCppBackend,
    health,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_jetson_verification_defaults_to_cpu_int8() -> None:
    typed_config = AsrVerificationConfig()
    default_config = read_yaml(DEFAULT_CONFIG)["asr_verify"]

    assert typed_config.device == "cpu"
    assert typed_config.compute_type == "int8"
    assert default_config["device"] == typed_config.device
    assert default_config["compute_type"] == typed_config.compute_type


def test_remote_verification_retains_cuda_float16() -> None:
    remote_config = load_settings("config/remote-server.yaml").asr_verify

    assert remote_config.device == "cuda"
    assert remote_config.compute_type == "float16"


def test_jetson_verification_image_checks_cpu_int8_capability() -> None:
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.asr-verify").read_text(
        encoding="utf-8"
    )

    assert "ctranslate2.get_supported_compute_types('cpu')" in dockerfile
    assert "'int8' in compute_types" in dockerfile
    assert dockerfile.count("--extra asr-verify") == 2
    assert "uv pip install" not in dockerfile


def test_verification_health_reports_the_effective_runtime(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        STATE,
        "backend",
        SimpleNamespace(name="faster_whisper", device="cpu", compute_type="int8"),
    )
    monkeypatch.setitem(STATE, "error", None)

    response = health()

    assert response["ok"] is True
    assert response["device"] == "cpu"
    assert response["compute_type"] == "int8"
    assert response["resources"]["system"]["kernel"]["release"]


def test_whisper_cpp_response_is_bounded() -> None:
    def oversized_response(
        request: httpx2.Request,
        /,
    ) -> httpx2.Response:
        del request
        return httpx2.Response(
            200,
            content=b"x" * (MAXIMUM_WHISPER_CPP_RESPONSE_BYTES + 1),
        )

    backend = object.__new__(WhisperCppBackend)
    backend.url = "http://whisper.test"
    backend.client = httpx2.Client(transport=httpx2.MockTransport(oversized_response))
    try:
        with pytest.raises(RuntimeError, match="response exceeded"):
            backend.transcribe(
                np.zeros(160, dtype=np.float32),
                VerificationRequest(),
            )
    finally:
        backend.shutdown()
