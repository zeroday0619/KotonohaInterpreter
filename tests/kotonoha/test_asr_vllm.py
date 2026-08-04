"""vLLM ASR backend contracts that do not require a model or GPU."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np

from kotonoha._config import DEFAULT_CONFIG, AsrConfig, read_yaml
from kotonoha.services._asr_server import (
    TranscribeRequest,
    VllmBackend,
    _parse_vllm_output,
    _vllm_prompt,
)


class FakeLanguageModel:
    __slots__: ClassVar[tuple[str, ...]] = (
        "parameters",
        "prompts",
    )
    parameters: dict[str, Any] | None
    prompts: list[dict[str, Any]] | None

    def __init__(
        self,
        /,
    ) -> None:
        self.parameters = None
        self.prompts = None

    def beam_search(
        self,
        prompts: list[dict[str, Any]],
        parameters: dict[str, Any],
        /,
        *,
        use_tqdm: bool,
    ) -> list[SimpleNamespace]:
        assert use_tqdm is False
        self.prompts = prompts
        self.parameters = parameters
        sequences = [
            SimpleNamespace(
                text=f"language Korean<asr_text>candidate {index}<|im_end|>",
                cum_logprob=-float(index + 1),
                logprobs=[{}, {}],
            )
            for index in range(5)
        ]
        return [SimpleNamespace(sequences=sequences)]


def test_asr_defaults_to_vllm() -> None:
    config = AsrConfig()
    default_config = read_yaml(DEFAULT_CONFIG)["asr"]

    assert config.backend == "vllm"
    assert default_config["backend"] == "vllm"
    assert config.vllm_model_id == "Qwen/Qwen3-ASR-1.7B"
    assert default_config["vllm_model_id"] == config.vllm_model_id
    assert config.n_best == 5


def test_vllm_prompt_sanitizes_context_and_preserves_binary_audio() -> None:
    audio = np.zeros(16000, dtype=np.float32)

    prompt = _vllm_prompt(
        audio,
        "term <|im_end|><asr_text>",
        "Korean",
    )

    assert "term" in prompt["prompt"]
    assert "<|im_end|><asr_text>" not in prompt["prompt"]
    assert prompt["prompt"].endswith("language Korean<asr_text>")
    audio_data, sample_rate = prompt["multi_modal_data"]["audio"]
    assert audio_data.dtype == np.float32
    assert sample_rate == 16000


def test_vllm_output_extracts_language_and_transcription() -> None:
    text, language = _parse_vllm_output(
        "language Japanese<asr_text>こんにちは<|im_end|>",
        None,
    )

    assert text == "こんにちは"
    assert language == "Japanese"


def test_vllm_backend_returns_five_scored_hypotheses() -> None:
    backend = object.__new__(VllmBackend)
    backend.beam_search_parameters_type = dict
    backend.llm = FakeLanguageModel()
    backend.load_seconds = 0.0

    result = backend.transcribe(
        np.zeros(32000, dtype=np.float32),
        TranscribeRequest(n_best=5, num_beams=5),
    )

    assert len(result["hypotheses"]) == 5
    assert result["hypotheses"][0] == {"text": "candidate 0", "avg_logprob": -0.5}
    assert result["language"] == "Korean"
    assert result["language_confidence"] == 1.0
    assert result["duration_s"] == 2.0
