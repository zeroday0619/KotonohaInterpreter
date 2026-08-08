"""Configuration path boundary validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from kotonoha._config import (
    AsrConfig,
    LanguageModelConfig,
    LanguageModelProfile,
    LatencyBudgetConfig,
    RemoteConfig,
    ServiceEndpointsConfig,
    SessionConfig,
    SharedMemoryConfig,
    VadConfig,
    accelerator_profile_path,
    load_settings,
)
from kotonoha._env import load_env_file, parse_env_file


@pytest.mark.parametrize(
    "profile",
    (
        "../../outside.family.model",
        "nvidia/../../outside.family.model",
        "nvidia.family./absolute",
        "NVIDIA.jetson.agx-orin",
    ),
)
def test_accelerator_profile_rejects_path_and_case_escape(
    _positional_only: object | None = None,
    /,
    *,
    profile: str,
) -> None:
    with pytest.raises(ValueError, match="accelerator profile"):
        accelerator_profile_path(profile)


def test_asr_model_revision_requires_an_immutable_commit() -> None:
    with pytest.raises(ValidationError):
        AsrConfig(model_revision="main")


def test_accuracy_and_allocation_limits_reject_unsafe_configuration() -> None:
    with pytest.raises(ValidationError):
        AsrConfig(n_best=1)
    with pytest.raises(ValidationError, match="duplicate"):
        SessionConfig(pair=["ko", "ko"])
    with pytest.raises(ValidationError):
        SharedMemoryConfig(slots=1024)
    with pytest.raises(ValidationError, match="neg_threshold"):
        VadConfig(threshold=0.3, neg_threshold=0.5)
    with pytest.raises(ValidationError):
        RemoteConfig(failover_after=0)
    with pytest.raises(ValidationError):
        RemoteConfig(token="short")


@pytest.mark.parametrize(
    "endpoint",
    (
        "file:///etc/passwd",
        "http://user:password@host:8001",
        "http://host:0",
        "http://host:8001?token=secret",
        "not-a-url",
    ),
)
def test_service_endpoints_reject_unsafe_or_malformed_urls(
    _positional_only: object | None = None,
    /,
    *,
    endpoint: str,
) -> None:
    with pytest.raises(ValidationError):
        ServiceEndpointsConfig(asr=endpoint)


def test_model_directory_and_engine_limits_reject_resource_escape() -> None:
    with pytest.raises(ValidationError):
        LanguageModelProfile(
            repo="example/model",
            directory="../../outside",
        )

    settings = load_settings()
    values = settings.llm.model_dump()
    values["compilation_cudagraph_capture_sizes"] = [1, -1]
    with pytest.raises(ValidationError):
        LanguageModelConfig(**values)

    values = settings.llm.model_dump()
    values["profiles"] = {"unused": values["profiles"]["translategemma"]}
    with pytest.raises(ValidationError, match="active profile"):
        LanguageModelConfig(**values)

    values = settings.llm.model_dump()
    values["max_tokens"] = 2049
    with pytest.raises(ValidationError):
        LanguageModelConfig(**values)


def test_latency_budget_rejects_a_negative_post_silence_target() -> None:
    with pytest.raises(ValidationError, match="total"):
        LatencyBudgetConfig(silence=800, total=799)


def test_unknown_placement_role_fails_during_configuration_load() -> None:
    settings = load_settings()
    values = settings.model_dump()
    values["placement"] = {"unknown": "remote"}

    with pytest.raises(ValidationError, match="unknown role"):
        type(settings)(**values)


def test_shared_memory_slot_must_hold_the_maximum_utterance() -> None:
    settings = load_settings()
    settings.shm.slot_seconds = 1

    with pytest.raises(ValidationError, match="slot_seconds"):
        type(settings)(**settings.model_dump())


def test_model_fetches_pin_every_external_snapshot() -> None:
    source = Path("scripts/fetch_models.sh").read_text(encoding="utf-8")
    revisions = (
        "bfdc0193023f121ea5b3cc7b176dbed570a68a59",
        "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
        "2769294da9567371363522aac9bbcfdd19447add",
        "85e237c12c027371202489a0ec509ded67b5e4b5",
        "edaa852ec7e145841d8ffdb056a99866b5f0a478",
        "10042cb0e6e7fdce748996a71dc3dc432a4e0c89",
        "d1b225e1caa17f1ddc7e62065d8637d0923f34e2",
    )

    assert all(revision in source for revision in revisions)
    assert source.count("--revision") == 7
    assert "/master/" not in source


def test_environment_file_parser_supports_documented_syntax() -> None:
    values = parse_env_file(
        """
        # comment
        export KOTONOHA_LANG=ja
        KOTONOHA_CONFIG="config/performance.yaml"
        INVALID LINE
        """
    )

    assert values == {
        "KOTONOHA_LANG": "ja",
        "KOTONOHA_CONFIG": "config/performance.yaml",
    }


def test_environment_file_preserves_process_values_and_rejects_foreign_names(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment_path = tmp_path / ".env"
    environment_path.write_text(
        "KOTONOHA_LANG=ja\nPATH=/untrusted\nMODELS_DIR=/models\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KOTONOHA_LANG", "ko")
    original_path = os.environ["PATH"]

    applied = load_env_file(environment_path)

    assert applied == {}
    assert os.environ["KOTONOHA_LANG"] == "ko"
    assert os.environ["PATH"] == original_path
    assert "MODELS_DIR" not in applied
