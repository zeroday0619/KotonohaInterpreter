#!/usr/bin/env python3
"""Spike 3: measure TranslateGemma through the resident WebSocket service."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from kotonoha._config import load_settings
from kotonoha.clients._llm import GenerationStatistics, LanguageModelClient
from kotonoha.prompts._translate import SRC_MARKER, build_translate_messages

MODELS = {
    "jetson": {
        "repo": "google/translategemma-4b-it",
        "directory": "translategemma-4b-it",
    },
    "a6000": {
        "repo": "google/translategemma-12b-it",
        "directory": "translategemma-12b-it",
    },
}
DEFAULT_CONTEXT_LENGTH = 2048
DEFAULT_OUTPUT_TOKENS = 96
MIN_TOKENS_PER_SECOND = 5.0
HYPOTHESES = [
    "다음 주 화요일까지 소프트웨어 정보를 정리해서 보내주시면 감사하겠습니다.",
    "다음 주 화요일까지 소프트웨어 정보를 정리해서 보내주시면 감사하겠습니다",
    "다음 주 화요일까지 소프트웨어 정보를 정리해서 보내 주시면 감사하겠습니다.",
    "다음 주 화요일까지 소프트웨어 정보를 정리해서 보내주시면 감사합니다.",
    "다음 주 화요일까지 소프트웨어 정보를 정리해서 보내주시면 감사하겠습니다요.",
]


def environment_info() -> dict[str, Any]:
    information: dict[str, Any] = {
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "containerized": Path("/.dockerenv").exists(),
        "container_image": os.environ.get("KOTONOHA_SPIKE_IMAGE"),
        "gpu_selection": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        information["torch"] = torch.__version__
        information["cuda"] = torch.version.cuda
        information["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            information["device"] = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            information["capability"] = f"{capability[0]}.{capability[1]}"
            information["memory_total_gib"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3,
                1,
            )
    except Exception as error:  # noqa: BLE001
        information["torch_error"] = repr(error)
    return information


def server_command(
    port: int,
    /,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "kotonoha.services._llm_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--loop",
        "uvloop",
    ]


def server_environment(
    models_directory: Path,
    /,
    *,
    context_length: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "KOTONOHA_CONFIG": "/workspace/config/default.yaml",
            "KOTONOHA__LLM__MODELS_DIR": str(models_directory),
            "KOTONOHA__LLM__MAX_MODEL_LEN": str(context_length),
            "KOTONOHA__LLM__GPU_MEMORY_UTILIZATION": str(gpu_memory_utilization),
            "KOTONOHA__LLM__ENFORCE_EAGER": "1" if enforce_eager else "0",
            "KOTONOHA_SKIP_LOCAL_CONFIG": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    return environment


def health_state(
    port: int,
    /,
) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return json.loads(response.read())
    except (json.JSONDecodeError, urllib.error.URLError, OSError):
        return None


def wait_for_server(
    process: subprocess.Popen[Any],
    /,
    *,
    port: int,
    timeout_seconds: float,
) -> tuple[bool, float, str | None]:
    started_at = time.perf_counter()
    while time.perf_counter() - started_at < timeout_seconds:
        if process.poll() is not None:
            return False, time.perf_counter() - started_at, f"server exited {process.returncode}"
        health = health_state(port)
        if health and health.get("ok"):
            return True, time.perf_counter() - started_at, None
        if health and health.get("error"):
            return False, time.perf_counter() - started_at, str(health["error"])
        time.sleep(2.0)
    return False, time.perf_counter() - started_at, "FastAPI service health timeout"


async def measure_once(
    port: int,
    /,
    *,
    output_tokens: int,
) -> dict[str, Any]:
    settings = load_settings()
    client = LanguageModelClient(f"http://127.0.0.1:{port}", settings.llm)
    messages = build_translate_messages(
        HYPOTHESES,
        source_language="ko",
        target_language="en",
    )
    statistics = GenerationStatistics()
    output_parts: list[str] = []
    try:
        async for delta in client.stream_chat(
            messages,
            statistics,
            max_tokens=output_tokens,
        ):
            output_parts.append(delta)
    finally:
        await client.aclose()
    output = "".join(output_parts)
    return {
        "ttft_ms": statistics.time_to_first_token_ms,
        "tokens": statistics.token_count,
        "total_ms": round((statistics.finished_at - statistics.started_at) * 1000, 1),
        "tok_per_s": statistics.tokens_per_second,
        "one_pass_marker": SRC_MARKER in output,
        "output": output[:600],
    }


def verdict(
    measurement: dict[str, Any] | None,
    /,
) -> dict[str, Any]:
    token_rate = (measurement or {}).get("tok_per_s")
    marker = bool((measurement or {}).get("one_pass_marker"))
    accepted = token_rate is not None and token_rate >= MIN_TOKENS_PER_SECOND and marker
    if token_rate is None:
        note = "TranslateGemma WebSocket 생성 속도를 측정하지 못했다. 로그를 확인할 것."
    elif token_rate < MIN_TOKENS_PER_SECOND:
        note = (
            f"TranslateGemma가 {token_rate} tok/s로 {MIN_TOKENS_PER_SECOND} tok/s 기준 미달이다."
        )
    elif not marker:
        note = "번역은 생성됐지만 교정 원문 마커가 없어 1회 패스 계약을 확인하지 못했다."
    else:
        note = "TranslateGemma WebSocket 스트림이 속도와 1회 교정-번역 출력 계약을 충족했다."
    return {
        "llm_profile": "translategemma" if accepted else "none",
        "tok_per_s": token_rate,
        "threshold": MIN_TOKENS_PER_SECOND,
        "one_pass_marker": marker,
        "ok": accepted,
        "note": note,
    }


async def benchmark(
    arguments: argparse.Namespace,
    /,
) -> dict[str, Any]:
    model_specification = MODELS[arguments.target]
    model = arguments.models_dir / model_specification["directory"]
    result: dict[str, Any] = {
        "spike": 3,
        "target": arguments.target,
        "question": (
            f"{model_specification['directory']}가 vLLM WebSocket으로 "
            "교정과 번역을 스트리밍하는가"
        ),
        "env": environment_info(),
        "model": {
            **model_specification,
            "path": str(model),
        },
        "conditions": {
            "max_model_len": arguments.context,
            "batch": 1,
            "output_tokens": arguments.output_tokens,
            "runs": arguments.runs,
            "gpu_memory_utilization": arguments.gpu_memory_utilization,
            "enforce_eager": not arguments.no_enforce_eager,
            "transport": "FastAPI /v1/realtime WebSocket",
        },
    }
    if not (model / "config.json").is_file():
        result["error"] = f"TranslateGemma snapshot is incomplete: {model}"
        result["verdict"] = verdict(None)
        return result

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = arguments.out.with_name("spike3-vllm.log")
    result["log"] = str(log_path)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            server_command(arguments.port),
            env=server_environment(
                arguments.models_dir,
                context_length=arguments.context,
                gpu_memory_utilization=arguments.gpu_memory_utilization,
                enforce_eager=not arguments.no_enforce_eager,
            ),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            ready, load_seconds, error = wait_for_server(
                process,
                port=arguments.port,
                timeout_seconds=arguments.startup_timeout,
            )
            result["server_load_s"] = round(load_seconds, 2)
            if not ready:
                result["error"] = error
                result["verdict"] = verdict(None)
                return result
            await measure_once(arguments.port, output_tokens=arguments.output_tokens)
            measurements = [
                await measure_once(arguments.port, output_tokens=arguments.output_tokens)
                for _ in range(arguments.runs)
            ]
            best = max(measurements, key=lambda item: item.get("tok_per_s") or 0.0)
            result["server"] = {
                "backend": "vllm_in_process",
                "endpoint": "/v1/realtime",
                "runs": measurements,
                "best": best,
                "health": health_state(arguments.port),
            }
            result["verdict"] = verdict(best)
        except Exception as error:  # noqa: BLE001
            result["error"] = repr(error)
            result["traceback"] = traceback.format_exc()
            result["verdict"] = verdict(None)
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
    return result


def main() -> int:
    os.environ["KOTONOHA_SKIP_LOCAL_CONFIG"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["jetson", "a6000"], default="jetson")
    parser.add_argument("--models-dir", type=Path, default=Path("./models/llm"))
    parser.add_argument("--port", type=int, default=18003)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--no-enforce-eager", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("spikes/out/spike3.json"))
    arguments = parser.parse_args()

    result = asyncio.run(benchmark(arguments))
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
