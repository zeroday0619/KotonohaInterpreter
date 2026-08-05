#!/usr/bin/env python3
"""Spike 3: measure vLLM translation throughput for MoE and dense profiles."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROFILES = {
    "moe": {
        "repo": "ELVISIO/Qwen3-30B-A3B-Instruct-2507-AWQ",
        "directory": "Qwen3-30B-A3B-Instruct-2507-AWQ",
        "note": "30B MoE, 3B active, AWQ 4-bit",
    },
    "dense": {
        "repo": "Qwen/Qwen3-14B-AWQ",
        "directory": "Qwen3-14B-AWQ",
        "note": "dense 14B, AWQ 4-bit",
    },
}

DEFAULT_CONTEXT_LENGTH = 2048
DEFAULT_OUTPUT_TOKENS = 60
MIN_TOKENS_PER_SECOND = 5.0
SYSTEM_PROMPT = (
    "You are a professional consecutive interpreter. Reconstruct what the speaker said "
    "from the ASR hypotheses, then translate it into English. Output the translation only."
)
USER_PROMPT = """## Conversation so far
[한국어] 오늘 회의는 세 시에 시작합니다.
[English] Today's meeting starts at three.
[한국어] 자료는 미리 공유해 주세요.
[English] Please share the materials in advance.

## Glossary (apply verbatim)
- 소프트웨어 → software
- 정보 → information

## ASR hypotheses (Korean), best first
1. 다음 주 화요일까지 소프트웨어 정보를 정리해서 보내주시면 감사하겠습니다.
2. 다음 주 화요일까지 소프트웨어 정보를 정리해서 보내주시면 감사하겠습니다
3. 다음 주 화요일까지 소프트웨어 정보를 정리해서 보내 주시면 감사하겠습니다.
4. 다음 주 화요일까지 소프트웨어 정보를 정리해서 보내주시면 감사합니다.
5. 다음 주 화요일까지 소프트웨어 정보를 정리해서 보내주시면 감사하겠습니다요.

Now output the English translation."""


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
    except Exception as error:  # noqa: BLE001
        information["torch_error"] = repr(error)
    return information


def _request_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("KOTONOHA_SERVICE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def wait_for_health(
    url: str,
    /,
    timeout_seconds: float,
) -> bool:
    deadline = time.time() + timeout_seconds
    request = urllib.request.Request(f"{url}/health", headers=_request_headers())
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2.0)
    return False


def stream_translation(
    url: str,
    /,
    *,
    served_model_name: str,
    output_tokens: int,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": served_model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers=_request_headers(),
    )
    started_at = time.perf_counter()
    first_token_at: float | None = None
    completion_tokens = 0
    output_parts: list[str] = []
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = event.get("usage") or {}
            completion_tokens = usage.get("completion_tokens", completion_tokens)
            choices = event.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if not content:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            output_parts.append(content)

    finished_at = time.perf_counter()
    generation_seconds = finished_at - (first_token_at or finished_at)
    return {
        "ttft_ms": round(((first_token_at or started_at) - started_at) * 1000, 1),
        "tokens": completion_tokens,
        "total_ms": round((finished_at - started_at) * 1000, 1),
        "tok_per_s": (
            round((completion_tokens - 1) / generation_seconds, 2)
            if completion_tokens > 1 and generation_seconds > 0
            else None
        ),
        "output": "".join(output_parts)[:400],
    }


def benchmark_profile(
    name: str,
    /,
    *,
    vllm_command: str,
    models_directory: Path,
    port: int,
    context_length: int,
    output_tokens: int,
    runs: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
) -> dict[str, Any]:
    specification = PROFILES[name]
    model = models_directory / specification["directory"]
    result: dict[str, Any] = {"profile": name, "model": str(model), **specification}
    if not (model / "config.json").is_file():
        return {
            **result,
            "error": f"vLLM model snapshot is incomplete: {model}",
        }

    served_model_name = f"kotonoha-spike-{name}"
    command = [
        vllm_command,
        "serve",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--max-model-len",
        str(context_length),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        "1",
        "--dtype",
        "half",
        "--quantization",
        "awq",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
        "--enforce-eager" if enforce_eager else "--no-enforce-eager",
    ]
    token = os.environ.get("KOTONOHA_SERVICE_TOKEN")
    if token:
        command.extend(("--api-key", token))

    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        load_started_at = time.perf_counter()
        if not wait_for_health(url, 600):
            result["server"] = {"error": "vLLM health timeout"}
            return result
        result["server_load_s"] = round(time.perf_counter() - load_started_at, 1)
        stream_translation(
            url,
            served_model_name=served_model_name,
            output_tokens=output_tokens,
        )
        measurements = [
            stream_translation(
                url,
                served_model_name=served_model_name,
                output_tokens=output_tokens,
            )
            for _ in range(runs)
        ]
        best = max(measurements, key=lambda measurement: measurement["tok_per_s"] or 0)
        result["server"] = {"runs": measurements, "best": best}
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    return result


def verdict(
    results: dict[str, Any],
    /,
) -> dict[str, Any]:
    def rate(
        result: dict[str, Any],
        /,
    ) -> float | None:
        return ((result.get("server") or {}).get("best") or {}).get("tok_per_s")

    moe_rate = rate(results.get("moe", {}))
    dense_rate = rate(results.get("dense", {}))
    lines = []
    if moe_rate is not None:
        lines.append(f"MoE 30B-A3B: {moe_rate} tok/s")
    if dense_rate is not None:
        lines.append(f"밀집 14B: {dense_rate} tok/s")

    if moe_rate is not None and moe_rate >= MIN_TOKENS_PER_SECOND and (
        dense_rate is None or moe_rate >= dense_rate
    ):
        choice = "moe"
        reason = "MoE 가 5 tok/s 이상이고 밀집보다 느리지 않다. 품질 비교 후 채택."
    elif dense_rate is not None and dense_rate >= MIN_TOKENS_PER_SECOND:
        choice = "dense"
        reason = f"MoE 가 기준 미달({moe_rate}) 또는 더 느림. 명세대로 밀집 14B 로 회귀."
    else:
        choice = "none"
        reason = (
            f"둘 다 {MIN_TOKENS_PER_SECOND} tok/s 미만 "
            f"(moe={moe_rate}, dense={dense_rate}). 절 단위 스트리밍 조건을 충족하지 못함."
        )
    return {
        "llm_profile": choice,
        "moe_tok_per_s": moe_rate,
        "dense_tok_per_s": dense_rate,
        "threshold": MIN_TOKENS_PER_SECOND,
        "note": " / ".join(lines) + " — " + reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["jetson", "a6000"], default="jetson")
    parser.add_argument("--vllm-command", default="vllm")
    parser.add_argument("--models-dir", type=Path, default=Path("./models/llm"))
    parser.add_argument("--port", type=int, default=18003)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--no-enforce-eager", action="store_true")
    parser.add_argument("--only", choices=list(PROFILES), default=None)
    parser.add_argument("--out", type=Path, default=Path("spikes/out/spike3.json"))
    arguments = parser.parse_args()

    result: dict[str, Any] = {
        "spike": 3,
        "target": arguments.target,
        "question": "MoE(활성 3B) vs 밀집 14B 실측 tok/s",
        "env": environment_info(),
        "conditions": {
            "max_model_len": arguments.context,
            "batch": 1,
            "output_tokens": arguments.output_tokens,
            "runs": arguments.runs,
            "gpu_memory_utilization": arguments.gpu_memory_utilization,
            "enforce_eager": not arguments.no_enforce_eager,
        },
    }
    for name in PROFILES:
        if arguments.only and name != arguments.only:
            continue
        print(f"── {name} 측정 중 …", file=sys.stderr)
        result[name] = benchmark_profile(
            name,
            vllm_command=arguments.vllm_command,
            models_directory=arguments.models_dir,
            port=arguments.port,
            context_length=arguments.context,
            output_tokens=arguments.output_tokens,
            runs=arguments.runs,
            gpu_memory_utilization=arguments.gpu_memory_utilization,
            enforce_eager=not arguments.no_enforce_eager,
        )
    result["verdict"] = verdict(result)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
