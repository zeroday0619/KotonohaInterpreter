#!/usr/bin/env python3
"""Spike 2 — validate vLLM-Omni Qwen3-TTS on the selected GPU.

The probe executes a FlashAttention CUDA kernel, starts the same vLLM-Omni server used
by deployment, and measures raw 24 kHz PCM streaming through ``/v1/audio/speech``.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import BinaryIO

PROBE_TEXT = {
    "Korean": "안녕하세요, 지금 통역기 성능을 확인하고 있습니다.",
    "English": "Hello, this is a latency probe for the interpreter.",
    "Japanese": "こんにちは、通訳機の性能を確認しています。",
    "Chinese": "您好，我们正在确认口译设备的性能。",
}
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
SERVED_MODEL_NAME = "kotonoha-tts"


def environment_info() -> dict:
    information: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "image": os.environ.get("KOTONOHA_SPIKE_IMAGE"),
    }
    try:
        import torch

        information.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "capability": (
                    ".".join(map(str, torch.cuda.get_device_capability(0)))
                    if torch.cuda.is_available()
                    else None
                ),
                "memory_total_gib": (
                    round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except Exception as error:  # noqa: BLE001
        information["torch_error"] = repr(error)
    try:
        from importlib.metadata import version

        information["vllm_omni"] = version("vllm-omni")
        information["vllm"] = version("vllm")
    except Exception as error:  # noqa: BLE001
        information["version_error"] = repr(error)
    return information


def probe_flash_attention() -> dict:
    result: dict = {"import": False, "kernel_ok": False}
    try:
        import torch
        from flash_attn import flash_attn_func

        result["import"] = True
        query = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.float16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        torch.cuda.synchronize()
        started_at = time.perf_counter()
        output = flash_attn_func(query, key, value, causal=False)
        torch.cuda.synchronize()
        result.update(
            {
                "kernel_ok": bool(torch.isfinite(output).all().item()),
                "kernel_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "shape": list(output.shape),
            }
        )
    except Exception as error:  # noqa: BLE001
        result["error"] = repr(error)
    return result


def gpu_memory_usage_mib() -> float | None:
    try:
        import torch

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return round((total_bytes - free_bytes) / 2**20, 1)
    except Exception:  # noqa: BLE001
        return None


def server_command(
    model: str,
    /,
    *,
    port: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
) -> list[str]:
    command = [
        "vllm",
        "serve",
        model,
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--stage-overrides",
        '{"1":{"max_num_seqs":1}}',
    ]
    command.append("--enforce-eager" if enforce_eager else "--no-enforce-eager")
    return command


def wait_for_server(
    process: subprocess.Popen[bytes],
    /,
    *,
    port: int,
    timeout_seconds: float,
) -> tuple[bool, float, str | None]:
    started_at = time.perf_counter()
    deadline = started_at + timeout_seconds
    last_error: str | None = None
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            return False, round(time.perf_counter() - started_at, 2), (
                f"vLLM-Omni exited with status {process.returncode}"
            )
        try:
            with urllib.request.urlopen(  # noqa: S310
                f"http://127.0.0.1:{port}/health",
                timeout=2.0,
            ) as response:
                if response.status == 200:
                    return True, round(time.perf_counter() - started_at, 2), None
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = repr(error)
        time.sleep(2.0)
    return False, round(time.perf_counter() - started_at, 2), last_error or "startup timeout"


def request_speech(
    text: str,
    /,
    *,
    language: str,
    port: int,
    timeout_seconds: float,
) -> dict:
    payload = json.dumps(
        {
            "input": text,
            "model": SERVED_MODEL_NAME,
            "voice": "vivian",
            "language": language,
            "task_type": "CustomVoice",
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
        }
    ).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_seconds)
    started_at = time.perf_counter()
    first_audio_at: float | None = None
    audio = bytearray()
    try:
        connection.request(
            "POST",
            "/v1/audio/speech",
            body=payload,
            headers={"content-type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read().decode(errors="replace")
            raise RuntimeError(f"Speech API returned {response.status}: {detail[:2000]}")
        while True:
            chunk = response.read1(65536)
            if not chunk:
                break
            if first_audio_at is None:
                first_audio_at = time.perf_counter()
            audio.extend(chunk)
    finally:
        connection.close()
    finished_at = time.perf_counter()
    if not audio or len(audio) % SAMPLE_WIDTH:
        raise RuntimeError(f"invalid raw PCM byte count: {len(audio)}")
    audio_seconds = len(audio) / (SAMPLE_RATE * SAMPLE_WIDTH)
    return {
        "bytes": len(audio),
        "samples": len(audio) // SAMPLE_WIDTH,
        "audio_seconds": round(audio_seconds, 3),
        "ttfa_ms": (
            round((first_audio_at - started_at) * 1000, 1)
            if first_audio_at is not None
            else None
        ),
        "e2e_ms": round((finished_at - started_at) * 1000, 1),
        "rtf": round((finished_at - started_at) / audio_seconds, 3),
    }


def terminate_server(
    process: subprocess.Popen[bytes],
    /,
) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_vllm_omni(
    model: str,
    /,
    *,
    port: int,
    runs: int,
    startup_timeout_seconds: float,
    request_timeout_seconds: float,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    log_path: Path,
) -> dict:
    result: dict = {
        "backend": "vllm_omni",
        "model": model,
        "served_model_name": SERVED_MODEL_NAME,
        "loaded": False,
        "languages": {},
        "log": str(log_path),
        "gpu_memory_samples_mib": [],
    }
    command = server_command(
        model,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )
    result["command"] = command
    baseline_memory = gpu_memory_usage_mib()
    if baseline_memory is not None:
        result["gpu_memory_samples_mib"].append(baseline_memory)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file: BinaryIO
    with log_path.open("wb") as log_file:
        try:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as error:  # noqa: BLE001
            result["error"] = f"server start: {error!r}"
            return result
        try:
            ready, startup_seconds, startup_error = wait_for_server(
                process,
                port=port,
                timeout_seconds=startup_timeout_seconds,
            )
            result["startup_s"] = startup_seconds
            if not ready:
                result["error"] = startup_error
                return result
            result["loaded"] = True
            loaded_memory = gpu_memory_usage_mib()
            if loaded_memory is not None:
                result["gpu_memory_samples_mib"].append(loaded_memory)
            for language, text in PROBE_TEXT.items():
                measurements = []
                try:
                    warmup = request_speech(
                        text,
                        language=language,
                        port=port,
                        timeout_seconds=request_timeout_seconds,
                    )
                    for _ in range(runs):
                        measurements.append(
                            request_speech(
                                text,
                                language=language,
                                port=port,
                                timeout_seconds=request_timeout_seconds,
                            )
                        )
                        current_memory = gpu_memory_usage_mib()
                        if current_memory is not None:
                            result["gpu_memory_samples_mib"].append(current_memory)
                    result["languages"][language] = {
                        "ok": True,
                        "warmup": warmup,
                        "runs": measurements,
                        "median_ttfa_ms": round(
                            statistics.median(
                                measurement["ttfa_ms"] for measurement in measurements
                            ),
                            1,
                        ),
                        "median_e2e_ms": round(
                            statistics.median(
                                measurement["e2e_ms"] for measurement in measurements
                            ),
                            1,
                        ),
                        "median_rtf": round(
                            statistics.median(
                                measurement["rtf"] for measurement in measurements
                            ),
                            3,
                        ),
                    }
                except Exception as error:  # noqa: BLE001
                    result["languages"][language] = {
                        "ok": False,
                        "error": repr(error),
                        "runs": measurements,
                    }
        finally:
            terminate_server(process)
    result["synthesized"] = bool(result["languages"]) and all(
        measurement.get("ok") for measurement in result["languages"].values()
    )
    memory_samples = result["gpu_memory_samples_mib"]
    if memory_samples:
        result["gpu_memory_peak_mib"] = max(memory_samples)
        result["gpu_memory_delta_mib"] = round(max(memory_samples) - memory_samples[0], 1)
    return result


def verdict(
    omni: dict,
    /,
) -> dict:
    korean = omni.get("languages", {}).get("Korean", {})
    ready = bool(omni.get("loaded") and omni.get("synthesized"))
    return {
        "tts_backend": "vllm_omni" if ready else "none",
        "ok": ready,
        "first_packet_ms_estimate": korean.get("median_ttfa_ms"),
        "note": (
            "vLLM-Omni Speech API 로 Qwen3-TTS 4개 언어 PCM 스트리밍에 성공했다."
            if ready
            else "vLLM-Omni 시작 또는 Qwen3-TTS PCM 스트리밍에 실패했다. 로그를 확인할 것."
        ),
    }


def main() -> int:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--target", choices=["jetson", "a6000"], default="jetson")
    argument_parser.add_argument("--runs", type=int, default=3)
    argument_parser.add_argument("--model", default="/models/Qwen3-TTS-0.6B")
    argument_parser.add_argument("--port", type=int, default=18004)
    argument_parser.add_argument("--startup-timeout", type=float, default=600.0)
    argument_parser.add_argument("--request-timeout", type=float, default=120.0)
    argument_parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    argument_parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    argument_parser.add_argument("--out", type=Path, default=Path("spikes/out/spike2.json"))
    argument_parser.add_argument(
        "--log",
        type=Path,
        default=Path("spikes/out/spike2-vllm-omni.log"),
    )
    arguments = argument_parser.parse_args()

    result = {
        "spike": 2,
        "target": arguments.target,
        "question": f"{arguments.target} 에서 vLLM-Omni Qwen3-TTS 가 동작하는가",
        "env": environment_info(),
        "conditions": {
            "runs": arguments.runs,
            "sample_rate": SAMPLE_RATE,
            "response_format": "pcm",
            "stream_format": "audio",
            "gpu_memory_utilization": arguments.gpu_memory_utilization,
            "enforce_eager": arguments.enforce_eager,
        },
        "flash_attn": probe_flash_attention(),
    }
    result["vllm_omni"] = run_vllm_omni(
        arguments.model,
        port=arguments.port,
        runs=arguments.runs,
        startup_timeout_seconds=arguments.startup_timeout,
        request_timeout_seconds=arguments.request_timeout,
        gpu_memory_utilization=arguments.gpu_memory_utilization,
        enforce_eager=arguments.enforce_eager,
        log_path=arguments.log,
    )
    result["verdict"] = verdict(result["vllm_omni"])

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
