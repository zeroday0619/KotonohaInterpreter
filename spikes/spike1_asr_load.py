#!/usr/bin/env python3
"""Spike 1 — does the Jetson vLLM load Qwen3-ASR?

What this decides:
  · whether vLLM loads it at all
  · how long transcribing six seconds of audio takes
  · whether N-best output is available
  · if it fails, how long the transformers path takes instead (needed to
    recompute the latency budget)

Run this probe through ``bash scripts/manage.sh benchmark <target>``. The management
entry point delegates to the Docker harness, which selects the target vLLM image, mounts
model snapshots read-only, and executes this script inside the container.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

TRANSFORMERS_ID = "Qwen/Qwen3-ASR-1.7B-hf"
VLLM_ID = "Qwen/Qwen3-ASR-1.7B"
N_BEST = 5
TRANSFORMERS_WORKER_MARKER = "KOTONOHA_TRANSFORMERS_RESULT="


def load_wav(
    path: Path,
    /,
    rate: int = 16000,
) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit(f"only 16-bit PCM WAV is supported: {path}")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != rate:
        import soxr

        x = np.asarray(soxr.resample(x, sr, rate), dtype=np.float32)
    return x


def synthetic_6s(
    rate: int = 16000,
    /,
) -> np.ndarray:
    """Dummy audio for timing only, when no real recording is at hand.

    The transcription content is meaningless.
    """
    t = np.arange(int(6 * rate)) / rate
    env = 0.5 * (1 + np.sin(2 * np.pi * 3.1 * t))
    sig = np.sin(2 * np.pi * 180 * t) + 0.4 * np.sin(2 * np.pi * 720 * t)
    return (0.2 * env * sig).astype(np.float32)


def env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "containerized": Path("/.dockerenv").exists(),
        "container_image": os.environ.get("KOTONOHA_SPIKE_IMAGE"),
        "gpu_selection": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
            info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = repr(e)
    return info


# -- the transformers path ------------------------------------------------
def _run_transformers_in_process(
    audio: np.ndarray,
    /,
    runs: int,
    model_identifier: str,
) -> dict:
    out: dict = {"backend": "transformers", "model": model_identifier}
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except Exception as e:  # noqa: BLE001
        return {**out, "loaded": False, "error": f"import: {e!r}"}

    try:
        t0 = time.perf_counter()
        proc = AutoProcessor.from_pretrained(model_identifier)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_identifier, dtype=torch.float16, device_map="auto"
        )
        model.eval()
        out["load_s"] = round(time.perf_counter() - t0, 2)
        out["loaded"] = True
    except Exception as e:  # noqa: BLE001
        return {**out, "loaded": False, "error": f"load: {e!r}"}

    def once(
        n_best: int,
        /,
    ) -> tuple[float, list[str], list[float]]:
        try:
            inputs = proc.apply_transcription_request(audio=audio, sampling_rate=16000)
        except TypeError:
            inputs = proc.apply_transcription_request(audio=audio)
        inputs = inputs.to(model.device, model.dtype)
        t = time.perf_counter()
        with torch.inference_mode():
            g = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                num_beams=max(n_best, 1),
                num_return_sequences=n_best,
                return_dict_in_generate=True,
                output_scores=True,
                early_stopping=True,
            )
        dt = time.perf_counter() - t
        seqs = g.sequences[:, inputs["input_ids"].shape[1] :]
        parsed = proc.decode(seqs, return_format="parsed")
        if isinstance(parsed, dict):
            parsed = [parsed]
        texts = [
            (p.get("transcription") if isinstance(p, dict) else str(p)) or "" for p in parsed
        ]
        scores = (
            [float(s) for s in g.sequences_scores.detach().cpu().tolist()]
            if getattr(g, "sequences_scores", None) is not None
            else []
        )
        return dt, texts, scores

    try:
        once(1)  # warm-up
        greedy = [once(1)[0] for _ in range(runs)]
        nb_dt, nb_texts, nb_scores = once(N_BEST)
        nbest = [nb_dt] + [once(N_BEST)[0] for _ in range(runs - 1)]
        out.update(
            greedy_ms=round(statistics.median(greedy) * 1000, 1),
            nbest_ms=round(statistics.median(nbest) * 1000, 1),
            nbest_count=len(nb_texts),
            nbest_ok=len(nb_texts) == N_BEST,
            has_logprobs=bool(nb_scores),
            sample=nb_texts[:N_BEST],
            scores=[round(s, 4) for s in nb_scores],
        )
    except Exception as e:  # noqa: BLE001
        out["infer_error"] = repr(e)
    return out


def run_transformers(
    audio: np.ndarray,
    /,
    runs: int,
    model_identifier: str,
) -> dict:
    fallback_python = os.environ.get("SPIKE_TRANSFORMERS_PYTHON")
    if not fallback_python or Path(fallback_python).resolve() == Path(sys.executable).resolve():
        return _run_transformers_in_process(audio, runs, model_identifier)

    completed = subprocess.run(
        [
            fallback_python,
            str(Path(__file__).resolve()),
            "--transformers-worker",
            "--runs",
            str(runs),
            "--model",
            model_identifier,
        ],
        input=audio.astype(np.float32, copy=False).tobytes(),
        check=False,
        capture_output=True,
    )
    for line in reversed(completed.stdout.decode("utf-8", errors="replace").splitlines()):
        if line.startswith(TRANSFORMERS_WORKER_MARKER):
            return json.loads(line.removeprefix(TRANSFORMERS_WORKER_MARKER))
    error = completed.stderr.decode("utf-8", errors="replace").strip()
    return {
        "backend": "transformers",
        "model": model_identifier,
        "loaded": False,
        "error": f"worker exited {completed.returncode}: {error[-2000:]}",
    }


def transformers_worker_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformers-worker", action="store_true")
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--model", required=True)
    arguments = parser.parse_args()
    audio = np.frombuffer(sys.stdin.buffer.read(), dtype=np.float32).copy()
    result = _run_transformers_in_process(audio, arguments.runs, arguments.model)
    print(f"{TRANSFORMERS_WORKER_MARKER}{json.dumps(result, ensure_ascii=False)}")
    return 0


# -- the vLLM path ---------------------------------------------------------
def run_vllm(
    audio: np.ndarray,
    /,
    runs: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool,
    model_identifier: str,
) -> dict:
    """First: does vLLM recognise this architecture at all?

    Even if it does, check that N-best (n > 1) and log-probabilities come out.
    Transcription without N-best does not satisfy §5.2, so that path cannot be
    adopted.
    """
    out: dict = {"backend": "vllm", "model": model_identifier}
    try:
        from vllm import LLM
        from vllm.sampling_params import BeamSearchParams
    except Exception as e:  # noqa: BLE001
        return {**out, "loaded": False, "error": f"import: {e!r}"}

    try:
        t0 = time.perf_counter()
        llm = LLM(
            model=model_identifier,
            max_model_len=max_model_len,
            dtype="float16",
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"audio": 1},
            enforce_eager=enforce_eager,
        )
        out["load_s"] = round(time.perf_counter() - t0, 2)
        out["loaded"] = True
    except Exception as e:  # noqa: BLE001
        return {**out, "loaded": False, "error": f"load: {e!r}"}

    try:
        parameters = BeamSearchParams(
            beam_width=N_BEST,
            max_tokens=256,
            temperature=0.0,
            length_penalty=1.0,
        )
        prompt = {
            "prompt": (
                "<|im_start|>user\n"
                "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "multi_modal_data": {"audio": (audio, 16000)},
        }
        llm.beam_search([prompt], parameters, use_tqdm=False)  # warm-up
        durations = []
        for _ in range(runs):
            start_time = time.perf_counter()
            result = llm.beam_search([prompt], parameters, use_tqdm=False)
            durations.append(time.perf_counter() - start_time)
        outputs = result[0].sequences
        out.update(
            nbest_ms=round(statistics.median(durations) * 1000, 1),
            nbest_count=len(outputs),
            nbest_ok=len(outputs) == N_BEST,
            has_logprobs=all(bool(output.logprobs) for output in outputs),
            sample=[output.text for output in outputs],
            scores=[round(float(output.cum_logprob), 4) for output in outputs],
        )
    except Exception as e:  # noqa: BLE001
        out["infer_error"] = repr(e)
    return out


def verdict(
    vllm: dict,
    /,
    tf: dict,
) -> dict:
    v_ok = vllm.get("loaded") and vllm.get("nbest_ok")
    t_ok = tf.get("loaded") and tf.get("nbest_ok")
    if v_ok and t_ok:
        rec = "vllm" if vllm.get("nbest_ms", 1e9) < tf.get("nbest_ms", 1e9) else "transformers"
    elif v_ok:
        rec = "vllm"
    elif t_ok:
        rec = "transformers"
    else:
        rec = "none"

    ms = (vllm if rec == "vllm" else tf).get("nbest_ms") if rec != "none" else None
    return {
        "recommended_backend": rec,
        "nbest_ms": ms,
        "asr_budget_ms": 900,
        "within_budget": (ms is not None and ms <= 900),
        "note": (
            "두 경로 모두 실패. Qwen3-ASR 채택 불가 — 대안 검토 필요."
            if rec == "none"
            else (
                f"{rec} 경로의 N-best5 전사가 {ms:.0f}ms. §6 예산 900ms 대비 "
                + ("여유 있음." if ms and ms <= 900 else "초과 — 지연 예산 재계산 필요.")
            )
        ),
    }


def main() -> int:
    if "--transformers-worker" in sys.argv:
        return transformers_worker_main()

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["jetson", "a6000"], default="jetson")
    ap.add_argument("--wav", type=Path, default=None, help="a real ~6 s recording, 16-bit PCM")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", choices=["vllm", "transformers"], default=None)
    ap.add_argument("--vllm-model", default=VLLM_ID)
    ap.add_argument("--transformers-model", default=TRANSFORMERS_ID)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--out", type=Path, default=Path("spikes/out/spike1.json"))
    a = ap.parse_args()

    if a.wav:
        audio = load_wav(a.wav)
        src = str(a.wav)
    else:
        audio = synthetic_6s()
        src = "synthetic (타이밍 전용, 전사 내용 무의미)"
        print("!! 실 녹음 없이 실행 중. 시간만 신뢰할 것.", file=sys.stderr)

    result = {
        "spike": 1,
        "target": a.target,
        "question": f"{a.target} vLLM 이 Qwen3-ASR 을 로드하는가",
        "audio": {"source": src, "seconds": round(len(audio) / 16000, 2)},
        "env": env_info(),
        "conditions": {
            "gpu_memory_utilization": a.gpu_memory_utilization,
            "max_model_len": a.max_model_len,
            "enforce_eager": a.enforce_eager,
        },
    }
    result["vllm"] = (
        run_vllm(
            audio,
            a.runs,
            a.gpu_memory_utilization,
            a.max_model_len,
            a.enforce_eager,
            a.vllm_model,
        )
        if a.only in (None, "vllm")
        else {"skipped": True}
    )
    result["transformers"] = (
        run_transformers(audio, a.runs, a.transformers_model)
        if a.only in (None, "transformers")
        else {"skipped": True}
    )
    result["verdict"] = verdict(result["vllm"], result["transformers"])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
