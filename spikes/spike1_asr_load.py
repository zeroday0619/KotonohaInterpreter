#!/usr/bin/env python3
"""Spike 1 — Jetson vLLM 이 Qwen3-ASR 을 로드하는가.

판정할 것:
  · vLLM 로드 성공 여부
  · 6초 오디오 전사 소요 시간
  · N-best 출력 가능 여부
  · 실패 시 transformers 경로의 소요 시간 (지연 예산 재계산용)

Jetson 에서 실행:
    python3 spikes/spike1_asr_load.py --wav samples/ko_6s.wav --out spikes/out/spike1.json

vLLM 컨테이너 안에서 돌릴 것:
    ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04
transformers 경로는 dustynv/jetson-containers 의 r36.4.0 계열 이미지에서.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

TRANSFORMERS_ID = "Qwen/Qwen3-ASR-1.7B-hf"
VLLM_ID = "Qwen/Qwen3-ASR-1.7B"
N_BEST = 5


def load_wav(path: Path, rate: int = 16000) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit(f"16-bit PCM WAV 만: {path}")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != rate:
        import soxr

        x = np.asarray(soxr.resample(x, sr, rate), dtype=np.float32)
    return x


def synthetic_6s(rate: int = 16000) -> np.ndarray:
    """실 녹음이 없을 때의 타이밍 전용 더미. 전사 내용은 의미 없다."""
    t = np.arange(int(6 * rate)) / rate
    env = 0.5 * (1 + np.sin(2 * np.pi * 3.1 * t))
    sig = np.sin(2 * np.pi * 180 * t) + 0.4 * np.sin(2 * np.pi * 720 * t)
    return (0.2 * env * sig).astype(np.float32)


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "machine": platform.machine()}
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


# ── transformers 경로 ────────────────────────────────────────────────────
def run_transformers(audio: np.ndarray, runs: int) -> dict:
    out: dict = {"backend": "transformers", "model": TRANSFORMERS_ID}
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except Exception as e:  # noqa: BLE001
        return {**out, "loaded": False, "error": f"import: {e!r}"}

    try:
        t0 = time.perf_counter()
        proc = AutoProcessor.from_pretrained(TRANSFORMERS_ID)
        model = AutoModelForMultimodalLM.from_pretrained(
            TRANSFORMERS_ID, dtype=torch.float16, device_map="auto"
        )
        model.eval()
        out["load_s"] = round(time.perf_counter() - t0, 2)
        out["loaded"] = True
    except Exception as e:  # noqa: BLE001
        return {**out, "loaded": False, "error": f"load: {e!r}"}

    def once(n_best: int) -> tuple[float, list[str], list[float]]:
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
        once(1)  # 워밍업
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


# ── vLLM 경로 ────────────────────────────────────────────────────────────
def run_vllm(audio: np.ndarray, runs: int) -> dict:
    """vLLM 이 이 모델을 아키텍처로 인식하는지부터 본다.

    인식하더라도 N-best(n>1) 와 로그확률 획득이 되는지 반드시 함께 확인한다.
    전사만 되고 N-best 가 안 나오면 §5.2 를 만족하지 못하므로 채택할 수 없다.
    """
    out: dict = {"backend": "vllm", "model": VLLM_ID}
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:  # noqa: BLE001
        return {**out, "loaded": False, "error": f"import: {e!r}"}

    try:
        t0 = time.perf_counter()
        llm = LLM(model=VLLM_ID, trust_remote_code=True, max_model_len=4096, dtype="float16")
        out["load_s"] = round(time.perf_counter() - t0, 2)
        out["loaded"] = True
    except Exception as e:  # noqa: BLE001
        return {**out, "loaded": False, "error": f"load: {e!r}"}

    try:
        sp = SamplingParams(
            temperature=0.0, max_tokens=256, n=N_BEST, logprobs=1, use_beam_search=True
        )
        prompt = {
            "prompt": "<|audio_bos|><|AUDIO|><|audio_eos|>",
            "multi_modal_data": {"audio": (audio, 16000)},
        }
        llm.generate([prompt], sp)  # 워밍업
        durs = []
        for _ in range(runs):
            t = time.perf_counter()
            res = llm.generate([prompt], sp)
            durs.append(time.perf_counter() - t)
        outs = res[0].outputs
        out.update(
            nbest_ms=round(statistics.median(durs) * 1000, 1),
            nbest_count=len(outs),
            nbest_ok=len(outs) == N_BEST,
            has_logprobs=any(o.cumulative_logprob is not None for o in outs),
            sample=[o.text for o in outs],
        )
    except Exception as e:  # noqa: BLE001
        out["infer_error"] = repr(e)
    return out


def verdict(vllm: dict, tf: dict) -> dict:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, default=None, help="6초 내외 실 녹음 (16-bit PCM)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", choices=["vllm", "transformers"], default=None)
    ap.add_argument("--out", type=Path, default=Path("spikes/out/spike1.json"))
    a = ap.parse_args()

    if a.wav:
        audio = load_wav(a.wav)
        src = str(a.wav)
    else:
        audio = synthetic_6s()
        src = "synthetic (타이밍 전용, 전사 내용 무의미)"
        print("!! 실 녹음 없이 실행 중. 시간만 신뢰할 것.", file=sys.stderr)

    res = {
        "spike": 1,
        "question": "Jetson vLLM 이 Qwen3-ASR 을 로드하는가",
        "audio": {"source": src, "seconds": round(len(audio) / 16000, 2)},
        "env": env_info(),
    }
    res["vllm"] = (
        run_vllm(audio, a.runs) if a.only in (None, "vllm") else {"skipped": True}
    )
    res["transformers"] = (
        run_transformers(audio, a.runs) if a.only in (None, "transformers") else {"skipped": True}
    )
    res["verdict"] = verdict(res["vllm"], res["transformers"])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
