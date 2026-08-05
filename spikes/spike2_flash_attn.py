#!/usr/bin/env python3
"""Spike 2 — does FlashAttention and Qwen3-TTS run on the selected GPU?

What this decides:
  · whether flash_attn imports AND its kernel actually runs. Import alone is
    not enough: an aarch64 wheel can import fine and then die in the kernel.
  · whether Qwen3-TTS loads with flash_attention_2, sdpa or eager
  · how long synthesis takes in each configuration, against the 300 ms
    first-packet budget in §6
  · if it all fails, the evidence for starting on MeloTTS instead

Run this probe through ``bash spikes/run_all.sh <target>``. Set ``SPIKE_TTS_IMAGE`` to a
candidate FlashAttention image when validating a build other than the deployment TTS
image.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
PROBE_TEXT = {
    "Korean": "안녕하세요, 지금 통역기 성능을 확인하고 있습니다.",
    "English": "Hello, this is a latency probe for the interpreter.",
}


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
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            info["device"] = torch.cuda.get_device_name(0)
            info["capability"] = f"sm_{cap[0]}{cap[1]}"
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = repr(e)
    return info


def probe_flash_attn() -> dict:
    """Do not judge on the import alone; actually run the kernel once."""
    out: dict = {}
    try:
        import flash_attn

        out["import"] = True
        out["version"] = getattr(flash_attn, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        return {"import": False, "error": repr(e)}

    try:
        import torch
        from flash_attn import flash_attn_func

        q = torch.randn(1, 128, 8, 64, dtype=torch.float16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        out["kernel_ok"] = bool(o.shape == q.shape) and bool(torch.isfinite(o).all())
        out["kernel_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    except Exception as e:  # noqa: BLE001
        out["kernel_ok"] = False
        out["kernel_error"] = repr(e)
    return out


def probe_qwen3_tts(
    attn: str,
    /,
    runs: int,
    model_identifier: str,
) -> dict:
    out: dict = {"attn_implementation": attn}
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except Exception as e:  # noqa: BLE001
        return {**out, "loaded": False, "error": f"import: {e!r}"}

    try:
        t0 = time.perf_counter()
        model = Qwen3TTSModel.from_pretrained(
            model_identifier,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation=attn,
        )
        out["load_s"] = round(time.perf_counter() - t0, 2)
        out["loaded"] = True
    except Exception as e:  # noqa: BLE001
        return {**out, "loaded": False, "error": f"load: {e!r}"}

    try:
        for lang, text in PROBE_TEXT.items():
            model.generate_custom_voice(text=text, language=lang, speaker="Vivian")  # warm-up
            durs = []
            for _ in range(runs):
                t = time.perf_counter()
                wavs, sr = model.generate_custom_voice(
                    text=text, language=lang, speaker="Vivian"
                )
                durs.append(time.perf_counter() - t)
            audio_s = len(wavs[0]) / sr
            best = min(durs)
            out[lang] = {
                "synth_ms": round(best * 1000, 1),
                "audio_s": round(audio_s, 2),
                "rtf": round(best / audio_s, 3),
            }
    except Exception as e:  # noqa: BLE001
        out["infer_error"] = repr(e)
    return out


def probe_melo(
    runs: int,
    /,
) -> dict:
    out: dict = {"backend": "melo"}
    try:
        from melo.api import TTS
    except Exception as e:  # noqa: BLE001
        return {**out, "loaded": False, "error": f"import: {e!r}"}
    try:
        t0 = time.perf_counter()
        m = TTS(language="KR", device="cuda:0")
        out["load_s"] = round(time.perf_counter() - t0, 2)
        out["loaded"] = True
        spk = next(iter(m.hps.data.spk2id.values()))
        m.tts_to_file(PROBE_TEXT["Korean"], spk, None, speed=1.0)
        durs = []
        for _ in range(runs):
            t = time.perf_counter()
            wav = m.tts_to_file(PROBE_TEXT["Korean"], spk, None, speed=1.0)
            durs.append(time.perf_counter() - t)
        audio_s = len(wav) / m.hps.data.sampling_rate
        best = min(durs)
        out["synth_ms"] = round(best * 1000, 1)
        out["audio_s"] = round(audio_s, 2)
        out["rtf"] = round(best / audio_s, 3)
    except Exception as e:  # noqa: BLE001
        out["infer_error"] = repr(e)
    return out


def verdict(
    fa: dict,
    /,
    qwen: list[dict],
    melo: dict,
) -> dict:
    working = [q for q in qwen if q.get("loaded") and "infer_error" not in q]
    budget = 300
    if working:
        best = min(
            working,
            key=lambda q: min(
                (v["synth_ms"] for v in q.values() if isinstance(v, dict) and "synth_ms" in v),
                default=1e9,
            ),
        )
        ms = min(
            (v["synth_ms"] for v in best.values() if isinstance(v, dict) and "synth_ms" in v),
            default=None,
        )
        return {
            "tts_backend": "qwen3",
            "attn_implementation": best["attn_implementation"],
            "first_packet_ms_estimate": ms,
            "within_budget": bool(ms and ms <= budget),
            "flash_attn_usable": bool(fa.get("kernel_ok")),
            "note": (
                f"Qwen3-TTS 가 {best['attn_implementation']} 로 동작. "
                f"절 하나 합성 {ms:.0f}ms (§6 예산 {budget}ms). "
                + (
                    "flash-attn 없이도 되므로 빌드에 시간 쓰지 말 것."
                    if not fa.get("kernel_ok")
                    else "flash-attn 사용 가능."
                )
            ),
        }
    return {
        "tts_backend": "melo" if melo.get("loaded") else "none",
        "flash_attn_usable": bool(fa.get("kernel_ok")),
        "first_packet_ms_estimate": melo.get("synth_ms"),
        "note": (
            "Qwen3-TTS 를 어떤 attn 구현으로도 못 올렸다. MeloTTS 로 Phase 1~3 을 시작하고, "
            "Qwen3-TTS 는 별도 트랙으로 분리할 것."
            if melo.get("loaded")
            else "Qwen3-TTS·MeloTTS 모두 실패. TTS 스택 재검토 필요."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["jetson", "a6000"], default="jetson")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("spikes/out/spike2.json"))
    ap.add_argument("--skip-melo", action="store_true")
    ap.add_argument("--model", default=TTS_MODEL)
    a = ap.parse_args()

    result: dict = {
        "spike": 2,
        "target": a.target,
        "question": f"{a.target} 에서 flash-attn 과 Qwen3-TTS 가 동작하는가",
        "env": env_info(),
    }
    result["flash_attn"] = probe_flash_attn()
    result["qwen3_tts"] = [
        probe_qwen3_tts(attention, a.runs, a.model)
        for attention in ("flash_attention_2", "sdpa", "eager")
    ]
    result["melo"] = {} if a.skip_melo else probe_melo(a.runs)
    result["verdict"] = verdict(result["flash_attn"], result["qwen3_tts"], result["melo"])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
