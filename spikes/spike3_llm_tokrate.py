#!/usr/bin/env python3
"""Spike 3 — measured token generation rate, MoE versus dense 14B.

Default conditions are context 2048, batch 1, and 60 output tokens. The A6000 track uses
the production context from `config/performance.yaml`. A result below 5 tok/s selects the
dense 14B profile. The MoE profile requires a generation rate at least equal to the dense
profile and acceptable translation quality.

The Orin provides 204.8 GB/s of memory bandwidth. MoE inference reads active
parameters, while expert routing changes memory locality between tokens. The
benchmark measures the resulting behavior instead of deriving it from model size.

The spike records two measurements:

  · llama-bench (raw generation speed)
  · a representative translation prompt streamed through llama-server, including TTFT

The llama-server measurement governs the decision because it includes production prompt
processing within the §6 first-clause stage.

    python3 spikes/spike3_llm_tokrate.py \\
        --bin /opt/llama.cpp/build/bin \\
        --models-dir ./models/gguf \\
        --out spikes/out/spike3.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROFILES = {
    "moe": {
        "repo": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        "file": "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
        "note": "30B MoE, 3B active",
    },
    "dense": {
        "repo": "unsloth/Qwen3-14B-GGUF",
        "file": "Qwen3-14B-Q4_K_M.gguf",
        "note": "dense 14B",
    },
}

DEFAULT_CONTEXT_LENGTH = 2048
DEFAULT_OUTPUT_TOKENS = 60
MIN_TOK_S = 5.0

# A prompt shaped like the real thing (§5.3 single pass: N-best + history + glossary)
SYSTEM = (
    "You are a professional consecutive interpreter. Reconstruct what the speaker said "
    "from the ASR hypotheses, then translate it into English. Output the translation only."
)
USER = """## Conversation so far
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


def run_llama_bench(
    bin_dir: Path,
    /,
    model: Path,
    ngl: int,
    output_tokens: int,
) -> dict:
    exe = bin_dir / "llama-bench"
    if not exe.exists():
        return {"skipped": f"not found: {exe}"}
    cmd = [
        str(exe), "-m", str(model), "-p", "512", "-n", str(output_tokens),
        "-b", "512", "-ngl", str(ngl), "-r", "3", "-o", "md",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    if p.returncode != 0:
        return {"error": p.stderr[-800:]}

    out = {"raw": p.stdout[-2000:]}
    for line in p.stdout.splitlines():
        m = re.search(r"\btg(\d+)\b.*?([\d.]+)\s*±", line)
        if m:
            out["tg_tok_per_s"] = float(m.group(2))
        m = re.search(r"\bpp(\d+)\b.*?([\d.]+)\s*±", line)
        if m:
            out["pp_tok_per_s"] = float(m.group(2))
    return out


def wait_health(
    url: str,
    /,
    timeout: float,
) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2.0)
    return False


def stream_translate(
    url: str,
    /,
    output_tokens: int,
) -> dict:
    """Stream a real translation prompt over SSE, timing both TTFT and rate."""
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": output_tokens,
            "stream": True,
            "cache_prompt": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    ttft = None
    n = 0
    text = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = obj.get("choices") or []
            if not ch:
                continue
            d = (ch[0].get("delta") or {}).get("content")
            if not d:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            n += 1
            text.append(d)
    total = time.perf_counter() - t0
    gen = total - (ttft or 0.0)
    return {
        "ttft_ms": round((ttft or 0) * 1000, 1),
        "tokens": n,
        "total_ms": round(total * 1000, 1),
        "tok_per_s": round((n - 1) / gen, 2) if n > 1 and gen > 0 else None,
        "output": "".join(text)[:400],
    }


def bench_profile(
    name: str,
    /,
    bin_dir: Path,
    models_dir: Path,
    ngl: int,
    port: int,
    context_length: int,
    output_tokens: int,
    runs: int,
) -> dict:
    spec = PROFILES[name]
    model = models_dir / spec["file"]
    result: dict = {"profile": name, "model": str(model), **spec}
    if not model.exists():
        return {**result, "error": f"GGUF missing: {model}. See scripts/fetch_models.sh"}

    result["llama_bench"] = run_llama_bench(bin_dir, model, ngl, output_tokens)

    exe = bin_dir / "llama-server"
    if not exe.exists():
        result["server"] = {"skipped": f"not found: {exe}"}
        return result

    url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            str(exe), "-m", str(model), "-c", str(context_length), "-ngl", str(ngl),
            "--port", str(port), "--host", "127.0.0.1", "-np", "1", "--no-webui",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        t0 = time.perf_counter()
        if not wait_health(url, timeout=600):
            result["server"] = {"error": "health timeout"}
            return result
        result["server_load_s"] = round(time.perf_counter() - t0, 1)
        stream_translate(url, output_tokens)  # warm-up
        measurements = [stream_translate(url, output_tokens) for _ in range(runs)]
        best = max(measurements, key=lambda r: r["tok_per_s"] or 0)
        result["server"] = {"runs": measurements, "best": best}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    return result


def verdict(
    results: dict,
    /,
) -> dict:
    def rate(
        r: dict,
        /,
    ) -> float | None:
        s = (r.get("server") or {}).get("best") or {}
        return s.get("tok_per_s") or (r.get("llama_bench") or {}).get("tg_tok_per_s")

    moe, dense = rate(results.get("moe", {})), rate(results.get("dense", {}))
    lines = []
    if moe is not None:
        lines.append(f"MoE 30B-A3B: {moe} tok/s")
    if dense is not None:
        lines.append(f"밀집 14B: {dense} tok/s")

    if moe is not None and moe >= MIN_TOK_S and (dense is None or moe >= dense):
        choice, why = "moe", "MoE 가 5 tok/s 이상이고 밀집보다 느리지 않다. 품질 비교 후 채택."
    elif dense is not None and dense >= MIN_TOK_S:
        choice, why = "dense", f"MoE 가 기준 미달({moe}) 또는 더 느림. 명세대로 밀집 14B 로 회귀."
    else:
        choice, why = (
            "none",
            f"둘 다 {MIN_TOK_S} tok/s 미만 (moe={moe}, dense={dense}). "
            "절 단위 스트리밍이 성립하지 않는다. 더 작은 모델 또는 낮은 양자화 검토 필요.",
        )

    return {
        "llm_profile": choice,
        "moe_tok_per_s": moe,
        "dense_tok_per_s": dense,
        "threshold": MIN_TOK_S,
        "note": " / ".join(lines) + " — " + why,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["jetson", "a6000"], default="jetson")
    ap.add_argument("--bin", type=Path, required=True, help="llama.cpp build bin directory")
    ap.add_argument("--models-dir", type=Path, default=Path("./models/gguf"))
    ap.add_argument("--ngl", type=int, default=999, help="number of layers offloaded to GPU")
    ap.add_argument("--port", type=int, default=18003)
    ap.add_argument("--context", type=int, default=DEFAULT_CONTEXT_LENGTH)
    ap.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", choices=list(PROFILES), default=None)
    ap.add_argument("--out", type=Path, default=Path("spikes/out/spike3.json"))
    a = ap.parse_args()

    result: dict = {
        "spike": 3,
        "target": a.target,
        "question": "MoE(활성 3B) vs 밀집 14B 실측 tok/s",
        "conditions": {
            "n_ctx": a.context,
            "batch": 1,
            "n_predict": a.output_tokens,
            "runs": a.runs,
        },
    }
    for name in PROFILES:
        if a.only and name != a.only:
            continue
        print(f"── {name} 측정 중 …", file=sys.stderr)
        result[name] = bench_profile(
            name,
            a.bin,
            a.models_dir,
            a.ngl,
            a.port,
            a.context,
            a.output_tokens,
            a.runs,
        )
    result["verdict"] = verdict(result)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
