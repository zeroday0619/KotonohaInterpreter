#!/usr/bin/env python3
"""평가셋 녹음 도구 (§11).

**Phase 1 과 병행해 만든다.** 이게 없으면 이후 모든 튜닝이 체감에 의존하게 되고
반드시 퇴행한다.

원칙: **실제 사용할 마이크로, 실제 사용할 공간에서** 녹음한다. 조용한 사무실에서
헤드셋으로 녹음한 셋으로 튜닝하면 현장에서 그대로 무너진다.

    python3 eval/record_set.py --lang ko --prompts eval/prompts/ko.txt --out eval/data/ko

각 발화는 스페이스로 시작/종료. 결과는 16kHz 16-bit WAV + manifest.jsonl.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import wave
from pathlib import Path

import numpy as np


def record_until_enter(rate: int, device=None) -> np.ndarray:
    import sounddevice as sd

    q: queue.Queue[np.ndarray] = queue.Queue()

    def cb(indata, frames, t, status):  # noqa: ANN001
        if status:
            print(status, file=sys.stderr)
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=rate, channels=1, dtype="float32", device=device, callback=cb):
        input("  녹음 중 … Enter 로 종료")
    parts = []
    while not q.empty():
        parts.append(q.get())
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def write_wav(path: Path, x: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["ko", "en", "zh-TW", "ja"])
    ap.add_argument("--prompts", type=Path, required=True, help="한 줄에 발화 하나")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--start", type=int, default=0, help="이어서 녹음할 인덱스")
    a = ap.parse_args()

    lines = [ln.strip() for ln in a.prompts.read_text(encoding="utf-8").splitlines() if ln.strip()]
    manifest = a.out / "manifest.jsonl"
    a.out.mkdir(parents=True, exist_ok=True)

    print(f"{a.lang}: {len(lines)}문장. 실제 마이크·실제 공간에서 녹음할 것.\n")
    with manifest.open("a", encoding="utf-8") as f:
        for i, text in enumerate(lines):
            if i < a.start:
                continue
            print(f"[{i + 1}/{len(lines)}] {text}")
            input("  Enter 로 시작")
            x = record_until_enter(a.rate, a.device)
            wav = a.out / f"{a.lang}_{i:03d}.wav"
            write_wav(wav, x, a.rate)
            f.write(
                json.dumps(
                    {
                        "id": f"{a.lang}_{i:03d}",
                        "lang": a.lang,
                        "audio": str(wav),
                        "reference": text,
                        "seconds": round(len(x) / a.rate, 2),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            print(f"  → {wav}  ({len(x) / a.rate:.1f}s)\n")
    print(f"완료: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
