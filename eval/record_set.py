#!/usr/bin/env python3
"""Recording tool for the evaluation set (§11).

**Build this alongside Phase 1.** Without it, every later tuning decision rests
on impressions, and regressions become inevitable.

The rule: record **with the microphone that will be used, in the room where it
will be used.** A set captured on a headset in a quiet office produces tuning
that collapses in the field.

    python3 eval/record_set.py --lang ko --prompts eval/prompts/ko.txt --out eval/data/ko

Each utterance starts and stops with Enter. Output is 16 kHz 16-bit WAV plus
a manifest.jsonl.
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
    ap.add_argument("--prompts", type=Path, required=True, help="one utterance per line")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--start", type=int, default=0, help="index to resume recording from")
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
