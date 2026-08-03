#!/usr/bin/env python3
"""평가셋을 실제 ASR 서비스에 태워 가설을 만든다 (Orin 에서 실행).

파이프라인 전체가 아니라 ASR 만 돈다. 프런트엔드(잡음 제거·VAD)의 영향까지 보려면
`kotonoha replay` 를 쓸 것.

    python3 eval/run_asr.py --manifest eval/data/ko/manifest.jsonl --out eval/out/ko.hyp.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kotonoha.cli import load_wav  # noqa: E402
from kotonoha.clients import AsrClient  # noqa: E402
from kotonoha.config import load_settings  # noqa: E402
from kotonoha.shmring import AudioRing  # noqa: E402


async def run(manifest: Path, out: Path, config: str | None) -> int:
    s = load_settings(config)
    ring = AudioRing.create(
        name=s.shm.name + "_eval",
        slots=4,
        slot_seconds=s.shm.slot_seconds,
        sample_rate=s.shm.sample_rate,
    )
    asr = AsrClient(s.services.asr, s.asr)
    if not await asr.wait_ready(timeout=600):
        print("ASR 서비스가 뜨지 않았다", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            pcm = load_wav(Path(item["audio"]), s.shm.sample_rate)
            ref = ring.publish(pcm)
            r = await asr.transcribe(ref)
            f.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "text": r.best,
                        "n_best": r.texts,
                        "language": r.language,
                        "language_confidence": r.language_confidence,
                        "avg_logprob": r.best_avg_logprob,
                        "infer_ms": r.infer_ms,
                        "lang_ref": item["lang"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
            print(f"  {item['id']}  {r.infer_ms:.0f}ms  {r.best[:50]}")

    await asr.aclose()
    ring.close()
    print(f"\n{n}건 → {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("-c", "--config", default=None)
    a = ap.parse_args()
    return asyncio.run(run(a.manifest, a.out, a.config))


if __name__ == "__main__":
    raise SystemExit(main())
