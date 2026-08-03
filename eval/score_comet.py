#!/usr/bin/env python3
"""번역 채점 — COMET. **개발 PC 에서만 실행한다. Orin 에 올리지 않는다(§11).**

COMET 은 XLM-R large 급 모델을 통째로 올린다. 통역기가 돌아야 할 기기의
메모리 대역폭을 채점기가 먹으면 안 된다.

BLEU 는 쓰지 않는다. 한국어·일본어 품질과 상관이 낮다.

    pip install 'unbabel-comet>=2.2'
    python3 eval/score_comet.py --hyp eval/out/ko2en.jsonl --model Unbabel/wmt22-comet-da

입력 JSONL 각 줄: {"id", "src", "mt", "ref"}
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", type=Path, required=True)
    ap.add_argument("--model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force-on-device", action="store_true")
    a = ap.parse_args()

    if platform.machine() == "aarch64" and not a.force_on_device:
        raise SystemExit(
            "aarch64 에서 실행하려 하고 있다. COMET 은 개발 PC 에서 오프라인 배치로 돌린다(§11).\n"
            "정말 필요하면 --force-on-device."
        )

    from comet import download_model, load_from_checkpoint

    rows = [json.loads(ln) for ln in a.hyp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    data = [{"src": r["src"], "mt": r["mt"], "ref": r["ref"]} for r in rows]

    model = load_from_checkpoint(download_model(a.model))
    res = model.predict(data, batch_size=a.batch_size, gpus=a.gpus)

    scored = sorted(
        (
            {"id": r["id"], "score": s, **d}
            for r, d, s in zip(rows, data, res.scores, strict=True)
        ),
        key=lambda x: x["score"],
    )
    print(f"항목 {len(rows)}건")
    print(f"COMET (system) {res.system_score:.4f}")
    print("\n최저 10건:")
    for x in scored[:10]:
        print(f"  {x['score']:.3f}  {x['id']}")
        print(f"    src: {x['src'][:70]}")
        print(f"    mt : {x['mt'][:70]}")
        print(f"    ref: {x['ref'][:70]}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(
            json.dumps(
                {"model": a.model, "system_score": res.system_score, "items": scored},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
