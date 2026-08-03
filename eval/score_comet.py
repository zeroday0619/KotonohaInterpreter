#!/usr/bin/env python3
"""Translation scoring — COMET. **Development machine only; never on the Orin (§11).**

COMET loads an XLM-R large-sized model. The scorer must not eat the memory
bandwidth of the machine that has to run the interpreter.

BLEU is not used: it correlates poorly with Korean and Japanese quality.

    uv sync --group eval
    uv run eval/score_comet.py --hyp eval/out/ko2en.jsonl --model Unbabel/wmt22-comet-da

Each line of the input JSONL: {"id", "src", "mt", "ref"}
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
            "Running on aarch64. COMET belongs in an offline batch on the "
            "development machine (§11).\nUse --force-on-device only if you really mean it."
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
