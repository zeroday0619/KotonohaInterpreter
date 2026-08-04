#!/usr/bin/env python3
"""ASR scoring — CER via jiwer (§11).

CER, not WER. Word boundaries are defined differently across Korean, Japanese
and Chinese, so WER is not a comparable number.

    # 1) run the evaluation set through the real pipeline to get hypotheses
    python3 eval/run_asr.py --manifest eval/data/ko/manifest.jsonl --out eval/out/ko.hyp.jsonl
    # 2) score them
    python3 eval/score_cer.py --manifest eval/data/ko/manifest.jsonl --hyp eval/out/ko.hyp.jsonl
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path


def norm(
    s: str,
    /,
    keep_punct: bool = False,
) -> str:
    s = unicodedata.normalize("NFKC", s).strip().lower()
    if keep_punct:
        return " ".join(s.split())
    return "".join(
        ch for ch in s if not ch.isspace() and unicodedata.category(ch)[0] not in ("P", "Z")
    )


def read_jsonl(
    p: Path,
    /,
) -> dict[str, dict]:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["id"]] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--hyp", type=Path, required=True)
    ap.add_argument("--keep-punct", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    import jiwer

    refs, hyps = read_jsonl(a.manifest), read_jsonl(a.hyp)
    ids = [i for i in refs if i in hyps]
    missing = [i for i in refs if i not in hyps]

    r = [norm(refs[i]["reference"], a.keep_punct) for i in ids]
    h = [norm(hyps[i].get("text", ""), a.keep_punct) for i in ids]

    overall = jiwer.cer(r, h)
    per_item = sorted(
        (
            {"id": i, "cer": jiwer.cer(rr, hh), "ref": rr, "hyp": hh}
            for i, rr, hh in zip(ids, r, h, strict=True)
        ),
        key=lambda d: -d["cer"],
    )

    print(f"항목 {len(ids)}건 (누락 {len(missing)})")
    print(f"CER  {overall:.4f}")
    print("\n최악 10건:")
    for d in per_item[:10]:
        print(f"  {d['cer']:.3f}  {d['id']}")
        print(f"    ref: {d['ref'][:70]}")
        print(f"    hyp: {d['hyp'][:70]}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cer": overall, "n": len(ids), "items": per_item}
        a.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
