#!/usr/bin/env python3
"""Merge hardware spike results into a host-specific report and settings patch.

    bash scripts/manage.sh benchmark jetson

Jetson results produce a local device overlay. A6000 results produce a remote service
overlay. Host-specific measurements remain separate.

The generated report itself is written in Korean, since that is what gets read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(
    d: Path,
    /,
    n: int,
) -> dict | None:
    p = d / f"spike{n}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def md_report(
    s1: dict | None,
    /,
    s2: dict | None,
    s3: dict | None,
    target: str = "jetson",
) -> str:
    report_title = (
        "# Phase 0 — Jetson 검증 스파이크 결과"
        if target == "jetson"
        else "# 고성능 모드 — A6000 검증 결과"
    )
    L: list[str] = [report_title, ""]

    def section(
        title: str,
        /,
        data: dict | None,
        body: Any,
    ) -> None:
        L.append(f"## {title}")
        if data is None:
            L.append("")
            L.append("**미실행.** 실기에서 스크립트를 돌린 뒤 다시 생성할 것.")
            L.append("")
            return
        L.extend(body(data))
        L.append("")

    def b1(
        d: dict,
        /,
    ) -> list[str]:
        v = d.get("verdict", {})
        conditions = d.get("conditions", {})
        out = [
            "",
            f"- 오디오: {d['audio']['seconds']}초 ({d['audio']['source']})",
            f"- 장치: {d['env'].get('device')} / {d['env'].get('capability')}"
            f" / torch {d['env'].get('torch')}",
            f"- vLLM 조건: GPU 메모리 {conditions.get('gpu_memory_utilization')}"
            f", 컨텍스트 {conditions.get('max_model_len')}"
            f", eager {conditions.get('enforce_eager')}",
            "",
            "| 경로 | 로드 | N-best 5 | 로그확률 | N-best 전사(ms) |",
            "|---|---|---|---|---|",
        ]
        for key in ("vllm", "transformers"):
            r = d.get(key, {})
            if r.get("skipped"):
                out.append(f"| {key} | 건너뜀 | | | |")
                continue
            out.append(
                f"| {key} | {'✅' if r.get('loaded') else '❌ ' + str(r.get('error'))[:60]} "
                f"| {'✅' if r.get('nbest_ok') else '❌'} "
                f"| {'✅' if r.get('has_logprobs') else '❌'} "
                f"| {r.get('nbest_ms', '—')} |"
            )
        out += ["", f"**판정: `asr.backend: {v.get('recommended_backend')}`** — {v.get('note')}"]
        return out

    def b2(
        d: dict,
        /,
    ) -> list[str]:
        v = d.get("verdict", {})
        fa = d.get("flash_attn", {})
        omni = d.get("vllm_omni", {})
        fa_err = str(fa.get("kernel_error") or fa.get("error"))[:70]
        out = [
            "",
            f"- 장치: {d['env'].get('device')} / {d['env'].get('capability')}",
            f"- 이미지: {d['env'].get('image')} / vLLM-Omni {d['env'].get('vllm_omni')}",
            f"- flash-attn import: {'✅' if fa.get('import') else '❌'} "
            f"/ 커널 실행: {'✅' if fa.get('kernel_ok') else '❌ ' + fa_err}",
            f"- vLLM-Omni 시작: {'✅' if omni.get('loaded') else '❌'} "
            f"({omni.get('startup_s', '—')}초) / 로그: {omni.get('log', '—')}",
            f"- GPU 메모리: peak {omni.get('gpu_memory_peak_mib', '—')} MiB "
            f"/ baseline 대비 +{omni.get('gpu_memory_delta_mib', '—')} MiB",
            "",
            "| 언어 | PCM 스트리밍 | 첫 패킷(ms) | 전체 합성(ms) | RTF |",
            "|---|---|---|---|---|",
        ]
        for language in ("Korean", "English", "Japanese", "Chinese"):
            measurement = (omni.get("languages") or {}).get(language, {})
            status = (
                "✅"
                if measurement.get("ok")
                else "❌ " + str(measurement.get("error"))[:50]
            )
            out.append(
                f"| {language} "
                f"| {status} "
                f"| {measurement.get('median_ttfa_ms', '—')} "
                f"| {measurement.get('median_e2e_ms', '—')} "
                f"| {measurement.get('median_rtf', '—')} |"
            )
        out += ["", f"**판정: `tts.backend: {v.get('tts_backend')}`** — {v.get('note')}"]
        return out

    def b3(
        d: dict,
        /,
    ) -> list[str]:
        v = d.get("verdict", {})
        c = d.get("conditions", {})
        out = [
            "",
            f"- 조건: ctx {c.get('max_model_len')}, 배치 {c.get('batch')}, "
            f"출력 {c.get('output_tokens')}토큰",
            "",
            "| 프로필 | 모델 | 실 프롬프트 tok/s | TTFT(ms) |",
            "|---|---|---|---|",
        ]
        for key in ("moe", "dense"):
            r = d.get(key)
            if not r:
                continue
            best = ((r.get("server") or {}).get("best")) or {}
            out.append(
                f"| {key} | {r.get('directory', '')} "
                f"| {best.get('tok_per_s', '—')} | {best.get('ttft_ms', '—')} |"
            )
        out += [
            "",
            f"**판정: `llm.profile: {v.get('llm_profile')}`** — {v.get('note')}",
            "",
            f"> 기준: {v.get('threshold')} tok/s. 음성 1초에 4~5토큰이 필요하므로(§5.4), "
            "이 아래면 절 단위 스트리밍 재생이 끊긴다.",
        ]
        return out

    section("Spike 1 — vLLM 이 Qwen3-ASR 을 로드하는가", s1, b1)
    section("Spike 2 — vLLM-Omni Qwen3-TTS 가 동작하는가", s2, b2)
    section("Spike 3 — MoE vs 밀집 14B 실측 tok/s", s3, b3)

    L += ["## 종합", ""]
    if s1 and s3:
        asr_ms = (s1.get("verdict") or {}).get("nbest_ms") or 0
        tts_ms = ((s2 or {}).get("verdict") or {}).get("first_packet_ms_estimate") or 0
        best = ((s3.get(s3["verdict"]["llm_profile"]) or {}).get("server") or {}).get("best") or {}
        ttft = best.get("ttft_ms") or 0
        total = 800 + 100 + asr_ms + 100 + ttft + tts_ms
        L += [
            "| 단계 | 목표(ms) | 실측/추정(ms) |",
            "|---|---|---|",
            "| 침묵 대기 | 800 | 800 |",
            "| 프런트엔드 | 100 | 미측정 (Phase 1) |",
            f"| ASR (N-best 5) | 900 | {asr_ms} |",
            "| 교차 검증(조건부 평균) | 100 | 미측정 (Phase 5) |",
            f"| 정정+번역 첫 절 | 700 | {ttft} |",
            f"| TTS 첫 패킷 | 300 | {tts_ms} |",
            f"| **발화 종료 → 첫 음성** | **2900** | **{total:.0f}** |",
            "",
            ("✅ 예산 내." if total <= 2900 else "❌ 예산 초과. 위 표에서 초과 단계를 특정할 것."),
        ]
        if target == "a6000":
            L += [
                "",
                "> 위 합계는 A6000 서버 내부 단계만 포함한다. Jetson에서 `kotonoha netcheck`를 "
                "실행해 RTT와 오디오 업로드 시간을 별도로 더해야 한다.",
            ]
    else:
        L.append("스파이크 1·3 이 모두 끝나야 지연 예산을 대조할 수 있다.")
    return "\n".join(L) + "\n"


def patch_yaml(
    s1: dict | None,
    /,
    s2: dict | None,
    s3: dict | None,
    target: str = "jetson",
) -> str:
    destination = "config/local.yaml" if target == "jetson" else "config/remote-server.local.yaml"
    lines = [f"# 실측 결과로 확정된 값. {destination} 로 복사하면 적용된다.", ""]
    if s1:
        b = (s1.get("verdict") or {}).get("recommended_backend")
        if b in ("vllm", "transformers"):
            lines += ["asr:", f"  backend: {b}"]
            if b == "vllm":
                conditions = s1.get("conditions") or {}
                lines += [
                    "  vllm_gpu_memory_utilization: "
                    f"{conditions.get('gpu_memory_utilization', 0.80)}",
                    f"  vllm_max_model_len: {conditions.get('max_model_len', 4096)}",
                    "  vllm_enforce_eager: "
                    f"{str(conditions.get('enforce_eager', True)).lower()}",
                ]
            lines.append("")
    if s2:
        t = (s2.get("verdict") or {}).get("tts_backend")
        if t == "vllm_omni":
            lines += ["tts:", f"  backend: {t}", ""]
    if s3:
        p = (s3.get("verdict") or {}).get("llm_profile")
        if p in ("moe", "dense"):
            lines += ["llm:", f"  profile: {p}", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["jetson", "a6000"], default="jetson")
    ap.add_argument("--dir", type=Path, default=Path("spikes/out"))
    ap.add_argument("--md", type=Path, default=Path("spikes/out/PHASE0.md"))
    ap.add_argument("--patch", type=Path, default=Path("spikes/out/local.yaml"))
    a = ap.parse_args()

    s1, s2, s3 = load(a.dir, 1), load(a.dir, 2), load(a.dir, 3)
    for result in (s1, s2, s3):
        if result and result.get("target", a.target) != a.target:
            raise SystemExit(
                f"result target {result.get('target')} does not match --target {a.target}"
            )
    a.md.parent.mkdir(parents=True, exist_ok=True)
    a.md.write_text(md_report(s1, s2, s3, a.target), encoding="utf-8")
    a.patch.write_text(patch_yaml(s1, s2, s3, a.target), encoding="utf-8")
    print(a.md.read_text(encoding="utf-8"))
    print(f"\n→ 설정 패치: {a.patch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
