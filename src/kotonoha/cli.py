"""kotonoha CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
from pathlib import Path

import numpy as np

from .config import Settings, load_settings
from .logging_setup import setup_logging


# ── 공통 조립 ────────────────────────────────────────────────────────────
def _build(settings: Settings, wav: Path | None = None):
    from .audio.capture import FileCapture, MicCapture
    from .audio.denoise import build_denoiser
    from .audio.playback import NullPlayback, Playback
    from .audio.vad import UtteranceSegmenter, build_vad
    from .core.orchestrator import Orchestrator

    v = settings.frontend.vad
    vad = build_vad(v.backend, settings.resolve(v.model_path), settings.audio.work_sample_rate)
    seg = UtteranceSegmenter(
        vad=vad,
        sample_rate=settings.audio.work_sample_rate,
        threshold=v.threshold,
        neg_threshold=v.neg_threshold,
        preroll_ms=v.preroll_ms,
        min_speech_ms=v.min_speech_ms,
        silence_ms=v.silence_ms,
        max_utterance_ms=v.max_utterance_ms,
    )

    if wav is not None:
        pcm = load_wav(wav, settings.audio.work_sample_rate)
        capture = FileCapture(pcm)
        playback = NullPlayback(settings.audio, settings.tts)
    else:
        capture = MicCapture(settings.audio, v)
        try:
            import sounddevice as sd

            sd.check_output_settings(
                device=settings.audio.output_device,
                samplerate=settings.audio.playback_sample_rate,
                channels=1,
            )
            playback = Playback(settings.audio, settings.tts)
        except Exception:  # noqa: BLE001
            playback = NullPlayback(settings.audio, settings.tts)

    d = settings.frontend.denoise
    denoiser = build_denoiser(d.enabled and wav is None, d.backend, d.post_filter_beta)

    return Orchestrator(settings, capture, seg, playback, denoiser)


def load_wav(path: Path, target_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"16-bit PCM WAV 만 지원: {path} ({width * 8}-bit)")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != target_rate:
        from .audio.resample import resample_once

        x = resample_once(x, rate, target_rate)
    return x


# ── 서브커맨드 ───────────────────────────────────────────────────────────
def cmd_run(args) -> int:
    s = load_settings(args.config)
    setup_logging(s.logging.level, s.resolve(s.logging.log_path), s.logging.console, "orch")
    from .tui import KotonohaApp

    orch = _build(s)
    KotonohaApp(orch).run()
    return 0


def cmd_replay(args) -> int:
    """마이크 없이 WAV 로 전체 경로를 굴린다. EOU/프리롤 회귀 확인용."""
    s = load_settings(args.config)
    # 파일 재생에는 누를 키가 없다. VAD 가 발화를 잘라야 하므로 자동 모드로 강제한다.
    s.session.mode = "auto"
    setup_logging(s.logging.level, s.resolve(s.logging.log_path), True, "replay")
    orch = _build(s, wav=Path(args.wav))

    async def go() -> None:
        await orch.start()
        await asyncio.sleep(args.seconds)
        await orch.stop()

    asyncio.run(go())
    print(f"턴 로그: {s.resolve(s.logging.turn_log_path)}")
    return 0


def cmd_devices(args) -> int:
    import sounddevice as sd

    print(sd.query_devices())
    print("\n기본 입력/출력:", sd.default.device)
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    target = {
        "asr": "kotonoha.services.asr_server:app",
        "verify": "kotonoha.services.asr_verify_server:app",
        "tts": "kotonoha.services.tts_server:app",
    }[args.service]
    port = args.port or {"asr": 8001, "verify": 8002, "tts": 8004}[args.service]
    uvicorn.run(target, host=args.host, port=port, log_level="info", workers=1)
    return 0


def cmd_glossary(args) -> int:
    import yaml

    s = load_settings(args.config)
    from .store import GlossaryEntry, Store

    st = Store(s.resolve(s.store.path))

    if args.action == "list":
        for g in st.all_glossary():
            print(f"{g.src_lang:>5} {g.src_term}  →  {g.tgt_lang} {g.tgt_term}   [{g.kind}]")
        return 0

    data = yaml.safe_load(Path(args.path).read_text(encoding="utf-8")) or {}
    entries = [GlossaryEntry(**e) for e in data.get("glossary", [])]
    n = st.upsert_glossary(entries) if entries else 0
    rules = [
        (r["pattern"], r["replacement"], bool(r.get("is_regex", False)), r.get("note"))
        for r in data.get("zh_rules", [])
    ]
    m = st.upsert_zh_rules(rules) if rules else 0
    print(f"용어 {n}건, 번체 규칙 {m}건 반영 → {s.resolve(s.store.path)}")
    return 0


def cmd_doctor(args) -> int:
    """실기 반입 전 점검. 추측하지 말고 여기서 확인한다(작업 규칙 3)."""
    import platform

    s = load_settings(args.config)
    print(f"python      {sys.version.split()[0]}  {platform.machine()}  {platform.system()}")
    print(f"config      {args.config or 'config/default.yaml'}")
    print(f"asr backend {s.asr.backend} / llm profile {s.llm.profile} / tts {s.tts.backend}")
    print(f"gguf        {s.resolve(s.llm.gguf_path)}")
    print()

    mods = [
        ("numpy", True), ("httpx", True), ("structlog", True), ("textual", True),
        ("soxr", True), ("sounddevice", False), ("opencc", False),
        ("onnxruntime", False), ("torch", False), ("transformers", False),
        ("faster_whisper", False), ("df", False), ("qwen_tts", False), ("melo", False),
    ]
    for name, required in mods:
        try:
            __import__(name)
            print(f"  [ok]   {name}")
        except Exception as e:  # noqa: BLE001
            tag = "FAIL" if required else "miss"
            print(f"  [{tag}] {name}  ({type(e).__name__})")

    print()
    vad_path = s.resolve(s.frontend.vad.model_path)
    print(f"  silero_vad.onnx  {'ok' if vad_path.exists() else 'MISSING'}  {vad_path}")

    async def probe() -> None:
        from .clients import AsrClient, AsrVerifyClient, LlmClient, TtsClient

        cs = [
            AsrClient(s.services.asr, s.asr),
            AsrVerifyClient(s.services.asr_verify, s.asr_verify),
            LlmClient(s.services.llm, s.llm),
            TtsClient(s.services.tts, s.tts),
        ]
        print("\n서비스:")
        for c in cs:
            h = await c.health()
            state = "UP" if h.get("ok") else "DOWN"
            print(f"  {c.name:<11} {state:<4} {json.dumps(h, ensure_ascii=False)[:96]}")
            await c.aclose()

    asyncio.run(probe())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kotonoha", description="순차식 4언어 오프라인 통역기")
    p.add_argument("-c", "--config", default=None, help="YAML 설정 경로")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="TUI 실행").set_defaults(fn=cmd_run)

    r = sub.add_parser("replay", help="WAV 로 파이프라인 재생 (마이크 없이)")
    r.add_argument("wav")
    r.add_argument("--seconds", type=float, default=30.0)
    r.set_defaults(fn=cmd_replay)

    sub.add_parser("devices", help="오디오 장치 목록").set_defaults(fn=cmd_devices)

    sv = sub.add_parser("serve", help="모델 서비스 기동")
    sv.add_argument("service", choices=["asr", "verify", "tts"])
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=None)
    sv.set_defaults(fn=cmd_serve)

    g = sub.add_parser("glossary", help="용어집 관리")
    g.add_argument("action", choices=["import", "list"])
    g.add_argument("path", nargs="?")
    g.set_defaults(fn=cmd_glossary)

    sub.add_parser("doctor", help="환경 점검").set_defaults(fn=cmd_doctor)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
