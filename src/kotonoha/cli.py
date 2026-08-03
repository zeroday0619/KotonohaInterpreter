"""kotonoha CLI, built on Typer.

The global --config option lives on the callback and reaches each command through
`ctx.obj`. Commands signal failure with `typer.Exit`; nothing returns an exit code
directly.

Command output text is unchanged from the argparse implementation. This module
migrates the argument parser, not the operator-facing output.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from .config import Settings, load_settings
from .logging_setup import setup_logging


# -- shared wiring --------------------------------------------------------
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
        raise ValueError(f"only 16-bit PCM WAV is supported: {path} ({width * 8}-bit)")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != target_rate:
        from .audio.resample import resample_once

        x = resample_once(x, rate, target_rate)
    return x


# -- application ----------------------------------------------------------
@dataclass
class AppState:
    """Carried on `ctx.obj` so every command sees the same --config."""

    config: Path | None = None


class ServiceName(str, Enum):
    asr = "asr"
    verify = "verify"
    tts = "tts"


SERVICE_TARGETS = {
    ServiceName.asr: ("kotonoha.services.asr_server:app", 8001),
    ServiceName.verify: ("kotonoha.services.asr_verify_server:app", 8002),
    ServiceName.tts: ("kotonoha.services.tts_server:app", 8004),
}

app = typer.Typer(
    name="kotonoha",
    help="순차식 4언어 오프라인 통역기",
    no_args_is_help=True,
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
glossary_app = typer.Typer(name="glossary", help="용어집 관리", no_args_is_help=True)
app.add_typer(glossary_app)

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "-c",
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="YAML 설정 경로",
    ),
]


@app.callback()
def cli(ctx: typer.Context, config: ConfigOption = None) -> None:
    ctx.obj = AppState(config=config)


def _settings(ctx: typer.Context) -> Settings:
    return load_settings(ctx.obj.config)


# -- commands -------------------------------------------------------------
@app.command()
def run(ctx: typer.Context) -> None:
    """TUI 실행"""
    s = _settings(ctx)
    setup_logging(s.logging.level, s.resolve(s.logging.log_path), s.logging.console, "orch")
    from .tui import KotonohaApp

    KotonohaApp(_build(s)).run()


@app.command()
def replay(
    ctx: typer.Context,
    wav: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="16-bit PCM WAV")],
    seconds: Annotated[float, typer.Option(help="실행 시간")] = 30.0,
) -> None:
    """WAV 로 파이프라인 재생 (마이크 없이)

    Runs the whole pipeline from a file. This is the regression path for
    end-of-utterance and preroll behaviour.
    """
    s = _settings(ctx)
    # There is no key to press when replaying a file, and the VAD has to do the
    # segmenting, so force automatic mode.
    s.session.mode = "auto"
    setup_logging(s.logging.level, s.resolve(s.logging.log_path), True, "replay")
    orch = _build(s, wav=wav)

    async def go() -> None:
        await orch.start()
        await asyncio.sleep(seconds)
        await orch.stop()

    asyncio.run(go())
    print(f"턴 로그: {s.resolve(s.logging.turn_log_path)}")


@app.command()
def devices() -> None:
    """오디오 장치 목록"""
    import sounddevice as sd

    print(sd.query_devices())
    print("\n기본 입력/출력:", sd.default.device)


@app.command()
def serve(
    service: Annotated[ServiceName, typer.Argument(help="기동할 서비스")],
    host: Annotated[str, typer.Option(help="바인드 주소")] = "0.0.0.0",
    port: Annotated[int | None, typer.Option(help="기본값은 서비스별 포트")] = None,
) -> None:
    """모델 서비스 기동"""
    import uvicorn

    target, default_port = SERVICE_TARGETS[service]
    uvicorn.run(target, host=host, port=port or default_port, log_level="info", workers=1)


@glossary_app.command("import")
def glossary_import(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="용어집 YAML")],
) -> None:
    """용어집·번체 규칙 YAML 을 DB 에 반영"""
    import yaml

    from .store import GlossaryEntry, Store

    s = _settings(ctx)
    st = Store(s.resolve(s.store.path))

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = [GlossaryEntry(**e) for e in data.get("glossary", [])]
    n = st.upsert_glossary(entries) if entries else 0
    rules = [
        (r["pattern"], r["replacement"], bool(r.get("is_regex", False)), r.get("note"))
        for r in data.get("zh_rules", [])
    ]
    m = st.upsert_zh_rules(rules) if rules else 0
    print(f"용어 {n}건, 번체 규칙 {m}건 반영 → {s.resolve(s.store.path)}")


@glossary_app.command("list")
def glossary_list(ctx: typer.Context) -> None:
    """등록된 용어 출력"""
    from .store import Store

    s = _settings(ctx)
    st = Store(s.resolve(s.store.path))
    for g in st.all_glossary():
        print(f"{g.src_lang:>5} {g.src_term}  →  {g.tgt_lang} {g.tgt_term}   [{g.kind}]")


@app.command()
def doctor(ctx: typer.Context) -> None:
    """환경 점검

    Pre-flight check before taking anything to the device. Do not guess at the
    environment; confirm it here.
    """
    import platform

    s = _settings(ctx)
    print(f"python      {sys.version.split()[0]}  {platform.machine()}  {platform.system()}")
    print(f"config      {ctx.obj.config or 'config/default.yaml'}")
    print(f"asr backend {s.asr.backend} / llm profile {s.llm.profile} / tts {s.tts.backend}")
    print(f"gguf        {s.resolve(s.llm.gguf_path)}")
    placement = s.resolved_placement()
    print(f"perf_mode   {s.perf_mode}  remote={'on' if s.remote.enabled else 'off'}")
    print("placement   " + "  ".join(f"{k}={v}" for k, v in placement.items()))
    if s.audio_leaves_device:
        print("            ! 이 모드에서는 발화 오디오가 기기 밖으로 나간다")
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
        from .clients import build_service_group

        group = build_service_group(s)
        print("\n서비스:")
        for role in group.all():
            for client in filter(None, (role.preferred, role.fallback)):
                h = await client.health()
                state = "UP" if h.get("ok") else "DOWN"
                tag = f"{role.role}@{client.side}"
                print(f"  {tag:<20} {state:<4} {json.dumps(h, ensure_ascii=False)[:80]}")
        await group.aclose()

    asyncio.run(probe())


@app.command()
def netcheck(
    ctx: typer.Context,
    samples: Annotated[int, typer.Option(help="역할당 측정 횟수")] = 10,
    seconds: Annotated[float, typer.Option(help="probe 발화 길이")] = 6.0,
) -> None:
    """외부 A6000 링크 지연·대역폭 측정

    Every remote stage pays the round trip, and the utterance audio pays the
    upload on top. §6 has 2.9 s total with no slack, so this gets measured
    rather than assumed.
    """
    import statistics

    import httpx

    from .clients.base import remote_transport_kwargs
    from .transport import encode_pcm

    s = _settings(ctx)
    if not s.remote.enabled:
        print("remote.enabled 가 false 다. config/performance.yaml 을 쓰거나 켜고 다시 실행할 것.")
        raise typer.Exit(code=1)

    placement = s.resolved_placement()
    remote_roles = [r for r, side in placement.items() if side == "remote"]
    if not remote_roles:
        print(f"perf_mode={s.perf_mode} 에서 원격으로 가는 역할이 없다.")
        raise typer.Exit(code=1)

    tk = remote_transport_kwargs(s.remote)
    pcm = np.zeros(int(seconds * s.shm.sample_rate), dtype=np.float32)
    blob = encode_pcm(pcm, s.remote.audio_encoding)

    async def go() -> bool:
        print(f"perf_mode   {s.perf_mode}")
        print("placement   " + "  ".join(f"{k}={v}" for k, v in placement.items()))
        print(f"probe       {seconds}s utterance, {s.remote.audio_encoding}, {len(blob)} bytes\n")

        rtts: dict[str, float] = {}
        uploads: dict[str, float] = {}
        failed: list[str] = []

        async with httpx.AsyncClient(
            headers=tk["headers"],
            verify=tk["verify"],
            timeout=httpx.Timeout(10.0, connect=tk["connect_timeout"]),
        ) as client:
            for role in remote_roles:
                url = s.url_for(role, "remote").rstrip("/")
                measured = []
                for _ in range(samples):
                    t0 = time.perf_counter()
                    try:
                        r = await client.get(f"{url}/health")
                        r.raise_for_status()
                    except Exception as e:  # noqa: BLE001
                        print(f"  {role:<11} DOWN  {e!r}")
                        failed.append(role)
                        break
                    measured.append((time.perf_counter() - t0) * 1000)
                if not measured:
                    continue
                p50 = statistics.median(measured)
                p95 = (
                    max(measured)
                    if len(measured) < 20
                    else statistics.quantiles(measured, n=20)[18]
                )
                rtts[role] = p50
                print(f"  {role:<11} UP    rtt p50 {p50:6.1f}ms   p95 {p95:6.1f}ms   {url}")

            # Only the roles that receive audio pay the upload.
            for role in [r for r in ("asr", "asr_verify") if r in rtts]:
                url = s.url_for(role, "remote").rstrip("/")
                measured = []
                for _ in range(samples):
                    t0 = time.perf_counter()
                    try:
                        r = await client.post(
                            f"{url}/echo",
                            files={"audio": ("probe.pcm", blob, "application/octet-stream")},
                        )
                        r.raise_for_status()
                    except Exception as e:  # noqa: BLE001
                        print(f"  {role:<11} upload FAILED  {e!r}")
                        break
                    measured.append((time.perf_counter() - t0) * 1000)
                if not measured:
                    continue
                med = statistics.median(measured)
                uploads[role] = med
                mbps = (len(blob) / 1e6) / (med / 1000)
                print(f"  {role:<11} upload median {med:6.1f}ms   {mbps:5.1f} MB/s")

        if failed:
            print(f"\n연결 실패: {', '.join(failed)} — 이 역할은 온보드로 폴백된다.")

        # What the link adds to one turn, stage by stage.
        overhead = uploads.get("asr", rtts.get("asr", 0.0))
        if s.asr_verify.mode == "always":
            overhead += uploads.get("asr_verify", rtts.get("asr_verify", 0.0))
        overhead += rtts.get("llm", 0.0) + rtts.get("tts", 0.0)

        b = s.budget_ms
        slack = b.total - b.silence
        print(f"\n턴당 링크 오버헤드 추정  {overhead:.0f}ms")
        print(f"§6 EOU→첫 음성 예산       {slack}ms")
        if overhead > slack * 0.25:
            print("  ! 링크가 예산의 25% 를 넘게 먹는다. hybrid(LLM 만 원격) 를 검토할 것.")
        else:
            print("  링크 자체는 예산 안에 든다. 남은 것은 모델 추론 시간이다.")
        return not failed

    if not asyncio.run(go()):
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
