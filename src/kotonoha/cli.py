"""kotonoha CLI, built on Typer.

The global --config option lives on the callback and reaches each command through
`ctx.obj`. Commands signal failure with `typer.Exit`; nothing returns an exit code
directly.

All operator-facing text goes through `i18n.t`. Typer renders command help at import
time, so help text follows the locale resolved then: KOTONOHA_LANG, ui.language in the
configuration, the system locale, then English. The --lang option is applied after
parsing and therefore affects command output, not the help screens.
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
from .i18n import available_locales, set_locale
from .i18n import t as _
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
    help=_("cli.app.help"),
    no_args_is_help=True,
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
glossary_app = typer.Typer(name="glossary", help=_("cli.glossary.help"), no_args_is_help=True)
app.add_typer(glossary_app)

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "-c",
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        help=_("cli.opt.config"),
    ),
]
LangOption = Annotated[
    str | None,
    typer.Option("--lang", help=_("cli.opt.lang")),
]


@app.callback()
def cli(ctx: typer.Context, config: ConfigOption = None, lang: LangOption = None) -> None:
    if lang and lang != "auto":
        if lang not in available_locales():
            raise typer.BadParameter(f"{lang} not in {', '.join(available_locales())}")
        set_locale(lang)
    ctx.obj = AppState(config=config)


def _settings(ctx: typer.Context) -> Settings:
    return load_settings(ctx.obj.config)


# -- commands -------------------------------------------------------------
@app.command(help=_("cli.run.help"))
def run(ctx: typer.Context) -> None:
    s = _settings(ctx)
    setup_logging(s.logging.level, s.resolve(s.logging.log_path), s.logging.console, "orch")
    from .tui import KotonohaApp

    KotonohaApp(_build(s)).run()


@app.command(help=_("cli.replay.help"))
def replay(
    ctx: typer.Context,
    wav: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help=_("cli.replay.arg.wav"))
    ],
    seconds: Annotated[float, typer.Option(help=_("cli.replay.opt.seconds"))] = 30.0,
) -> None:
    # Runs the whole pipeline from a file. This is the regression path for
    # end-of-utterance and preroll behaviour.
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
    print(_("cli.replay.turn_log", path=s.resolve(s.logging.turn_log_path)))


@app.command(help=_("cli.devices.help"))
def devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())
    print("\n" + _("cli.devices.default"), sd.default.device)


@app.command(help=_("cli.serve.help"))
def serve(
    service: Annotated[ServiceName, typer.Argument(help=_("cli.serve.arg.service"))],
    host: Annotated[str, typer.Option(help=_("cli.serve.opt.host"))] = "0.0.0.0",
    port: Annotated[int | None, typer.Option(help=_("cli.serve.opt.port"))] = None,
) -> None:
    import uvicorn

    target, default_port = SERVICE_TARGETS[service]
    uvicorn.run(target, host=host, port=port or default_port, log_level="info", workers=1)


@glossary_app.command("import", help=_("cli.glossary.import.help"))
def glossary_import(
    ctx: typer.Context,
    path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help=_("cli.glossary.import.arg.path"))
    ],
) -> None:
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
    print(_("cli.glossary.imported", terms=n, rules=m, path=s.resolve(s.store.path)))


@glossary_app.command("list", help=_("cli.glossary.list.help"))
def glossary_list(ctx: typer.Context) -> None:
    from .store import Store

    s = _settings(ctx)
    st = Store(s.resolve(s.store.path))
    for g in st.all_glossary():
        print(f"{g.src_lang:>5} {g.src_term}  →  {g.tgt_lang} {g.tgt_term}   [{g.kind}]")


@app.command(help=_("cli.doctor.help"))
def doctor(ctx: typer.Context) -> None:
    # Pre-flight check before taking anything to the device. Do not guess at the
    # environment; confirm it here.
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
        print("            " + _("cli.doctor.audio_offbox"))
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
        print("\n" + _("cli.doctor.services"))
        for role in group.all():
            for client in filter(None, (role.preferred, role.fallback)):
                h = await client.health()
                state = "UP" if h.get("ok") else "DOWN"
                tag = f"{role.role}@{client.side}"
                print(f"  {tag:<20} {state:<4} {json.dumps(h, ensure_ascii=False)[:80]}")
        await group.aclose()

    asyncio.run(probe())


@app.command(help=_("cli.netcheck.help"))
def netcheck(
    ctx: typer.Context,
    samples: Annotated[int, typer.Option(help=_("cli.netcheck.opt.samples"))] = 10,
    seconds: Annotated[float, typer.Option(help=_("cli.netcheck.opt.seconds"))] = 6.0,
) -> None:
    # Every remote stage pays the round trip, and the utterance audio pays the
    # upload on top. §6 has 2.9 s total with no slack, so this gets measured
    # rather than assumed.
    import statistics

    import httpx

    from .clients.base import remote_transport_kwargs
    from .transport import encode_pcm

    s = _settings(ctx)
    if not s.remote.enabled:
        print(_("cli.netcheck.remote_disabled"))
        raise typer.Exit(code=1)

    placement = s.resolved_placement()
    remote_roles = [r for r, side in placement.items() if side == "remote"]
    if not remote_roles:
        print(_("cli.netcheck.no_remote_roles", mode=s.perf_mode))
        raise typer.Exit(code=1)

    tk = remote_transport_kwargs(s.remote)
    pcm = np.zeros(int(seconds * s.shm.sample_rate), dtype=np.float32)
    blob = encode_pcm(pcm, s.remote.audio_encoding)

    async def go() -> bool:
        print(f"perf_mode   {s.perf_mode}")
        print("placement   " + "  ".join(f"{k}={v}" for k, v in placement.items()))
        print(
            _(
                "cli.netcheck.probe",
                seconds=seconds,
                encoding=s.remote.audio_encoding,
                size=len(blob),
            )
            + "\n"
        )

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
            print("\n" + _("cli.netcheck.failed", roles=", ".join(failed)))

        # What the link adds to one turn, stage by stage.
        overhead = uploads.get("asr", rtts.get("asr", 0.0))
        if s.asr_verify.mode == "always":
            overhead += uploads.get("asr_verify", rtts.get("asr_verify", 0.0))
        overhead += rtts.get("llm", 0.0) + rtts.get("tts", 0.0)

        b = s.budget_ms
        slack = b.total - b.silence
        print("\n" + _("cli.netcheck.overhead", ms=f"{overhead:.0f}"))
        print(_("cli.netcheck.budget", ms=slack))
        print(
            _("cli.netcheck.over_budget")
            if overhead > slack * 0.25
            else _("cli.netcheck.within_budget")
        )
        return not failed

    if not asyncio.run(go()):
        raise typer.Exit(code=1)


@app.command(help=_("cli.config.help"))
def config(ctx: typer.Context) -> None:
    from .tui.config_app import ConfigApp

    ConfigApp(config_path=ctx.obj.config).run()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
