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

from .async_runtime import run as run_async
from .config import Settings, load_settings
from .i18n import _, available_locales, set_locale
from .logging_setup import setup_logging


# -- shared wiring --------------------------------------------------------
def _build(settings: Settings, wav: Path | None = None, text_only: bool = False):
    from .audio.capture import FileCapture, MicCapture, NullCapture
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

    if text_only:
        # The keyboard is the input source, so no audio device is opened.
        capture = NullCapture()
        playback = _output_or_null(settings)
    elif wav is not None:
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
    denoise_enabled = d.enabled and wav is None and not text_only
    denoiser = build_denoiser(denoise_enabled, d.backend, d.post_filter_beta)

    return Orchestrator(settings, capture, seg, playback, denoiser)


def _output_or_null(settings: Settings):
    from .audio.playback import NullPlayback, Playback

    try:
        import sounddevice as sd

        sd.check_output_settings(
            device=settings.audio.output_device,
            samplerate=settings.audio.playback_sample_rate,
            channels=1,
        )
        return Playback(settings.audio, settings.tts)
    except Exception:  # noqa: BLE001
        return NullPlayback(settings.audio, settings.tts)


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
    help=_("Consecutive four-language offline speech interpreter"),
    no_args_is_help=True,
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
glossary_app = typer.Typer(name="glossary", help=_("Manage the glossary"), no_args_is_help=True)
app.add_typer(glossary_app)
history_app = typer.Typer(
    name="history", help=_("Browse past interpretation turns"), no_args_is_help=True
)
app.add_typer(history_app)

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "-c",
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        help=_("Path to a YAML configuration file"),
    ),
]
LangOption = Annotated[
    str | None,
    typer.Option("--lang", help=_("Interface language: auto, en, ko, ja, zh-TW")),
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
@app.command(help=_("Start the terminal interface"))
def run(ctx: typer.Context) -> None:
    s = _settings(ctx)
    setup_logging(
        s.logging.level,
        s.resolve(s.logging.log_path),
        s.logging.console,
        "orch",
        terminal_interface=True,
    )
    from .tui import KotonohaApp

    run_async(KotonohaApp(_build(s)).run_async())


@app.command(help=_("Start the integrated terminal interface"))
def tui(ctx: typer.Context) -> None:
    from .tui import run_unified_tui

    run_async(run_unified_tui(ctx.obj.config, _build))


@app.command(help=_("Run the pipeline from a WAV file, without a microphone"))
def replay(
    ctx: typer.Context,
    wav: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help=_("16-bit PCM WAV file"))
    ],
    seconds: Annotated[float, typer.Option(help=_("Run duration in seconds"))] = 30.0,
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

    run_async(go())
    print(_("Turn log: {path}", path=s.resolve(s.logging.turn_log_path)))


@app.command("text", help=_("Interpret typed text without a microphone"))
def text_command(
    ctx: typer.Context,
    message: Annotated[str, typer.Argument(help=_("Text to interpret"))],
    source: Annotated[
        str | None,
        typer.Option("--from", help=_("Source language; omit to read it from the script")),
    ] = None,
    speak: Annotated[
        bool, typer.Option("--speak/--no-speak", help=_("Play the synthesized translation"))
    ] = True,
) -> None:
    # The translation chain without a microphone: the same path the interpreter
    # uses in text mode, which makes it the way to exercise translation and TTS
    # on a host that has no audio input.
    s = _settings(ctx)
    s.session.mode = "text"
    setup_logging(s.logging.level, s.resolve(s.logging.log_path), False, "text")
    orch = _build(s, text_only=True)
    if not speak:
        from .audio.playback import NullPlayback

        orch.playback = NullPlayback(s.audio, s.tts)

    async def go() -> dict | None:
        await orch.start()
        accepted = await orch.submit_text(message, src_lang=source)
        if accepted and speak:
            await orch.playback.wait_drained(timeout=s.tts.timeout_s * 4)
        record = None
        for event in orch.bus.drain_nowait(1024):
            if event.kind == "history":
                record = event.payload
        await orch.stop()
        return record

    result = run_async(go())
    if result is None:
        print(_("The turn was refused: empty input, or another turn is running"))
        raise typer.Exit(code=1)
    print(_("[{lang}] {text}", lang=result.get("src_lang"), text=result.get("source_text") or ""))
    print(
        _("[{lang}] {text}", lang=result.get("tgt_lang"), text=result.get("translation") or "")
    )


@app.command(help=_("List audio devices"))
def devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())
    print("\n" + _("Default input/output:"), sd.default.device)


@app.command(help=_("Start a model service"))
def serve(
    service: Annotated[ServiceName, typer.Argument(help=_("Service to start"))],
    host: Annotated[str, typer.Option(help=_("Bind address"))] = "0.0.0.0",
    port: Annotated[
        int | None, typer.Option(help=_("Port; defaults to the port assigned to the service"))
    ] = None,
) -> None:
    import uvicorn

    target, default_port = SERVICE_TARGETS[service]
    uvicorn.run(
        target,
        host=host,
        port=port or default_port,
        log_level="info",
        loop="uvloop",
        workers=1,
    )


@glossary_app.command("import", help=_("Load glossary and Traditional Chinese rules from YAML"))
def glossary_import(
    ctx: typer.Context,
    path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help=_("Glossary YAML file"))
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
    print(
        _(
            "{terms} terms and {rules} rules applied to {path}",
            terms=n,
            rules=m,
            path=s.resolve(s.store.path),
        )
    )


@glossary_app.command("list", help=_("List registered terms"))
def glossary_list(ctx: typer.Context) -> None:
    from .store import Store

    s = _settings(ctx)
    st = Store(s.resolve(s.store.path))
    for g in st.all_glossary():
        print(f"{g.src_lang:>5} {g.src_term}  →  {g.tgt_lang} {g.tgt_term}   [{g.kind}]")


@app.command(help=_("Report environment, role placement and service health"))
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
        print("            " + _("! In this mode utterance audio leaves the device"))
    print()

    mods = [
        ("numpy", True), ("httpx", True), ("structlog", True), ("textual", True),
        ("uvloop", True),
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

    # .mo is generated at install time, so a bare source checkout has none and the
    # interface silently falls back to English. Say so rather than leaving it to be
    # discovered.
    from .i18n import DEFAULT_LOCALE, available_locales, mo_path

    uncompiled = [
        code
        for code in available_locales()
        if code != DEFAULT_LOCALE and not mo_path(code).exists()
    ]
    if uncompiled:
        print(_("Translation catalogs not compiled: {locales}", locales=", ".join(uncompiled)))
        print(_("Run: uv run python scripts/i18n.py compile"))
    else:
        print(_("Translation catalogs compiled: {locales}", locales=", ".join(
            code for code in available_locales() if code != DEFAULT_LOCALE
        )))

    async def probe() -> None:
        from .clients import build_service_group

        group = build_service_group(s)
        print("\n" + _("Services:"))
        for role in group.all():
            for client in filter(None, (role.preferred, role.fallback)):
                h = await client.health()
                state = "UP" if h.get("ok") else "DOWN"
                tag = f"{role.role}@{client.side}"
                print(f"  {tag:<20} {state:<4} {json.dumps(h, ensure_ascii=False)[:80]}")
        await group.aclose()

    run_async(probe())


@app.command(help=_("Measure latency and throughput to the external server"))
def netcheck(
    ctx: typer.Context,
    samples: Annotated[int, typer.Option(help=_("Measurements per role"))] = 10,
    seconds: Annotated[float, typer.Option(help=_("Probe utterance length in seconds"))] = 6.0,
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
        print(_("remote.enabled is false. Use config/performance.yaml, or enable it and retry."))
        raise typer.Exit(code=1)

    placement = s.resolved_placement()
    remote_roles = [r for r, side in placement.items() if side == "remote"]
    if not remote_roles:
        print(_("No role is routed remotely under perf_mode={mode}.", mode=s.perf_mode))
        raise typer.Exit(code=1)

    tk = remote_transport_kwargs(s.remote)
    pcm = np.zeros(int(seconds * s.shm.sample_rate), dtype=np.float32)
    blob = encode_pcm(pcm, s.remote.audio_encoding)

    async def go() -> bool:
        print(f"perf_mode   {s.perf_mode}")
        print("placement   " + "  ".join(f"{k}={v}" for k, v in placement.items()))
        print(
            _("probe       {seconds}s utterance, {encoding}, {size} bytes",
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
                for _measurement_index in range(samples):
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
                for _measurement_index in range(samples):
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
            print(
                "\n"
                + _(
                    "Connection failed: {roles}. These roles fall back on-board.",
                    roles=", ".join(failed),
                )
            )

        # What the link adds to one turn, stage by stage.
        overhead = uploads.get("asr", rtts.get("asr", 0.0))
        if s.asr_verify.mode == "always":
            overhead += uploads.get("asr_verify", rtts.get("asr_verify", 0.0))
        overhead += rtts.get("llm", 0.0) + rtts.get("tts", 0.0)

        b = s.budget_ms
        slack = b.total - b.silence
        print("\n" + _("Estimated link overhead per turn  {ms} ms", ms=f"{overhead:.0f}"))
        print(_("Budget, end-of-utterance to first audio  {ms} ms", ms=slack))
        print(
            _("  ! The link consumes more than 25% of the budget. Consider hybrid mode.")
            if overhead > slack * 0.25
            else _("  The link fits within the budget. The remainder is model inference time.")
        )
        return not failed

    if not run_async(go()):
        raise typer.Exit(code=1)


@app.command(help=_("Edit the configuration in a terminal interface"))
def config(ctx: typer.Context) -> None:
    from .tui.config_app import ConfigApp

    run_async(ConfigApp(config_path=ctx.obj.config).run_async())


# -- history ----------------------------------------------------------------
SearchOption = Annotated[
    str | None, typer.Option("--search", help=_("Match text in the source or the translation"))
]
HistoryLangOption = Annotated[
    str | None, typer.Option("--lang", help=_("Filter by detected source language"))
]
OutcomeOption = Annotated[
    str | None,
    typer.Option("--outcome", help=_("Filter by turn outcome, for example ok or llm_timeout")),
]


def _open_store(settings):
    from .store import Store

    return Store(settings.resolve(settings.store.path))


@history_app.command("browse", help=_("Open the history browser"))
def history_browse(ctx: typer.Context) -> None:
    from .tui.history_app import HistoryApp

    run_async(HistoryApp(config_path=ctx.obj.config).run_async())


@history_app.command("list", help=_("Print past turns to standard output"))
def history_list(
    ctx: typer.Context,
    search: SearchOption = None,
    lang: HistoryLangOption = None,
    outcome: OutcomeOption = None,
    limit: Annotated[int, typer.Option("--limit", help=_("Maximum turns to return"))] = 20,
    full: Annotated[
        bool, typer.Option("--full", help=_("Print complete text instead of an excerpt"))
    ] = False,
) -> None:
    settings = _settings(ctx)
    store = _open_store(settings)
    try:
        entries = store.search_turns(
            query=search, src_lang=lang, outcome=outcome, limit=limit
        )
        total = store.count_turns(query=search, src_lang=lang, outcome=outcome)
    finally:
        store.close()

    if not entries:
        print(_("No turns match"))
        return

    # Oldest first when printing: a terminal is read top to bottom.
    for entry in reversed(entries):
        stamp = entry.when.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{stamp}  {entry.src_lang or '?'}→{entry.tgt_lang or '?'}  [{entry.outcome}]")
        if full:
            print(f"  {entry.source_text or ''}")
            print(f"  {entry.translation or ''}")
        else:
            print(f"  {_excerpt(entry.source_text)}")
            print(f"  {_excerpt(entry.translation)}")
    print()
    print(_("{shown} of {total} turns", shown=len(entries), total=total))


@history_app.command("export", help=_("Export past turns as JSONL"))
def history_export(
    ctx: typer.Context,
    out: Annotated[Path, typer.Argument(help=_("Destination JSONL file"))],
    search: SearchOption = None,
    lang: HistoryLangOption = None,
    outcome: OutcomeOption = None,
    limit: Annotated[int, typer.Option("--limit", help=_("Maximum turns to return"))] = 10_000,
) -> None:
    from .tui.history_app import export_jsonl

    settings = _settings(ctx)
    store = _open_store(settings)
    try:
        entries = store.search_turns(
            query=search, src_lang=lang, outcome=outcome, limit=limit
        )
    finally:
        store.close()

    if not entries:
        print(_("No turns match"))
        raise typer.Exit(code=1)
    export_jsonl(entries, out)
    print(_("Exported {count} turns to {path}", count=len(entries), path=out))


def _excerpt(value: str | None, width: int = 78) -> str:
    text = (value or "").replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> None:
    app()


if __name__ == "__main__":
    main()
