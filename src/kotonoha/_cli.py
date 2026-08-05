"""kotonoha CLI, built on Typer.

The global --config option lives on the callback and reaches each command through
`context.obj`. Commands signal failure with `typer.Exit`; nothing returns an exit code
directly.

All operator-facing text goes through `i18n._`. Typer renders command help at import
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
from typing import Annotated, Any, ClassVar, Final

import numpy as np
import typer

from kotonoha._async_runtime import run as run_async
from kotonoha._call_compatibility import keyword_compatible
from kotonoha._config import Settings, load_settings
from kotonoha._i18n import _, available_locales, set_locale
from kotonoha._logging_setup import setup_logging
from kotonoha._typer_i18n import (
    LocalizedTyperCommand,
    LocalizedTyperGroup,
    configure_typer_chrome,
)

configure_typer_chrome()


# -- shared wiring --------------------------------------------------------
def _build(
    settings: Settings,
    /,
    wave_path: Path | None = None,
    text_only: bool = False,
) -> Any:
    from kotonoha.audio._capture import FileCapture, MicCapture, NullCapture
    from kotonoha.audio._denoise import build_denoiser
    from kotonoha.audio._playback import NullPlayback, Playback
    from kotonoha.audio._vad import UtteranceSegmenter, build_vad
    from kotonoha.core._orchestrator import Orchestrator

    vad_config = settings.frontend.vad
    voice_activity_detector = build_vad(
        vad_config.backend,
        settings.resolve(vad_config.model_path),
        settings.audio.work_sample_rate,
    )
    segmenter = UtteranceSegmenter(
        vad=voice_activity_detector,
        sample_rate=settings.audio.work_sample_rate,
        threshold=vad_config.threshold,
        neg_threshold=vad_config.neg_threshold,
        preroll_ms=vad_config.preroll_ms,
        min_speech_ms=vad_config.min_speech_ms,
        silence_ms=vad_config.silence_ms,
        max_utterance_ms=vad_config.max_utterance_ms,
    )

    if text_only:
        # The keyboard is the input source, so no audio device is opened.
        capture = NullCapture()
        playback = _output_or_null(settings)
    elif wave_path is not None:
        audio_samples = load_wave_file(wave_path, settings.audio.work_sample_rate)
        capture = FileCapture(audio_samples)
        playback = NullPlayback(settings.audio, settings.tts)
    else:
        capture = MicCapture(settings.audio, vad_config)
        try:
            import sounddevice

            sounddevice.check_output_settings(
                device=settings.audio.output_device,
                samplerate=settings.audio.playback_sample_rate,
                channels=1,
            )
            playback = Playback(settings.audio, settings.tts)
        except Exception:  # noqa: BLE001
            playback = NullPlayback(settings.audio, settings.tts)

    denoise_config = settings.frontend.denoise
    denoise_enabled = denoise_config.enabled and wave_path is None and not text_only
    denoiser = build_denoiser(
        denoise_enabled,
        denoise_config.backend,
        denoise_config.post_filter_beta,
    )

    return Orchestrator(settings, capture, segmenter, playback, denoiser)


def _output_or_null(
    settings: Settings,
    /,
) -> Any:
    from kotonoha.audio._playback import NullPlayback, Playback

    try:
        import sounddevice

        sounddevice.check_output_settings(
            device=settings.audio.output_device,
            samplerate=settings.audio.playback_sample_rate,
            channels=1,
        )
        return Playback(settings.audio, settings.tts)
    except Exception:  # noqa: BLE001
        return NullPlayback(settings.audio, settings.tts)


def load_wave_file(
    path: Path,
    /,
    target_rate: int,
) -> np.ndarray:
    with wave.open(str(path), "rb") as wave_reader:
        sample_rate = wave_reader.getframerate()
        channel_count = wave_reader.getnchannels()
        sample_width = wave_reader.getsampwidth()
        raw_audio = wave_reader.readframes(wave_reader.getnframes())
    if sample_width != 2:
        raise ValueError(f"only 16-bit PCM WAV is supported: {path} ({sample_width * 8}-bit)")
    audio_samples = np.frombuffer(raw_audio, dtype="<i2").astype(np.float32) / 32768.0
    if channel_count > 1:
        audio_samples = audio_samples.reshape(-1, channel_count).mean(axis=1)
    if sample_rate != target_rate:
        from kotonoha.audio._resample import resample_once

        audio_samples = resample_once(audio_samples, sample_rate, target_rate)
    return audio_samples


# -- application ----------------------------------------------------------
@dataclass(slots=True)
class AppState:
    """Carry the global configuration path through Typer's context object."""

    config: Path | None = None


class ServiceName(str, Enum):
    __slots__: ClassVar[tuple[str, ...]] = ()
    asr: Final = "asr"
    verify: Final = "verify"
    tts: Final = "tts"


SERVICE_TARGETS = {
    ServiceName.asr: ("kotonoha.services._asr_server:app", 8001),
    ServiceName.verify: ("kotonoha.services._asr_verify_server:app", 8002),
    ServiceName.tts: ("kotonoha.services._tts_server:app", 8004),
}

app = typer.Typer(
    name="kotonoha",
    cls=LocalizedTyperGroup,
    help=_("Consecutive four-language offline speech interpreter"),
    no_args_is_help=True,
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
glossary_app = typer.Typer(
    name="glossary",
    cls=LocalizedTyperGroup,
    help=_("Manage the glossary"),
    no_args_is_help=True,
)
app.add_typer(glossary_app)
history_app = typer.Typer(
    name="history",
    cls=LocalizedTyperGroup,
    help=_("Browse past interpretation turns"),
    no_args_is_help=True,
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
@keyword_compatible
def cli(
    context: typer.Context,
    /,
    config: ConfigOption = None,
    language: LangOption = None,
) -> None:
    if language and language != "auto":
        if language not in available_locales():
            raise typer.BadParameter(f"{language} not in {', '.join(available_locales())}")
        set_locale(language)
    context.obj = AppState(config=config)


def _settings(
    context: typer.Context,
    /,
) -> Settings:
    return load_settings(context.obj.config)


# -- commands -------------------------------------------------------------
@app.command(cls=LocalizedTyperCommand, help=_("Start the terminal interface"))
@keyword_compatible
def run(
    context: typer.Context,
    /,
) -> None:
    settings = _settings(context)
    setup_logging(
        settings.logging.level,
        settings.resolve(settings.logging.log_path),
        settings.logging.console,
        "orchestrator",
        terminal_interface=True,
    )
    from kotonoha.tui._app import KotonohaApp

    run_async(KotonohaApp(_build(settings)).run_async())


@app.command(cls=LocalizedTyperCommand, help=_("Start the integrated terminal interface"))
@keyword_compatible
def tui(
    context: typer.Context,
    /,
) -> None:
    from kotonoha.tui._workflow import run_unified_tui

    run_async(run_unified_tui(context.obj.config, _build))


@app.command(
    cls=LocalizedTyperCommand,
    help=_("Run the pipeline from a WAV file, without a microphone"),
)
@keyword_compatible
def replay(
    context: typer.Context,
    /,
    wave_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help=_("16-bit PCM WAV file"))
    ],
    seconds: Annotated[float, typer.Option(help=_("Run duration in seconds"))] = 30.0,
) -> None:
    # Runs the whole pipeline from a file. This is the regression path for
    # end-of-utterance and preroll behaviour.
    settings = _settings(context)
    # There is no key to press when replaying a file, and the VAD has to do the
    # segmenting, so force automatic mode.
    settings.session.mode = "auto"
    setup_logging(
        settings.logging.level,
        settings.resolve(settings.logging.log_path),
        True,
        "replay",
    )
    orchestrator = _build(settings, wave_path=wave_file)

    async def execute() -> None:
        await orchestrator.start()
        await asyncio.sleep(seconds)
        await orchestrator.stop()

    run_async(execute())
    print(_("Turn log: {path}", path=settings.resolve(settings.logging.turn_log_path)))


@app.command(
    "text",
    cls=LocalizedTyperCommand,
    help=_("Interpret typed text without a microphone"),
)
@keyword_compatible
def text_command(
    context: typer.Context,
    /,
    message: Annotated[str, typer.Argument(help=_("Text to interpret"))],
    source: Annotated[
        str | None,
        typer.Option(
            "--from",
            help=_("Source language; omit to detect it from the writing system"),
        ),
    ] = None,
    speak: Annotated[
        bool, typer.Option("--speak/--no-speak", help=_("Play the synthesized translation"))
    ] = True,
) -> None:
    # The translation chain without a microphone: the same path the interpreter
    # uses in text mode, which makes it the way to exercise translation and TTS
    # on a host that has no audio input.
    settings = _settings(context)
    settings.session.mode = "text"
    setup_logging(
        settings.logging.level,
        settings.resolve(settings.logging.log_path),
        False,
        "text",
    )
    orchestrator = _build(settings, text_only=True)
    if not speak:
        from kotonoha.audio._playback import NullPlayback

        orchestrator.playback = NullPlayback(settings.audio, settings.tts)

    async def execute() -> dict | None:
        await orchestrator.start()
        accepted = await orchestrator.submit_text(message, src_lang=source)
        if accepted and speak:
            await orchestrator.playback.wait_drained(timeout=settings.tts.timeout_s * 4)
        record = None
        for event in orchestrator.event_bus.drain_nowait(1024):
            if event.kind == "history":
                record = event.payload
        await orchestrator.stop()
        return record

    result = run_async(execute())
    if result is None:
        print(_("The turn cannot start because the input is empty or another turn is running"))
        raise typer.Exit(code=1)
    print(_("[{lang}] {text}", lang=result.get("src_lang"), text=result.get("source_text") or ""))
    print(
        _("[{lang}] {text}", lang=result.get("tgt_lang"), text=result.get("translation") or "")
    )


@app.command(cls=LocalizedTyperCommand, help=_("List audio devices"))
@keyword_compatible
def devices() -> None:
    import sounddevice

    print(sounddevice.query_devices())
    print("\n" + _("Default input/output:"), sounddevice.default.device)


@app.command(cls=LocalizedTyperCommand, help=_("Start a model service"))
@keyword_compatible
def serve(
    service: Annotated[ServiceName, typer.Argument(help=_("Service to start"))],
    /,
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


@glossary_app.command(
    "import",
    cls=LocalizedTyperCommand,
    help=_("Load glossary and Traditional Chinese rules from YAML"),
)
@keyword_compatible
def glossary_import(
    context: typer.Context,
    /,
    path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help=_("Glossary YAML file"))
    ],
) -> None:
    import yaml

    from kotonoha.store._db import GlossaryEntry, Store

    settings = _settings(context)
    store = Store(settings.resolve(settings.store.path))
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = [GlossaryEntry(**entry) for entry in data.get("glossary", [])]
        term_count = store.upsert_glossary(entries) if entries else 0
        rules = [
            (
                rule["pattern"],
                rule["replacement"],
                bool(rule.get("is_regex", False)),
                rule.get("note"),
            )
            for rule in data.get("zh_rules", [])
        ]
        rule_count = store.upsert_zh_rules(rules) if rules else 0
    finally:
        store.close()
    print(
        _(
            "{terms} terms and {rules} rules applied to {path}",
            terms=term_count,
            rules=rule_count,
            path=settings.resolve(settings.store.path),
        )
    )


@glossary_app.command(
    "list",
    cls=LocalizedTyperCommand,
    help=_("List registered terms"),
)
@keyword_compatible
def glossary_list(
    context: typer.Context,
    /,
) -> None:
    from kotonoha.store._db import Store

    settings = _settings(context)
    store = Store(settings.resolve(settings.store.path))
    try:
        for entry in store.all_glossary():
            print(
                f"{entry.src_lang:>5} {entry.src_term}  →  "
                f"{entry.tgt_lang} {entry.tgt_term}   [{entry.kind}]"
            )
    finally:
        store.close()


@app.command(
    cls=LocalizedTyperCommand,
    help=_("Report environment, role placement and service health"),
)
@keyword_compatible
def doctor(
    context: typer.Context,
    /,
) -> None:
    # Pre-flight check before taking anything to the device. Do not guess at the
    # environment; confirm it here.
    import platform

    settings = _settings(context)
    print(f"python      {sys.version.split()[0]}  {platform.machine()}  {platform.system()}")
    print(f"config      {context.obj.config or 'config/default.yaml'}")
    print(
        f"asr backend {settings.asr.backend} / llm profile {settings.llm.profile} / "
        f"tts {settings.tts.backend}"
    )
    print(f"llm model   {settings.resolve(settings.llm.model_path)}")
    placement = settings.resolved_placement()
    print(
        f"perf_mode   {settings.perf_mode}  "
        f"remote={'on' if settings.remote.enabled else 'off'}"
    )
    print(
        "placement   "
        + "  ".join(f"{role}={side}" for role, side in placement.items())
    )
    if settings.audio_leaves_device:
        print("            " + _("! In this mode utterance audio leaves the device"))
    print()

    modules = [
        ("numpy", True), ("httpx", True), ("structlog", True), ("textual", True),
        ("uvloop", True),
        ("soxr", True), ("sounddevice", False), ("opencc", False),
        ("onnxruntime", False), ("torch", False),
        ("vllm", settings.asr.backend == "vllm"),
        ("transformers", settings.asr.backend == "transformers"),
        ("faster_whisper", False), ("df", False), ("qwen_tts", False), ("melo", False),
    ]
    for name, required in modules:
        try:
            __import__(name)
            print(f"  [ok]   {name}")
        except Exception as error:  # noqa: BLE001
            tag = "FAIL" if required else "miss"
            print(f"  [{tag}] {name}  ({type(error).__name__})")

    print()
    vad_path = settings.resolve(settings.frontend.vad.model_path)
    print(f"  silero_vad.onnx  {'ok' if vad_path.exists() else 'MISSING'}  {vad_path}")

    # .mo is generated at install time, so a bare source checkout has none and the
    # interface silently falls back to English. Say so rather than leaving it to be
    # discovered.
    from kotonoha._i18n import DEFAULT_LOCALE, available_locales, mo_path

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
        from kotonoha.clients._build import build_service_group

        group = build_service_group(settings)
        print("\n" + _("Services:"))
        for role in group.all():
            for client in filter(None, (role.preferred, role.fallback)):
                health = await client.health()
                state = "UP" if health.get("ok") else "DOWN"
                tag = f"{role.role}@{client.side}"
                details = json.dumps(health, ensure_ascii=False)[:80]
                print(f"  {tag:<20} {state:<4} {details}")
        await group.aclose()

    run_async(probe())


@app.command(
    cls=LocalizedTyperCommand,
    help=_("Measure latency and throughput to the external server"),
)
@keyword_compatible
def netcheck(
    context: typer.Context,
    /,
    samples: Annotated[int, typer.Option(help=_("Measurements per role"))] = 10,
    seconds: Annotated[float, typer.Option(help=_("Probe utterance length in seconds"))] = 6.0,
) -> None:
    # Every remote stage pays the round trip, and the utterance audio pays the
    # upload on top. §6 has 2.9 s total with no slack, so this gets measured
    # rather than assumed.
    import statistics

    import httpx

    from kotonoha._transport import encode_pcm
    from kotonoha.clients._base import remote_transport_kwargs

    settings = _settings(context)
    if not settings.remote.enabled:
        print(_("remote.enabled is false. Use config/performance.yaml, or enable it and retry."))
        raise typer.Exit(code=1)

    placement = settings.resolved_placement()
    remote_roles = [role for role, side in placement.items() if side == "remote"]
    if not remote_roles:
        print(_("No role is routed remotely under perf_mode={mode}.", mode=settings.perf_mode))
        raise typer.Exit(code=1)

    transport_options = remote_transport_kwargs(settings.remote)
    audio_samples = np.zeros(int(seconds * settings.shm.sample_rate), dtype=np.float32)
    encoded_audio = encode_pcm(audio_samples, settings.remote.audio_encoding)

    async def execute() -> bool:
        print(f"perf_mode   {settings.perf_mode}")
        print(
            "placement   "
            + "  ".join(f"{role}={side}" for role, side in placement.items())
        )
        print(
            _("probe       {seconds}s utterance, {encoding}, {size} bytes",
                seconds=seconds,
                encoding=settings.remote.audio_encoding,
                size=len(encoded_audio),
            )
            + "\n"
        )

        round_trip_times: dict[str, float] = {}
        upload_times: dict[str, float] = {}
        failed_roles: list[str] = []

        async with httpx.AsyncClient(
            headers=transport_options["headers"],
            verify=transport_options["verify"],
            timeout=httpx.Timeout(10.0, connect=transport_options["connect_timeout"]),
        ) as client:
            for role in remote_roles:
                url = settings.url_for(role, "remote").rstrip("/")
                measurements = []
                for _measurement_index in range(samples):
                    started_at = time.perf_counter()
                    try:
                        response = await client.get(f"{url}/health")
                        response.raise_for_status()
                    except Exception as error:  # noqa: BLE001
                        print(f"  {role:<11} DOWN  {error!r}")
                        failed_roles.append(role)
                        break
                    measurements.append((time.perf_counter() - started_at) * 1000)
                if not measurements:
                    continue
                median_latency = statistics.median(measurements)
                percentile_95_latency = (
                    max(measurements)
                    if len(measurements) < 20
                    else statistics.quantiles(measurements, n=20)[18]
                )
                round_trip_times[role] = median_latency
                print(
                    f"  {role:<11} UP    rtt p50 {median_latency:6.1f}ms   "
                    f"p95 {percentile_95_latency:6.1f}ms   {url}"
                )

            # Only the roles that receive audio pay the upload.
            for role in (
                role for role in ("asr", "asr_verify") if role in round_trip_times
            ):
                url = settings.url_for(role, "remote").rstrip("/")
                measurements = []
                for _measurement_index in range(samples):
                    started_at = time.perf_counter()
                    try:
                        response = await client.post(
                            f"{url}/echo",
                            files={
                                "audio": (
                                    "probe.pcm",
                                    encoded_audio,
                                    "application/octet-stream",
                                )
                            },
                        )
                        response.raise_for_status()
                    except Exception as error:  # noqa: BLE001
                        print(f"  {role:<11} upload FAILED  {error!r}")
                        break
                    measurements.append((time.perf_counter() - started_at) * 1000)
                if not measurements:
                    continue
                median_upload = statistics.median(measurements)
                upload_times[role] = median_upload
                megabytes_per_second = (len(encoded_audio) / 1e6) / (median_upload / 1000)
                print(
                    f"  {role:<11} upload median {median_upload:6.1f}ms   "
                    f"{megabytes_per_second:5.1f} MB/s"
                )

        if failed_roles:
            print(
                "\n"
                + _(
                    "Connection failed: {roles}. These roles will use onboard fallback services.",
                    roles=", ".join(failed_roles),
                )
            )

        # What the link adds to one turn, stage by stage.
        overhead = upload_times.get("asr", round_trip_times.get("asr", 0.0))
        if settings.asr_verify.mode == "always":
            overhead += upload_times.get(
                "asr_verify", round_trip_times.get("asr_verify", 0.0)
            )
        overhead += round_trip_times.get("llm", 0.0) + round_trip_times.get("tts", 0.0)

        budget = settings.budget_ms
        slack = budget.total - budget.silence
        print("\n" + _("Estimated link overhead per turn  {ms} ms", ms=f"{overhead:.0f}"))
        print(_("Budget, end-of-utterance to first audio  {ms} ms", ms=slack))
        print(
            _("  ! The link consumes more than 25% of the budget. Consider hybrid mode.")
            if overhead > slack * 0.25
            else _("  The link fits within the budget. The remainder is model inference time.")
        )
        return not failed_roles

    if not run_async(execute()):
        raise typer.Exit(code=1)


@app.command(
    cls=LocalizedTyperCommand,
    help=_("Edit the configuration in a terminal interface"),
)
@keyword_compatible
def config(
    context: typer.Context,
    /,
) -> None:
    from kotonoha.tui._config_app import ConfigApp

    run_async(ConfigApp(config_path=context.obj.config).run_async())


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


def _open_store(
    settings: Any,
    /,
) -> Any:
    from kotonoha.store._db import Store

    return Store(settings.resolve(settings.store.path))


@history_app.command(
    "browse",
    cls=LocalizedTyperCommand,
    help=_("Open the history browser"),
)
@keyword_compatible
def history_browse(
    context: typer.Context,
    /,
) -> None:
    from kotonoha.tui._history_app import HistoryApp

    run_async(HistoryApp(config_path=context.obj.config).run_async())


@history_app.command(
    "list",
    cls=LocalizedTyperCommand,
    help=_("Print past turns to standard output"),
)
@keyword_compatible
def history_list(
    context: typer.Context,
    /,
    search: SearchOption = None,
    source_language: HistoryLangOption = None,
    outcome: OutcomeOption = None,
    limit: Annotated[int, typer.Option("--limit", help=_("Maximum turns to return"))] = 20,
    full: Annotated[
        bool, typer.Option("--full", help=_("Print complete text instead of an excerpt"))
    ] = False,
) -> None:
    settings = _settings(context)
    store = _open_store(settings)
    try:
        entries = store.search_turns(
            query=search, src_lang=source_language, outcome=outcome, limit=limit
        )
        total = store.count_turns(
            query=search,
            src_lang=source_language,
            outcome=outcome,
        )
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


@history_app.command(
    "export",
    cls=LocalizedTyperCommand,
    help=_("Export past turns as JSONL"),
)
@keyword_compatible
def history_export(
    context: typer.Context,
    /,
    destination: Annotated[Path, typer.Argument(help=_("Destination JSONL file"))],
    search: SearchOption = None,
    source_language: HistoryLangOption = None,
    outcome: OutcomeOption = None,
    limit: Annotated[int, typer.Option("--limit", help=_("Maximum turns to return"))] = 10_000,
) -> None:
    from kotonoha.tui._history_app import export_jsonl

    settings = _settings(context)
    store = _open_store(settings)
    try:
        entries = store.search_turns(
            query=search, src_lang=source_language, outcome=outcome, limit=limit
        )
    finally:
        store.close()

    if not entries:
        print(_("No turns match"))
        raise typer.Exit(code=1)
    export_jsonl(entries, destination)
    print(_("Exported {count} turns to {path}", count=len(entries), path=destination))


def _excerpt(
    value: str | None,
    /,
    width: int = 78,
) -> str:
    text = (value or "").replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> None:
    app()


if __name__ == "__main__":
    main()
