# Kotonoha Interpreter

Consecutive speech interpreter for NVIDIA Jetson AGX Orin 64GB, with an optional
high-performance mode backed by an external RTX A6000 server.

## Overview

The system converts a completed utterance into speech in a target language. It is
consecutive, not simultaneous: translation begins after end-of-utterance is detected.
All inference runs on hardware owned by the operator. No cloud service is contacted.

Detailed environment preparation, installation, deployment, security, update, rollback,
and acceptance procedures are defined in
[`docs/installation-and-deployment.md`](docs/installation-and-deployment.md).

| Property | Value |
|---|---|
| Languages | Korean, English, Traditional Chinese (Taiwan), Japanese |
| Directions | 12 |
| Source language | Automatic identification, with inheritance fallback for short utterances |
| Target language | Session configuration: language pair, fixed target, or broadcast |
| Latency target | 2.9 s from end-of-utterance to first audio |
| Priority | Accuracy over latency |
| User interface | Terminal UI (Textual), localized in English, Korean, Japanese and Traditional Chinese |

## Status

Phase 0 validation spikes have not been executed. They require the target hardware and
determine three architectural decisions.

| Decision | Setting | Spike | Current default |
|---|---|---|---|
| ASR runtime | `asr.backend` | 1 | `transformers` |
| TTS backend | `tts.backend` | 2 | `melo` |
| Translation model class | `llm.profile` | 3 | `dense` |

Each decision is expressed as configuration, not code. `spikes/report.py` converts spike
output into `config/local.yaml`, which is layered over the defaults without code changes.

`VllmBackend` in `src/kotonoha/services/asr_server.py` raises `NotImplementedError`.
Whether vLLM loads Qwen3-ASR on sm_87 and whether it exposes N-best output is the question
Spike 1 answers. Implementing it beforehand would invalidate the latency budget.

Verified on a macOS development workstation: 201 unit tests pass, `ruff` reports no
findings, and end-to-end smoke runs against mock services exercise the on-board path, the
remote upload path, and link failover. These runs validate control flow and
instrumentation. They do not measure model inference time.

## Requirements

### Hardware

| Component | Specification | Constraint |
|---|---|---|
| Compute module | Jetson AGX Orin 64GB | Unified memory; model size is not the limiting factor |
| Memory bandwidth | 204.8 GB/s | Primary bottleneck |
| GPU architecture | sm_87 | Excludes kernels built only for sm_80 or sm_86 |
| Instruction set | aarch64 | Excludes x86-only wheels and AVX-dependent libraries |
| External accelerator (optional) | RTX A6000, 48 GB, sm_86 | Used by the high-performance mode |

Set MAXN power mode and lock clocks before any measurement. Verify with `jtop` that
thermal throttling did not occur during the measurement window.

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Software

| Component | Version | Notes |
|---|---|---|
| JetPack | 6.2 | L4T r36.4.x |
| CUDA | 12.6 | |
| Python | 3.10 on device, 3.12 on workstation | Source targets 3.10 syntax |
| Package manager | uv 0.12 | `uv.lock` is committed |
| Async event loop | uvloop 0.22.1 | Required by the orchestrator, TUI, CLI probes, and Python services |
| Base images | `dustynv/*:r36.4.0` | Verified combination; do not upgrade without revalidation |
| vLLM container | `ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04` | Spike 1 only |

## Architecture

### Process topology

The audio frontend and orchestrator run on the Orin. Model services are resident
processes. No model is loaded per request.

```
[microphone 48 kHz]
   |
   v  sounddevice
Audio frontend (CPU)              audio/capture.py, denoise.py, vad.py
  DeepFilterNet3 (48 kHz)
  Silero VAD, 300 ms preroll
  End-of-utterance: 800 ms silence
   |
   v  shared-memory ring          shmring.py
Orchestrator (asyncio/uvloop)     core/orchestrator.py
  State machine, language routing, quality gate
   |
   +--> :8001 asr          Qwen3-ASR 1.7B, N-best 5, language identification
   +--> :8002 asr-verify   faster-whisper large-v3, conditional
   +--> :8003 llm          llama.cpp server, single-pass correction and translation
   +--> :8004 tts          Qwen3-TTS 0.6B or MeloTTS
   |
   v
[speaker + terminal UI]
```

Utterance audio is transferred through a shared-memory ring buffer. Services receive a
reference of the form `{name, slot, seq, frames}` as JSON. Base64 transport of a
six-second PCM buffer costs 100-200 ms per turn and is not used.

Ring geometry is 8 slots of 30 s at 16 kHz float32, 15.4 MB total. A consumer re-reads the
slot descriptor after reading and raises `StaleSlotError` when the sequence number
changed, rather than returning audio from a later utterance.

### Turn workflow

| Step | Component | Output |
|---|---|---|
| 1 | VAD segmenter | Utterance PCM including preroll |
| 2 | DeepFilterNet3 | Noise-suppressed 48 kHz audio, resampled to 16 kHz |
| 3 | Shared-memory ring | `AudioRef` |
| 4 | Primary ASR | Five hypotheses with average log-probability, language tag |
| 5 | Language decision | Language code and provenance, `lid` or `inherited` |
| 6 | Quality gate | Cross-verification decision |
| 7 | Cross-verification ASR | Second hypothesis, conditional |
| 8 | Translation LLM | Streamed translation, then reconstructed source |
| 9 | Clause streamer | Clause boundaries |
| 10 | TTS | PCM chunks |
| 11 | Playback | Audio output, five instrumentation marks |

State machine:

| State | Action | Transition |
|---|---|---|
| `IDLE` | VAD active, microphone open | Speech detected, to `LISTENING`; typed input, to `PROCESSING` |
| `LISTENING` | Accumulate audio including preroll | 800 ms silence, to `PROCESSING` |
| `PROCESSING` | ASR, language identification, translation | First clause, to `SPEAKING` |
| `SPEAKING` | Play TTS queue, microphone closed | Queue drained, to `IDLE` |

Half-duplex gating occurs in `Orchestrator._on_state_change` and nowhere else.
`MicCapture.close_gate()` discards queued blocks and resets resampler and VAD state.
Without this gating, TTS output re-enters the microphone, is detected as a new utterance,
and produces an unbounded loop.

## Design constraints

The following are requirements, not optimization targets.

| Constraint | Implementation | Rationale |
|---|---|---|
| VAD preroll 200-300 ms | `audio/vad.py` | Korean tense-stop onsets and the pause preceding a Japanese sokuon are clipped without it. The failure presents as an ASR quality defect. |
| ASR N-best 5 | `services/asr_server.py` | Consecutive operation removes the reason to decode greedily. |
| Single-pass correction and translation | `prompts/translate.py` | Separated stages allow the translation stage to amplify correction-stage errors. |
| Clause-level streaming handoff | `core/clauses.py` | Requires 5 tok/s or higher; approximately 4-5 tokens are consumed per second of speech. |
| Conditional cross-verification | `core/quality.py` | Unconditional invocation adds 0.8 s per turn on the Orin. |

Preroll length is computed with `math.ceil` plus one frame slot. The additional slot
accounts for the frame that triggered speech onset, which belongs to the utterance rather
than to the preroll. `tests/test_vad_segmenter.py` asserts that preroll frames are present
at the head of the emitted utterance.

Language-specific handling:

- Traditional Chinese: the ASR context prompt is written in Traditional characters, and
  OpenCC `s2twp` is applied to both ASR output and translation output. Residual Mainland
  vocabulary is corrected through the `zh_rules` table.
- Short utterances: below `asr.lid.min_duration_s` (1.0 s) or below
  `asr.lid.min_confidence` (0.60), the previously detected language is inherited and the
  provenance is displayed.
- Translation is direct between source and target. English pivoting is not used.

## Components

| Path | Responsibility |
|---|---|
| `src/kotonoha/config.py` | Layered configuration, role placement resolution |
| `src/kotonoha/config_store.py` | Atomic configuration validation and persistence |
| `src/kotonoha/shmring.py` | Shared-memory audio ring buffer |
| `src/kotonoha/transport.py` | Audio payload abstraction, PCM encoding |
| `src/kotonoha/metrics.py` | Five-point turn instrumentation, budget comparison |
| `src/kotonoha/audio/` | Capture, noise suppression, VAD segmentation, playback |
| `src/kotonoha/core/` | State machine, language routing, quality gate, clause streaming, orchestrator |
| `src/kotonoha/clients/` | Service clients, placement router, failover |
| `src/kotonoha/services/` | Resident model servers, bearer-token middleware, remote configuration API |
| `src/kotonoha/prompts/` | ASR context biasing, single-pass translation prompt |
| `src/kotonoha/store/` | SQLite glossary, turn history, Traditional Chinese rules |
| `src/kotonoha/tui/` | Terminal interface, configuration editor, history browser |
| `src/kotonoha/i18n.py` | Locale resolution and message lookup |
| `src/kotonoha/locale/` | gettext catalogs, `.po` and compiled `.mo` |
| `scripts/i18n.py` | Catalog extraction, update, compilation and checks |
| `spikes/` | Phase 0 validation harness |
| `eval/` | Evaluation set recording and scoring |

## High-performance mode

### Modes

The Orin cannot be assumed to sustain 5 tok/s on a 30B MoE at Q4_K_M. An external
RTX A6000 removes that constraint. Three modes are defined, separated by whether utterance
audio leaves the device.

| Mode | Orin | RTX A6000 | Audio leaves device |
|---|---|---|---|
| `onboard` | All roles | None | No |
| `hybrid` | Audio frontend, ASR, verification, TTS | LLM | No |
| `remote` | Audio frontend | ASR, verification, LLM, TTS | Yes |

`hybrid` moves the largest latency contributor while transferring text only. It preserves
the self-contained property of the device.

`remote` produces lower total latency when the link is adequate. Utterance audio crosses
the network in this mode. The condition is logged at startup and displayed in the terminal
UI status bar.

Per-role overrides are available through the `placement` mapping and take precedence over
`perf_mode`. When `remote.enabled` is `false`, all roles resolve to `local` regardless of
`perf_mode`. A mode pointing at an unreachable host would otherwise produce a per-turn
timeout.

### Audio transport

Shared memory does not cross host boundaries. Remote ASR and verification receive audio as
a multipart binary part on `POST /transcribe/upload`. Base64 encoding is not used.

| Encoding | 6 s utterance at 16 kHz | Transfer time at 1 Gb/s |
|---|---|---|
| `f32le` | 384,000 bytes | 3.2 ms |
| `s16le` | 192,000 bytes | 1.6 ms |

`s16le` is the default. Quantization to 16-bit is inaudible at 16 kHz and does not affect
ASR output. `AudioPayload` carries both the shared-memory reference and the PCM buffer;
the client selects according to its side. The orchestrator does not branch on placement.

TTS negotiates the return encoding. The client requests `s16le` when remote and trusts the
`X-Encoding` response header over its own request, so that a service which ignores the
parameter cannot corrupt the stream.

### Failover

Every remote role retains a loaded on-board counterpart.

| Condition | Behavior |
|---|---|
| Transport failure on a unary call | The call is retried once on the on-board service. The turn completes. |
| `remote.failover_after` consecutive failures | The role is marked degraded and routed on-board. |
| Remote healthy for `remote.recover_after_s` | The role returns to remote. |
| Transport failure on a stream, before the first chunk | The stream is restarted on the on-board service. |
| Transport failure on a stream, after the first chunk | Reported as an error. No rewind is attempted. |
| Application error, 4xx | Not counted as a transport failure and not retried elsewhere. |

Successful calls reset the failure counter, so isolated failures do not accumulate into a
placement change. Turn records include `placement` and `failovers`. Without them, a turn
served by the on-board fallback is indistinguishable from one served by the A6000.

### Configuration

`config/performance.yaml` is a complete overlay applied over `config/default.yaml`.

```yaml
perf_mode: remote

remote:
  enabled: true
  services:
    asr: http://a6000.lan:8001
    asr_verify: http://a6000.lan:8002
    llm: http://a6000.lan:8003
    tts: http://a6000.lan:8004
  token: null
  verify_tls: true
  failover_after: 2
  recover_after_s: 30.0
  audio_encoding: s16le

llm:
  profile: moe
  n_ctx: 4096
  max_tokens: 768

asr:
  backend: transformers
  n_best: 5

asr_verify:
  mode: always
  compute_type: float16

tts:
  backend: qwen3
```

Differences from the on-board configuration:

| Setting | On-board | RTX A6000 | Reason |
|---|---|---|---|
| `llm.profile` | `dense` | `moe` | 48 GB holds the 30B MoE at Q4_K_M |
| `llm.n_ctx` | 2048 | 4096 | Six turns of history plus glossary |
| `asr_verify.mode` | `conditional` | `always` | Verification cost no longer justifies gating |
| `asr_verify.compute_type` | `int8_float16` | `float16` | Quantization was a bandwidth concession |
| `tts.backend` | `melo` | `qwen3` | flash-attn builds for sm_86 |

VAD preroll, the 800 ms end-of-utterance threshold, and clause streaming thresholds are
unchanged. Those are frontend properties and the frontend remains on the Orin.

`asr.backend` remains `transformers` in the overlay. sm_86 is a supported vLLM target, but
the value is changed only after Spike 1 is executed on the A6000 itself.

### Deployment

The commands below are a summary. Use the complete procedure in
[`docs/installation-and-deployment.md`](docs/installation-and-deployment.md) for model
paths, authentication, health checks, restart scope, and deployment acceptance.

On the RTX A6000 host:

```bash
export KOTONOHA_SERVICE_TOKEN=$(openssl rand -hex 32)
docker compose -f docker/compose.remote.yaml up -d
```

`docker/compose.remote.yaml` does not set `ipc: host`. Services in this deployment receive
audio over HTTP. Service configuration is `config/remote-server.yaml`.

On the Orin:

```bash
export KOTONOHA__REMOTE__TOKEN=<value from the A6000 host>
uv run kotonoha -c config/performance.yaml netcheck
uv run kotonoha -c config/performance.yaml run
```

`netcheck` reports per-role round-trip time and measured upload throughput, then compares
aggregate link overhead against the latency budget. Overhead above 25 % of the budget
indicates that `hybrid` is the appropriate mode. Output against a loopback mock service:

```
perf_mode   remote
placement   asr=remote  asr_verify=remote  llm=remote  tts=remote
probe       6.0s utterance, s16le, 192000 bytes

  asr         UP    rtt p50    1.0ms   p95    5.0ms   http://127.0.0.1:8099
  asr         upload median    1.0ms   192.4 MB/s
```

### Security considerations

Services on the A6000 listen on a routable interface. An unauthenticated `/transcribe`
endpoint is an open transcription service for any host that can reach it.

- Setting `KOTONOHA_SERVICE_TOKEN` on a service enables bearer-token enforcement.
  `/health`, `/docs`, and `/openapi.json` remain open so that health checks and `netcheck`
  function.
- Token comparison uses `hmac.compare_digest`.
- When the variable is unset, the service logs `auth.disabled` at startup. Absence of
  authentication is never silent.
- `remote.verify_tls` and `remote.ca_bundle` control certificate validation on the client.
- The llama.cpp service on port 8003 does not use the project FastAPI authentication
  middleware. Restrict the port to the Jetson or protect it with an authenticated reverse
  proxy.

Jetson services also use host networking and bind to `0.0.0.0`. The default Jetson
Compose environment does not enable bearer authentication. Restrict ports 8001-8004 at
the Jetson host firewall.

The mechanism is a shared secret on a trusted network. It does not replace network
isolation of the A6000 host.

In `remote` mode, utterance audio is transmitted to a second host. Where that is not
acceptable, `hybrid` provides the majority of the latency benefit with text-only transfer.

## Configuration

Three layers are merged in order. Each layer overrides the previous one.

| Order | Source | Purpose |
|---|---|---|
| 1 | `config/default.yaml` | Complete baseline |
| 2 | File passed to `--config` or `KOTONOHA_CONFIG` | Overlay stating only differences |
| 3 | `config/local.yaml` | Host-specific values and Phase 0 results |

Environment variables override all three. Nested keys use a double underscore.

```bash
KOTONOHA__PERF_MODE=hybrid KOTONOHA__REMOTE__ENABLED=true kotonoha run
```

`Settings.settings_customise_sources` reorders the pydantic-settings sources so that
environment variables outrank the loaded YAML. The default ordering places initialization
values first, which silently discards these overrides.

## Installation

This section provides the minimal installation path. Use
[`docs/installation-and-deployment.md`](docs/installation-and-deployment.md) for the
complete workstation, Jetson, and A6000 runbook.

Quick deployment after model artifacts are present:

```bash
# Jetson AGX Orin
bash scripts/deploy.sh jetson

# RTX A6000 host
bash scripts/deploy.sh a6000

# Remove containers and the Compose network while preserving data
bash scripts/deploy.sh uninstall jetson
bash scripts/deploy.sh uninstall a6000
```

The script validates the host and model layout, preserves existing local configuration,
builds and starts resident services, and waits for model health checks. It does not start
the interactive orchestrator. When the Docker socket requires root privileges, the
script automatically uses `sudo docker`; do not run the complete script with `sudo`.
Add `--remove-images` to an uninstall command to remove project-built images. Models,
configuration, secrets, logs, and SQLite data remain preserved.

Dependencies are managed with uv. `uv.lock` is committed so that the development
workstation and the target device resolve identical versions. Lock environments are
restricted to `darwin-arm64`, `linux-aarch64`, and `linux-x86_64`. The x86_64 environment
is required by the A6000 service images.

### Development workstation

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run kotonoha doctor
```

`uv sync` installs the `dev` group. Evaluation tooling is installed separately with
`uv sync --group eval`. The `device` extra targets aarch64 with CUDA and must not be
installed on the workstation.

### Jetson AGX Orin

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
bash scripts/fetch_models.sh
docker compose -f docker/compose.yaml up -d asr asr-verify llm tts
docker compose -f docker/compose.yaml run --rm orchestrator
```

Container images install into the base image system Python with `UV_SYSTEM_PYTHON=1`. A
project virtual environment would shadow the CUDA build of PyTorch present in the base
image. Runtime dependencies are installed from `uv export --frozen`. Four packages are
installed outside the lock file because their aarch64 resolution differs from the
development workstation: `onnxruntime`, `deepfilternet`, `qwen-tts`, `melotts`. Each image
prints `torch.version.cuda` at the end of the build, so replacement of the CUDA build by a
PyPI CPU build is detected at build time rather than on the device.

### Model artifacts

| Repository | Artifact | Size |
|---|---|---|
| `Qwen/Qwen3-ASR-1.7B-hf` | Full repository | 4.7 GB |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Full repository | Required only if Spike 2 succeeds |
| `Systran/faster-whisper-large-v3` | Full repository | |
| `unsloth/Qwen3-14B-GGUF` | `Qwen3-14B-Q4_K_M.gguf` | 9 GB |
| `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` | `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` | 18.6 GB |
| snakers4/silero-vad | `silero_vad.onnx` | 2 MB |

`scripts/fetch_models.sh` downloads all artifacts. Repository identifiers were confirmed
by lookup in 2026-08.

## Operations

### Command line

| Command | Function |
|---|---|
| `kotonoha tui` | Start the integrated control center for interpretation, configuration, and operations |
| `kotonoha run` | Start the interpreter directly |
| `kotonoha doctor` | Report environment, role placement, and service health |
| `kotonoha config` | Edit the configuration in a terminal interface |
| `kotonoha history browse` | Search past turns in a terminal interface |
| `kotonoha history list` | Print past turns to standard output |
| `kotonoha history export <path>` | Export past turns as JSONL |
| `kotonoha text "<utterance>"` | Interpret typed text without a microphone |
| `kotonoha netcheck` | Measure link latency and throughput to the A6000 |
| `kotonoha devices` | List audio devices |
| `kotonoha replay <wav>` | Run the pipeline from a WAV file without a microphone |
| `kotonoha glossary import <yaml>` | Load glossary and Traditional Chinese rules |
| `kotonoha serve <asr\|verify\|tts>` | Start one service without Docker |

Global options: `-c/--config` selects a configuration overlay, `--lang` selects the
interface language for command output.

`kotonoha replay` forces automatic mode, because no key is available to signal
push-to-talk. It is the regression path for end-of-utterance and preroll behavior.

The interface is built on Typer. Every command accepts `-h` or `--help`, arguments are
validated before the command body runs, and shell completion is installed with
`kotonoha --install-completion`.

`kotonoha tui` is the primary operator entry point. The control center opens the
interpreter, the local and remote configuration editor, the operations screen, or the
license screen. The operations screen exposes replay, device discovery, service startup,
diagnostics, link measurement, glossary management, and shell completion with the same
arguments and validation as the CLI. It streams child-process output and can terminate
long-running replay and service processes. The license screen displays the packaged MIT
license and the version and declared license of each installed direct dependency. Model
artifacts are identified as separately licensed. Exiting any screen returns to the
control center. Settings saved in the editor are reloaded before the interpreter starts.

Prefix commands with `uv run` on the development workstation. Inside containers the
package is installed into the system Python and the prefix is unnecessary.

Top-level asynchronous command handlers and all Textual applications run through
`uvloop.run()`. Uvicorn services set `--loop uvloop` explicitly. Startup fails when
uvloop is unavailable instead of silently selecting the standard asyncio event loop.

### Terminal UI

| Key | Function |
|---|---|
| `space` | Start or stop speaking |
| `a` | Toggle push-to-talk and automatic mode |
| `r` | Cycle target routing mode |
| `c` | Clear transcript panels |
| `q` | Exit |

Push-to-talk is a toggle because terminals do not deliver key-release events. Preroll
remains active in push-to-talk mode.

The status bar reports state, microphone gating, performance mode, and whether utterance
audio leaves the device. The lower log panel receives the same
structured JSON records written to `logging.log_path` and renders them as timestamp,
level, service, event, and key-value fields. `logging.console=true` enables this panel in
TUI processes and remains the default. TUI processes do not write raw JSON to the
terminal because it would corrupt the Textual display. Model service processes continue
to emit raw JSON to their container consoles. The service panel reports the side serving
each role and whether it is degraded.

The interpreter uses a configurable frame scheduler, `ui.refresh_hz`, with a default of
60 Hz. Audio levels use time-based attack and release interpolation over a 65-step meter.
Translation deltas and event bursts are coalesced before each Textual refresh batch, so a
fast model cannot schedule one terminal repaint per token. Idle scheduler ticks do not
repaint the terminal.

### Instrumentation

Each turn appends one JSON object to `data/logs/turns.jsonl`.

```
EOU detected -> ASR complete -> first clause -> first audio packet -> queue drained
```

Recorded fields: detected language, language provenance, language-identification
confidence, ASR average log-probability, cross-verification invocation and divergence,
input audio duration, output token count, tokens per second, performance mode, role
placement, failover count, and per-stage budget overruns.

Application logs are written to `data/logs/kotonoha.jsonl`. The files are separate so that
the turn log can be parsed without filtering.

Latency budget:

| Stage | Target |
|---|---|
| Silence wait | 800 ms |
| Frontend | 100 ms |
| ASR, N-best 5 | 900 ms |
| Cross-verification, conditional average | 100 ms |
| Correction and translation, first clause | 700 ms |
| TTS, first packet | 300 ms |
| End-of-utterance to first audio | 2,900 ms |

Stages exceeding their allocation appear in `over_budget_ms` with the overrun in
milliseconds. The terminal UI displays measured values against targets.

### Failure handling

| Condition | Response |
|---|---|
| Language-identification confidence below threshold, or utterance shorter than 1.0 s | Inherit previous language, display provenance |
| Empty ASR result | Treat as silence, return to `IDLE`, play nothing |
| LLM first clause exceeds 3 s | Display transcript, skip TTS |
| Jetson TTS failure | MeloTTS fallback inside the service |
| Remote TTS failure | Retry against the resident Jetson MeloTTS service before the first audio chunk |
| Remote transport failure | Retry on-board, then degrade the role |
| Unhandled exception during a turn | Log, emit UI error, force `IDLE` |

## Localization

Standard gettext, with the English source string as the message id.

| Locale | Code | Catalog |
|---|---|---|
| English | `en` | None; message ids are the English text |
| Korean | `ko` | `src/kotonoha/locale/ko/LC_MESSAGES/kotonoha.po` |
| Japanese | `ja` | `src/kotonoha/locale/ja/LC_MESSAGES/kotonoha.po` |
| Traditional Chinese (Taiwan) | `zh-TW` | `src/kotonoha/locale/zh_TW/LC_MESSAGES/kotonoha.po` |

An untranslated string falls through to its message id, so a gap degrades to readable
English rather than a symbolic key.

Resolution order, highest first:

| Order | Source |
|---|---|
| 1 | `KOTONOHA_LANG` |
| 2 | `ui.language` in the configuration, when not `auto` |
| 3 | `LC_ALL`, `LC_MESSAGES`, or `LANG` |
| 4 | English |

```bash
KOTONOHA_LANG=ja kotonoha --help     # help screens and command output
kotonoha --lang ja doctor            # command output only
```

Typer renders command help at import time, before `--lang` is parsed. `KOTONOHA_LANG`
and `ui.language` therefore affect help screens; `--lang` affects command output.

### Maintaining the catalogs

```bash
uv run python scripts/i18n.py extract    # rebuild the .pot template from source
uv run python scripts/i18n.py update     # merge new strings into each .po
uv run python scripts/i18n.py compile    # regenerate the .mo files
uv run python scripts/i18n.py check      # report untranslated, fuzzy and stale
```

`.po` is committed and is the source of truth. `.mo` is **compiled at install time** by
the build hook in `hatch_build.py` and is not committed, so the repository holds no
generated artifact that can fall out of step with its source. The hook runs for every
hatchling target, editable installs included, which is how the Jetson containers get
their catalogs — they install with `uv pip install --system --no-deps -e .` and build
isolation supplies Babel.

A source checkout that has not been installed has no `.mo` and falls back to English.
`kotonoha doctor` reports that, and `scripts/i18n.py compile` fixes it without a
reinstall.

`tests/test_i18n.py` runs the extractor and fails when a string is missing or
untranslated in any catalog, when a translation's `{placeholders}` differ from the message
id, or when an entry is fuzzy. The suite compiles the catalogs itself, so it never asserts
against a stale artifact.

Editing with Poedit, Weblate or any gettext tool works, because the files are ordinary
`.po`. Babel is a development dependency only; the runtime uses the standard library's
`gettext`.

## Text input mode

Utterances can come from the keyboard instead of the microphone. The typed path skips
capture, ASR and cross-verification, and rejoins the spoken path at routing, so
translation, TTS, instrumentation and history are identical.

Uses: interpreting for someone who prefers to type, working in an environment too loud
for the VAD, and exercising translation and TTS on a host with no audio input.

| Session mode | Input |
|---|---|
| `push_to_talk` | Microphone, space bar starts and stops |
| `auto` | Microphone, VAD segments |
| `text` | Keyboard, microphone gate closed |

In the interpreter:

| Key | Action |
|---|---|
| `t` | Enter text mode and focus the field |
| `enter` | Submit the utterance |
| `escape` | Leave text mode |
| `a` | Cycle push_to_talk, auto, text |

The microphone is closed while typing. In `auto` mode the VAD would otherwise segment
room noise into a turn while the operator is still composing, and in `push_to_talk` the
space bar belongs to the field.

### Source language

There is no acoustic language identification for typed text, so the script decides:

| Evidence | Language |
|---|---|
| Hangul present | `ko` |
| Kana present | `ja` |
| Han without kana | `zh-TW` |
| Latin | `en` |

Han without kana is read as Chinese. Kanji-only Japanese occurs but not in conversational
input, and `session.text_source_language` overrides the detection when it does. When the
script is inconclusive — punctuation and digits only — the previous language is inherited,
the same rule §5 applies to a short spoken utterance.

### Command line

```bash
kotonoha text "다음 주 화요일까지 보내주세요"
kotonoha text "Send it by Tuesday" --from en --no-speak
```

Turn records carry `input_mode` as `voice` or `text`. For a typed turn `audio_seconds` is
null and the ASR stage measures zero, so latency figures from the two paths stay
distinguishable.

## Interpretation history

Every turn is persisted to the `turns` table in `data/kotonoha.db`, including the
diagnostics that explain it: language provenance, language-identification confidence,
ASR average log-probability, whether cross-verification ran, and the outcome.

### In the interpreter

The live panes are cleared at the start of each utterance, so a third column holds
completed turns. It is seeded from the database on start and appended to as turns finish,
which means the preceding exchanges are still on screen after a restart.

| Setting | Effect |
|---|---|
| `ui.history_turns` | Turns kept in the panel. `0` hides it. Default 20. |
| `h` | Toggle the panel, to give the live panes the width |

The panel appends from the event the orchestrator emits when it persists a turn, not from
a query. A query per turn would run inside the latency budget.

### Browser

```bash
kotonoha history browse
```

| Key | Action |
|---|---|
| `/` | Focus the search field |
| `n`, `p` | Next and previous page |
| `e` | Export the rows currently listed, filters included |
| `r` | Reload |
| `escape` | Back to the control center |

Search matches the source text and the translation, because an operator recalls whichever
side they were reading. Filters for source language and outcome compose with it. Results
are paged at 200 rows with `LIMIT` applied in SQL, so table size does not grow with the
archive.

Selecting a row shows the full text and the diagnostic columns for that turn.

### Command line

```bash
kotonoha history list --search software --lang ko --limit 20
kotonoha history list --full
kotonoha history export ./data/exports/session.jsonl --outcome ok
```

Exports are JSONL, one turn per line, with an ISO timestamp added alongside the stored
epoch value.

## Configuration editor

```bash
kotonoha config
```

The target selector switches between the local device and the remote A6000. Local edits
are written to `config/local.yaml`. Remote edits are sent to the authenticated management
API on the remote ASR service and written to `config/remote-server.local.yaml` on that
host. Baseline and overlay files are never modified.

| Key | Action |
|---|---|
| `s` | Validate and save |
| `r` | Reload from disk |
| `m` | Focus the category menu |
| `q` | Exit |

The editor reflects every leaf in the `Settings` pydantic model. The current schema
contains 105 editable values in 13 categories: Interface, Session, Audio, Frontend,
Runtime services, External server, Primary ASR, Verification ASR, Translation LLM,
Speech synthesis, Language processing, Context and storage, and Observability.

The local target exposes the complete schema. The remote target exposes only settings
consumed by the resident ASR, verification ASR, LLM and TTS processes. The remote API
returns this capability list and rejects client-owned paths, including credentials,
routing policy, audio devices and local storage.

The right panel contains the fields for the selected category. Switching categories
keeps unsaved widget values in memory. A count beside a category identifies values
already present in the active override file. Lists and mappings use YAML flow syntax,
for example `[ko, en]` and `{llm: remote}`.

Each field shows the dotted configuration path, the effective value, and a localized
description. Paths are not translated: they are what has to be typed into a YAML file.
A field already present in `local.yaml` is marked as modified.

A candidate configuration is validated by constructing `Settings` from the same layer
order the runtime uses. Nothing is written unless that succeeds, so the editor cannot
leave a device with a configuration that fails to load. Constraints declared on the model
are enforced here as well; `frontend.vad.preroll_ms` below 200 ms is rejected.

Remote access uses `remote.services.asr`, `remote.token`, `remote.verify_tls` and
`remote.ca_bundle` from the local client configuration. The `/admin/config` endpoint is
covered by the same Bearer middleware as transcription endpoints. It is unavailable
without a valid `KOTONOHA_SERVICE_TOKEN` when authentication is enabled.

The remote management API also writes `config/remote-llm.env`. `run_llm.sh` sources that
file so changes to the LLM profile, model file, context size, batch size, models directory
and GPU-layer count apply after the remote stack restarts. The API does not reload
resident models in the request path.

Saved local values take effect on the next interpreter start. Saved remote values take
effect after restarting the remote services.

## Evaluation

The evaluation set is constructed in parallel with Phase 1. Without it, subsequent tuning
depends on subjective assessment and regressions are not detected.

```bash
uv run eval/record_set.py --lang ko --prompts eval/prompts/ko.txt --out eval/data/ko
uv run eval/run_asr.py    --manifest eval/data/ko/manifest.jsonl --out eval/out/ko.hyp.jsonl
uv run eval/score_cer.py  --manifest eval/data/ko/manifest.jsonl --hyp eval/out/ko.hyp.jsonl
uv run --group eval eval/score_comet.py --hyp eval/out/ko2en.jsonl
```

Requirements:

- 100 utterances per language, recorded with the microphone and in the acoustic
  environment used in deployment.
- Reference transcripts and reference translations.
- ASR scored with character error rate using `jiwer`. Word error rate is not comparable
  across Korean, Japanese, and Chinese because word boundaries are defined differently.
- Translation scored with COMET using `unbabel-comet`. BLEU correlates poorly with Korean
  and Japanese quality and is not used.
- COMET executes on the development workstation. `eval/score_comet.py` refuses to run on
  aarch64 unless `--force-on-device` is supplied.

## Limitations

| Item | Status |
|---|---|
| Phase 0 measurements | Not executed. ASR runtime, TTS backend, and LLM profile are unconfirmed. |
| vLLM ASR backend | Not implemented. Blocked on Spike 1. |
| Noise suppression | Applied per utterance, not per frame. Frame-level processing requires the libdf frame API. Measured duration is recorded in `notes.denoise_ms` and compared against the 100 ms frontend budget. |
| Language-identification confidence | Derived from agreement among the five hypotheses, because the model does not expose a language probability. Correlation with actual accuracy is unverified. |
| TTS streaming | The Qwen3-TTS model card documents no streaming API. Synthesis is per clause, so first-packet latency equals single-clause synthesis time. |
| aarch64 package support | `onnxruntime`, `deepfilternet`, and `ctranslate2` are unverified on the target. `kotonoha doctor` reports availability. |
| Container base image tags | Placeholders. Confirm with `jetson-containers` `autotag` and record in `.env`. |
| Push-to-talk | Implemented as a toggle. Terminals do not report key release. |

## Phase plan

| Phase | Scope | State |
|---|---|---|
| 0 | Validation spikes | Pending hardware |
| 1 | English-Korean minimal path, evaluation set construction | Not started |
| 2 | Clause streaming chain, 3 s first-audio verification | Not started |
| 3 | Gating, state machine, complete failure handling | Not started |
| 4 | Four-language expansion, Traditional Chinese post-processing, routing modes | Not started |
| 5 | Accuracy tuning: N-best correction, conditional verification, six-turn context, back-translation checks | Not started |
| — | High-performance mode with external RTX A6000 | Implemented |

## Out of scope

- Cloud APIs for translation, ASR, or TTS
- Simultaneous interpretation policies such as AlignAtt or LocalAgreement
- Vector databases and embedding models; the glossary is injected as a prompt prefix
- English-pivot translation
- Browser-based microphone capture; gating and preroll control require direct device access
- Per-request model loading
- Unvalidated JetPack, CUDA, or base image upgrades

Accuracy work proceeds in the order: frontend, prompt and context, N-best correction,
model size.

## References

- `docs/installation-and-deployment.md` — installation and deployment runbook
- `spikes/README.md` — Phase 0 procedure and acceptance criteria
- `config/default.yaml` — complete configuration baseline
- `config/performance.yaml` — high-performance mode overlay
- `config/remote-server.yaml` — service configuration for the RTX A6000 host
- `docker/compose.yaml` — Jetson deployment
- `docker/compose.remote.yaml` — RTX A6000 deployment
