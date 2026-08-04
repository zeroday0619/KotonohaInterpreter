# Kotonoha Interpreter

Consecutive four-language speech interpreter for NVIDIA Jetson AGX Orin 64GB, with an
optional RTX A6000 service host.

## Overview

Kotonoha processes a completed utterance, translates it directly into the configured
target language, and synthesizes the translation. It does not implement simultaneous
interpretation. ASR, translation, and TTS operate without cloud APIs.

| Property | Value |
|---|---|
| Languages | Korean, English, Traditional Chinese (Taiwan), Japanese |
| Directions | 12 direct translation directions |
| Source language | Automatic identification with previous-language fallback |
| Target routing | Language pair, fixed target, or broadcast |
| Interaction | Push-to-talk, automatic VAD, or typed input |
| Interface | Localized Textual TUI |
| Latency objective | 2.9 seconds from end-of-utterance to first audio |
| Design priority | Accuracy before latency |

## Project Status

Phase 0 target-device validation has not been completed. The following settings remain
provisional until measured on the Jetson:

| Decision | Setting | Current default | Validation |
|---|---|---|---|
| Primary ASR runtime | `asr.backend` | `vllm` | Spike 1 acceptance measurement |
| TTS backend | `tts.backend` | `melo` | Spike 2 |
| Translation model class | `llm.profile` | `dense` | Spike 3 |

The primary ASR service uses vLLM multimodal beam search and returns five scored
hypotheses. Transformers remains available as an explicit fallback. Spike 1 must still
measure model loading, N-best latency, and sm_87 behavior on the Jetson.

No Jetson compatibility, model latency, token generation rate, or thermal result is
claimed without target-device measurements. See [Phase 0 validation](spikes/README.md).

## Architecture

The audio frontend and orchestrator run on the Jetson. Model processes remain resident
across turns.

```text
[microphone]
    |
    v
Audio frontend (CPU)
  DeepFilterNet3 -> Silero VAD -> 300 ms preroll -> 800 ms EOU silence
    |
    v
Shared-memory audio ring
    |
    v
Orchestrator (asyncio and uvloop)
    |-- :8001 primary ASR       Qwen3-ASR 1.7B, N-best 5, LID
    |-- :8002 verification ASR  faster-whisper large-v3, conditional
    |-- :8003 translation LLM   llama.cpp, correction and translation
    `-- :8004 TTS               Qwen3-TTS 0.6B or MeloTTS
    |
    v
[speaker and terminal UI]
```

### Turn Workflow

| Stage | Contract |
|---|---|
| Capture | 48 kHz mono audio from PortAudio |
| Segmentation | VAD includes 200-300 ms preroll and closes after 800 ms silence |
| Primary ASR | Five hypotheses, average log-probability, and language label |
| Language decision | Inherits the previous language for short or low-confidence input |
| Verification | Runs only when the quality gate activates on the Jetson |
| Translation | Corrects transcription and translates in one LLM pass |
| Streaming | Sends complete clauses to TTS before LLM completion |
| Playback | Closes the microphone until the TTS queue is empty |

The state machine permits these transitions:

```text
IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE
  `----------------> PROCESSING                 typed input
```

`Orchestrator._on_state_change` owns half-duplex microphone gating. TTS output cannot
re-enter capture while the state is `SPEAKING`.

### Audio Transport

Local ASR services receive an `AudioRef` containing `{name, slot, seq, frames}` and read
PCM from POSIX shared memory. Remote ASR services receive binary multipart PCM. Neither
path base64-encodes audio.

The orchestrator publishes both representations in `AudioPayload`. Client routing selects
the representation, so placement does not branch the processing pipeline.

## Accuracy Constraints

These contracts require explicit approval and regression coverage before modification.

| Constraint | Implementation |
|---|---|
| VAD preroll remains 200-300 ms | `src/kotonoha/audio/_vad.py` |
| Primary ASR returns N-best 5 | `src/kotonoha/services/_asr_server.py` |
| One LLM pass performs correction and translation | `src/kotonoha/prompts/_translate.py` |
| Translation reaches TTS by clause | `src/kotonoha/core/_clauses.py` |
| Cross-verification remains conditional on the Jetson | `src/kotonoha/core/_quality.py` |
| Half-duplex gating remains centralized | `src/kotonoha/core/_orchestrator.py` |
| Audio remains binary or shared-memory based | `src/kotonoha/_shmring.py`, `src/kotonoha/_transport.py` |

Traditional Chinese input and output pass through OpenCC `s2twp`. Translation prompts
also enforce Taiwanese vocabulary, including `軟體`, `影片`, `資訊`, and `滑鼠`.

## Runtime Placement

`perf_mode` controls service placement.

| Mode | ASR | Verification | LLM | TTS | Audio leaves Jetson |
|---|---|---|---|---|---|
| `onboard` | Jetson | Jetson | Jetson | Jetson | No |
| `hybrid` | Jetson | Jetson | A6000 | Jetson | No |
| `remote` | A6000 | A6000 | A6000 | A6000 | Yes |

Each remote role retains a resident Jetson fallback. Transport failures retry locally.
Application errors such as HTTP 4xx responses do not activate failover. Streaming roles
fail over only before the first chunk because an active stream cannot be rewound.

Remote services require a bearer token when `KOTONOHA_SERVICE_TOKEN` is set. Plain HTTP
does not provide confidentiality. Restrict service ports to a trusted network or place a
TLS reverse proxy in front of them.

## Configuration

Configuration layers merge in this order, with later values taking precedence:

1. `config/default.yaml`
2. The file selected with `--config` or `KOTONOHA_CONFIG`
3. `config/local.yaml`, or `KOTONOHA_LOCAL_CONFIG`
4. `KOTONOHA__*` environment variables

Nested environment keys use a double underscore:

```bash
KOTONOHA__PERF_MODE=hybrid \
KOTONOHA__REMOTE__ENABLED=true \
KOTONOHA__REMOTE__SERVICES__LLM=http://a6000.internal:8003 \
uv run kotonoha doctor
```

`kotonoha config` exposes every local `Settings` leaf. Local changes are validated and
written atomically to `config/local.yaml`. Remote changes use the authenticated
`/admin/config` endpoint and only accept settings consumed by resident model services.
Remote model changes require a service restart.

## Installation and Deployment

The deployment guide defines host preparation, model staging, Docker access, Jetson and
A6000 deployment, security controls, rollback, troubleshooting, and acceptance checks:

- [Installation and Deployment](docs/installation-and-deployment.md)
- [Phase 0 Validation Spikes](spikes/README.md)

Quick deployment commands:

```bash
# Jetson model services
bash scripts/deploy.sh jetson

# RTX A6000 model services
bash scripts/deploy.sh a6000

# Preserve models, configuration, logs, and SQLite data while removing services
bash scripts/deploy.sh uninstall jetson
bash scripts/deploy.sh uninstall a6000
```

The deployment script uses `sudo docker` when the current account cannot access the
Docker daemon directly. It does not start the interactive orchestrator.

## Operator Interface

Install the development environment with uv:

```bash
uv sync
```

Primary commands:

| Command | Function |
|---|---|
| `uv run kotonoha tui` | Open the integrated control center |
| `uv run kotonoha run` | Open the interpreter directly |
| `uv run kotonoha config` | Edit local or remote configuration |
| `uv run kotonoha history browse` | Browse completed turns |
| `uv run kotonoha text "<utterance>"` | Interpret typed text |
| `uv run kotonoha replay <wav> --seconds 12` | Replay a WAV through the voice pipeline |
| `uv run kotonoha devices` | List audio devices |
| `uv run kotonoha doctor` | Report environment and service health |
| `uv run kotonoha netcheck` | Measure remote latency and upload throughput |
| `uv run kotonoha serve <service>` | Start a Python model service |

The integrated TUI provides interpreter, configuration, history, operations, and license
views. Structured JSON logs render as bounded, human-readable records in the interpreter
footer without writing terminal control output to the application log.

### Text Input

Text mode closes the microphone and enters `PROCESSING` directly from `IDLE`. It skips
capture, ASR, and cross-verification, then rejoins the voice path at correction and
translation.

Source-language selection follows this order:

1. Explicit `--from` or `session.text_source_language`
2. Unicode script detection
3. Previous language inheritance

### History

SQLite stores source text, translation, language decision provenance, ASR confidence,
verification status, timing data, placement, failovers, and outcome. The TUI keeps the
current turn separate from completed history and clears live ASR and translation panes
when the next turn starts.

## Localization

English message identifiers are the source text. Korean, Japanese, and Traditional
Chinese translations live under `src/kotonoha/locale/`. Locale resolution order is:

1. `KOTONOHA_LANG`
2. `ui.language`
3. `LC_ALL`, `LC_MESSAGES`, or `LANG`
4. English

Typer renders command help during import. Set `KOTONOHA_LANG` to localize help output;
`--lang` controls command output after parsing.

Catalog maintenance:

```bash
uv run python scripts/i18n.py extract
uv run python scripts/i18n.py update
uv run python scripts/i18n.py compile
uv run python scripts/i18n.py check
```

Commit `.po` files. The build hook compiles `.mo` files during installation.

## Instrumentation

Every turn records five timestamps:

```text
EOU detected -> ASR complete -> first clause -> first audio packet -> queue drained
```

The turn record also contains detected language, LID confidence, ASR average
log-probability, verification activation, audio duration, output token count, placement,
failovers, and outcome.

| File | Content |
|---|---|
| `data/logs/kotonoha.jsonl` | Application events |
| `data/logs/turns.jsonl` | One structured record per turn |
| `data/kotonoha.db` | Glossary and interpretation history |

## Development

```bash
uv run ruff check .
uv run pytest -q
```

Tests run without models, microphones, or network access. The source targets Python 3.10
for JetPack containers. Development workstations can run newer Python versions.

Evaluation recordings require the production microphone and room. The target data set
contains 100 utterances per language with reference transcripts and translations. ASR
uses CER. Translation uses COMET in offline batches on the development workstation.

## Model Identifiers

The repository currently configures these identifiers:

| Component | Identifier |
|---|---|
| Primary ASR, vLLM | `Qwen/Qwen3-ASR-1.7B` |
| Primary ASR, Transformers fallback | `Qwen/Qwen3-ASR-1.7B-hf` |
| TTS | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| MoE GGUF | `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` |
| Dense GGUF | `unsloth/Qwen3-14B-GGUF` |
| Jetson vLLM image | `ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04` |

## Limitations

- Phase 0 target measurements remain pending.
- Jetson vLLM model loading and latency remain unverified until Spike 1 runs on sm_87.
- Complete model inference cannot be validated on the macOS development workstation.
- Push-to-talk is a terminal toggle because terminals do not expose key-release events.
- Remote configuration changes do not reload resident models.
- TLS termination and a production supervisor outside Docker Compose are not included.

## Scope Exclusions

- Cloud ASR, translation, or TTS APIs
- Simultaneous interpretation policies
- English-pivot translation
- Browser microphone capture
- Per-request model loading
- Vector databases or embedding models for glossary lookup
- Unvalidated JetPack, CUDA, L4T, or base-image upgrades
