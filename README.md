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

Jetson Phase 0 and A6000 performance acceptance have not been completed. These settings
remain provisional until measured on their deployment hosts:

| Decision | Setting | Current default | Validation |
|---|---|---|---|
| Primary ASR runtime | `asr.backend` | `vllm` | Spike 1 acceptance measurement |
| TTS backend | `tts.backend` | `melo` | Spike 2 |
| Translation model class | `llm.profile` | `dense` | Spike 3 |

The primary ASR service implements vLLM multimodal beam search and returns five scored
hypotheses. Spike 1 must still verify model loading, latency, and sm_87 behavior on the
Jetson. No target compatibility or performance result is claimed without target
measurement.

## Quick Start

Install the development environment:

```bash
uv sync
```

Open the integrated terminal interface:

```bash
uv run kotonoha tui
```

Run the workstation quality gates:

```bash
uv run ruff check .
uv run pytest -q
```

Deploy model services:

```bash
bash scripts/deploy.sh jetson
bash scripts/deploy.sh a6000
```

## Documentation

The [documentation index](docs/README.md) organizes project documentation by category.

| Category | Document |
|---|---|
| Architecture | [System Architecture](docs/architecture/README.md) |
| User guide | [Operator Guide](docs/user-guide/README.md) |
| Deployment | [Installation and Deployment](docs/deployment/installation.md) |
| Operations | [Service Runbook](docs/operations/service-runbook.md) |
| Performance | [Performance Measurement](docs/performance/measurement.md) |
| Development | [Development](docs/development/README.md) |

## Limitations

- Jetson Phase 0 and A6000 performance measurements remain pending.
- Jetson vLLM model loading and latency remain unverified until Spike 1 runs on sm_87.
- Complete model inference cannot be validated on the macOS development workstation.
- Push-to-talk is a terminal toggle because terminals do not expose key-release events.
- Remote configuration changes do not reload resident models.
- TLS termination and a production supervisor outside Docker Compose are not included.
