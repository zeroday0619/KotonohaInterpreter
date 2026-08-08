# Kotonoha Interpreter

Accelerator-aware consecutive four-language speech interpreter. NVIDIA Jetson AGX Orin
supports appliance deployment, while RTX A6000-class hosts can run the complete browser
service and resident model stack.

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
| Interface | Localized multi-session Web UI |
| Latency objective | 2.9 seconds from end-of-utterance to first audio |
| Design priority | Accuracy before latency |

## Project Status

Jetson Phase 0 and A6000 performance acceptance have not been completed. These settings
remain provisional until measured on their deployment hosts:

| Decision | Setting | Current default | Validation |
|---|---|---|---|
| Primary ASR runtime | `asr.backend` | `vllm` | Spike 1 acceptance measurement |
| TTS backend | `tts.backend` | `vllm_omni` | Spike 2 |
| Translation model | `llm.profile` | `translategemma` | Spike 3 |

The primary ASR and translation FastAPI services own resident vLLM engines. TTS uses the
vLLM-Omni Speech API with the local Qwen3-TTS snapshot. Translation uses TranslateGemma
4B on Jetson and 12B on A6000, streaming through `/v1/realtime` WebSocket without a
nested vLLM HTTP server. Spikes 1-3
must still verify model loading, concurrent residency, latency, throughput, and sm_87
behavior on the Jetson. No target compatibility or performance result is claimed without
target measurement. The [Implementation Plan](docs/planning/README.md) defines the phase
gates.

The A6000 deployment queries GPU UUIDs and available memory before first startup. It
assigns ASR, verification ASR, translation, and TTS to physical GPUs from configurable
memory reservations. The generated UUID mapping remains stable until an operator requests
reallocation.

## Quick Start

Install the development environment:

```bash
bash scripts/manage.sh setup workstation
```

Open the Web control center:

```bash
uv run kotonoha web
```

Run the workstation quality gates:

```bash
bash scripts/manage.sh check
```

Deploy model services:

```bash
bash scripts/manage.sh deploy jetson
bash scripts/manage.sh deploy a6000
```

Deploy the browser interface with the resident model stack:

```bash
cp .env.example .env
bash scripts/manage.sh web jetson
bash scripts/manage.sh web a6000
```

The Web UI provides interpretation, complete configuration editing, history management,
operator tools, dependency licenses, and application logs. Configuration writes are
validated and applied without restarting the Web process. Resident model settings invoke
service-level backend reloads.

The [management script reference](docs/operations/management.md) defines model staging,
target setup, benchmarking, deployment, and dry-run commands.

## Documentation

The [documentation index](docs/README.md) organizes project documentation by category.

| Category | Document |
|---|---|
| Planning | [Implementation Plan](docs/planning/README.md) |
| Architecture | [System Architecture](docs/architecture/README.md) |
| User guide | [Operator Guide](docs/user-guide/README.md) |
| Deployment | [Installation and Deployment](docs/deployment/installation.md) |
| Environment | [Environment Variables](docs/deployment/environment.md) |
| Operations | [Service Runbook](docs/operations/service-runbook.md) |
| Performance | [Performance Measurement](docs/performance/measurement.md) |
| Development | [Development](docs/development/README.md) |

## Limitations

- Jetson Phase 0 and A6000 performance measurements remain pending.
- Jetson vLLM model loading and latency remain unverified until Spike 1 runs on sm_87.
- Translation AWQ loading and throughput remain unverified until Spike 3 runs on each
  target.
- Default A6000 GPU memory reservations remain provisional until peak resident memory is
  measured with the complete remote stack.
- Complete model inference cannot be validated on the macOS development workstation.
- Push-to-talk uses a toggle control in the Web interface and the Space keyboard shortcut.
- A resident model is temporarily unavailable while its in-process backend reloads.
- TLS termination and a production supervisor outside Docker Compose are not included.

## AI-Generated Code Notice

Parts of this project were created with assistance from AI tools (e.g. large language models). All AI-assisted contributions were reviewed and adapted by maintainers before inclusion. If you need provenance for specific changes, please refer to the Git history and commit messages.
