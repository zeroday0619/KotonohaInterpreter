# Hardware Spike Harness

## Purpose

This directory contains the executable probes and report generator used by the hardware
performance procedure. Measurement conditions, acceptance thresholds, and operator steps
are defined in [Performance Measurement](../docs/performance/measurement.md).

The harness supports two isolated targets.

| Target | Output directory | Configuration patch |
|---|---|---|
| Jetson AGX Orin | `spikes/out` | `local.yaml` |
| RTX A6000 | `spikes/out/a6000` | `remote-server.local.yaml` |

Results from different targets are not interchangeable. Each JSON result records its
target, runtime environment, and benchmark conditions.

## Components

| File | Role |
|---|---|
| `spike1_asr_load.py` | Measures Qwen3-ASR loading, N-best output, scores, and latency |
| `spike2_flash_attn.py` | Executes FlashAttention and measures vLLM-Omni Qwen3-TTS streaming |
| `spike3_llm_tokrate.py` | Compares MoE and dense AWQ generation through vLLM |
| `run_all.sh` | Selects target images and runs each probe through Docker Compose |
| `report.py` | Produces the target report and validated configuration values |
| `../docker/compose.spikes.yaml` | Defines isolated ASR, TTS, LLM, and report containers |

## Docker Interface

```bash
bash scripts/manage.sh benchmark jetson
bash scripts/manage.sh benchmark a6000
bash scripts/manage.sh benchmark jetson --only 1
bash scripts/manage.sh benchmark a6000 --only 3
```

The management script delegates hardware measurements to `run_all.sh`. The runner
accesses Docker directly when the current account has daemon permission and falls back to
`sudo docker` when elevated access is required. The runner passes target-specific Compose
variables through an explicit `sudo env` invocation because standard sudo policy removes
exported shell variables. It refreshes the target-specific ASR image through the Docker
build cache and builds the FastAPI TTS image from official vLLM-Omni. User-configured image
tags are inspected but never rebuilt or pulled. The runner always regenerates the target
report from the available result files.

The harness does not install or execute vLLM in the host Python environment. The source
tree remains mounted at `/workspace`, and model snapshots are mounted read-only at
`/models`. Short-lived probes run as root because the vendor images install their Python
and vLLM environments for the default root runtime user. Jetson probes invoke
`/opt/venv/bin/python` explicitly instead of falling back to the Ubuntu system Python.
The prepared A6000 ASR image creates `/opt/kotonoha-venv` with access to the NGC system
packages, then uses `uv sync --frozen` to install the locked project dependencies without
modifying the externally managed system Python. It then replaces the locked NumPy 1.x
wheel inside that child environment with the NGC SciPy-compatible `>=2,<2.3` range.
This target-only ABI overlay does not modify the workstation or Jetson environments.
Other A6000 probes use the Python executable supplied by their image. The container
entrypoint returns result files and
their output directory to the invoking user's UID and GID before it exits. Image build
failure stops the runner before any probe is started.

## Images

| Target | Default ASR image | LLM and report image | Default TTS image |
|---|---|---|---|
| Jetson | `kotonohainterpreter-spike-asr:jetson` | `ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04` | `kotonohainterpreter-spike-tts:jetson` |
| A6000 | `kotonohainterpreter-spike-asr:a6000` | `nvcr.io/nvidia/vllm:26.07-py3` | `kotonohainterpreter-spike-tts:a6000` |

The default ASR image derives from the target vLLM image and adds the locked application
runtime dependencies required by the probe, including `soxr`. The native Hugging Face
ASR fallback remains isolated because it requires Transformers 5.13 or newer. Its install
clears the Jetson image's `UV_CONSTRAINT` only for that child environment; the vendor
vLLM environment retains its Transformers 4.57.3 constraint. The TTS image derives from
`vllm/vllm-omni:v0.26.0`, installs the locked FastAPI application into a uv environment
with access to the base system packages, and runs the same
`kotonoha.services._tts_server` path used by deployment. That service wraps the
vLLM-Omni engine and speech-serving objects in-process; it does not start an internal
HTTP server. No second TTS runtime enters the project lock.

The Jetson vLLM image targets CUDA architecture 8.7, but its r36.4 runtime predates the
Jetson Linux 39.2 host contract. Successful container and kernel execution on Orin sm_87
remain required Spike 1 results. The vLLM-Omni 0.26.0 manifest contains Linux arm64 and
amd64 variants; this does not establish Jetson Linux, CUDA, model, or kernel compatibility.
Spike 2 starts the server, executes a FlashAttention kernel, and requests four-language
raw PCM before accepting the path. Set `SPIKE_SKIP_BUILD=1` to use existing ASR and TTS
images without building or pulling them. A configured candidate image must already exist.
On CUDA targets the probe calls vLLM's bundled FlashAttention-2 interface rather than
requiring the separately packaged `flash_attn` module.

The A6000 NGC image advertises CUDA architecture 8.6 and vLLM `0.24.0+092c4842`.
Manifest metadata does not replace Spike 1 and Spike 3 execution on the A6000.

## Configuration

The A6000 runner accepts tuning conditions through environment variables:

| Variable | Default | Consumer |
|---|---:|---|
| `SPIKE_VLLM_IMAGE` | Target-specific vendor image from the table above | ASR build, Spike 3, report |
| `SPIKE_ASR_IMAGE` | Target-specific local image from the table above | Spike 1 |
| `SPIKE_TTS_IMAGE` | Target-specific image from the table above | Spike 2 |
| `SPIKE_GPU_DEVICE` | `all` on Jetson; `0` on A6000 | NVIDIA container runtime |
| `SPIKE_SKIP_BUILD` | `0` | ASR build and TTS image preparation |
| `MODELS_DIR` | `./models` | Read-only `/models` mount |
| `WAV` | `samples/ko_6s.wav` | Spike 1 |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.80` | Spike 1 |
| `VLLM_MAX_MODEL_LEN` | `4096` | Spike 1 |
| `VLLM_ENFORCE_EAGER` | `1` | Spike 1 |
| `LLM_CONTEXT` | `4096` | Spike 3 on A6000 |
| `LLM_OUTPUT_TOKENS` | `60` | Spike 3 |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.80` | Spike 3 |
| `LLM_MODELS_DIR` | `./models/llm` | Spike 3 |
| `BENCHMARK_RUNS` | `3` | Spikes 2 and 3 |
| `TTS_GPU_MEMORY_UTILIZATION` | `0.25` | Spike 2 |
| `ASR_ONLY` | `vllm` on A6000 | Spike 1 |
| `OUT` | Target output directory | All probes |

## Report Output

The report container runs after every invocation. `report.py` rejects result files whose
recorded target differs from the selected target. It does not combine server inference
measurements with network estimates. Link acceptance remains a separate operation in the
performance procedure.

## Constraints

- Run `bash scripts/manage.sh models fetch` before the harness. The wrapper rejects
  incomplete model snapshots before starting Docker.
- Qwen3-TTS reads `/models/Qwen3-TTS-0.6B`; runtime downloads remain disabled.
- Run model probes on the named deployment hardware.
- Use a real six-second recording when evaluating ASR content.
- Treat synthetic audio as timing-only input.
- Keep candidate runs under distinct file names.
- Copy only the accepted candidate to `spike1.json`, `spike2.json`, or `spike3.json`.
- Do not report unmeasured compatibility, latency, throughput, memory, or thermal values.
