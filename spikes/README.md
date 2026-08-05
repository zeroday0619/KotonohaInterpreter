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
| `spike2_flash_attn.py` | Executes FlashAttention and measures Qwen3-TTS backends |
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
`sudo docker` when elevated access is required. It builds the target-specific TTS spike
image when it is absent, then starts short-lived containers for the selected probes. It
always regenerates the target report from the available result files.

The harness does not install or execute vLLM in the host Python environment. The source
tree remains mounted at `/workspace`, model snapshots are mounted read-only at `/models`,
and result files are written through `/workspace/spikes/out` with the invoking user's UID
and GID.

## Images

| Target | ASR, LLM, report image | Default TTS image |
|---|---|---|
| Jetson | `ghcr.io/nvidia-ai-iot/vllm:r38.2.arm64-sbsa-cu130-24.04` | `kotonohainterpreter-spike-tts:jetson` |
| A6000 | `nvcr.io/nvidia/vllm:26.07-py3` | `kotonohainterpreter-spike-tts:a6000` |

The Jetson vLLM image advertises CUDA architecture 11.0, so successful kernel execution
on Orin sm_87 remains a required Spike 1 result. Set `SPIKE_TTS_IMAGE` to a separately
built FlashAttention candidate before Spike 2 when testing an image other than the
deployment TTS image. Set `SPIKE_SKIP_BUILD=1` to reject a missing TTS image instead of
building the default. A configured candidate image must already exist; the harness never
builds deployment contents under a user-supplied candidate tag.

The A6000 NGC image advertises CUDA architecture 8.6 and vLLM `0.24.0+092c4842`.
Manifest metadata does not replace Spike 1 and Spike 3 execution on the A6000.

## Configuration

The A6000 runner accepts tuning conditions through environment variables:

| Variable | Default | Consumer |
|---|---:|---|
| `SPIKE_VLLM_IMAGE` | Target-specific image from the table above | Spikes 1 and 3, report |
| `SPIKE_TTS_IMAGE` | Target-specific image from the table above | Spike 2 |
| `SPIKE_GPU_DEVICE` | `all` on Jetson; `0` on A6000 | NVIDIA container runtime |
| `SPIKE_SKIP_BUILD` | `0` | TTS image preparation |
| `MODELS_DIR` | `./models` | Read-only `/models` mount |
| `WAV` | `samples/ko_6s.wav` | Spike 1 |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.80` | Spike 1 |
| `VLLM_MAX_MODEL_LEN` | `4096` | Spike 1 |
| `VLLM_ENFORCE_EAGER` | `1` | Spike 1 |
| `LLM_CONTEXT` | `4096` | Spike 3 on A6000 |
| `LLM_OUTPUT_TOKENS` | `60` | Spike 3 |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.80` | Spike 3 |
| `LLM_MODELS_DIR` | `./models/llm` | Spike 3 |
| `BENCHMARK_RUNS` | `3` | Spike 3 |
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
- Run model probes on the named deployment hardware.
- Use a real six-second recording when evaluating ASR content.
- Treat synthetic audio as timing-only input.
- Keep candidate runs under distinct file names.
- Copy only the accepted candidate to `spike1.json`, `spike2.json`, or `spike3.json`.
- Do not report unmeasured compatibility, latency, throughput, memory, or thermal values.
