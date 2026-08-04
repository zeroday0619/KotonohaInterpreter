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
| `spike3_llm_tokrate.py` | Compares MoE and dense GGUF generation behavior |
| `run_all.sh` | Runs probes supported by the current container |
| `report.py` | Produces the target report and validated configuration values |

## Batch Interface

```bash
bash spikes/run_all.sh jetson
bash spikes/run_all.sh a6000
```

The probes can require different model images. Repeated execution in role-specific
containers is expected. `run_all.sh` preserves target separation even when an individual
container supports only one probe.

The A6000 runner accepts tuning conditions through environment variables:

| Variable | Default | Consumer |
|---|---:|---|
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.80` | Spike 1 |
| `VLLM_MAX_MODEL_LEN` | `4096` | Spike 1 |
| `VLLM_ENFORCE_EAGER` | `1` | Spike 1 |
| `LLM_CONTEXT` | `4096` | Spike 3 on A6000 |
| `LLM_OUTPUT_TOKENS` | `60` | Spike 3 |
| `BENCHMARK_RUNS` | `3` | Spike 3 |
| `ASR_ONLY` | `vllm` on A6000 | Spike 1 |
| `OUT` | Target output directory | All probes |

## Report Interface

Jetson report:

```bash
python3 spikes/report.py \
  --target jetson \
  --dir spikes/out \
  --md spikes/out/PHASE0.md \
  --patch spikes/out/local.yaml
```

A6000 report:

```bash
python3 spikes/report.py \
  --target a6000 \
  --dir spikes/out/a6000 \
  --md spikes/out/a6000/PERFORMANCE.md \
  --patch spikes/out/a6000/remote-server.local.yaml
```

`report.py` rejects result files whose recorded target differs from `--target`. It does
not combine server inference measurements with network estimates. Link acceptance remains
a separate operation in the performance procedure.

## Constraints

- Run model probes on the named deployment hardware.
- Use a real six-second recording when evaluating ASR content.
- Treat synthetic audio as timing-only input.
- Keep candidate runs under distinct file names.
- Copy only the accepted candidate to `spike1.json`, `spike2.json`, or `spike3.json`.
- Do not report unmeasured compatibility, latency, throughput, memory, or thermal values.
