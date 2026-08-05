# Performance Measurement

## Objective

This procedure validates Jetson compatibility and optimizes the optional RTX A6000
high-performance mode. It defines the authoritative measurement conditions, acceptance
thresholds, evidence, and configuration outputs.

The [Implementation Plan](../planning/README.md) defines Phase 0 governance and the
approval gate for subsequent phases.

| Track | Host | Purpose | Output directory |
|---|---|---|---|
| Phase 0 | Jetson AGX Orin | Validate hardware compatibility and on-device fallbacks | `spikes/out` |
| High-performance mode | RTX A6000 | Tune remote services and approve role placement | `spikes/out/a6000` |

Phase 0 remains a gate for Jetson deployment. A6000 acceptance does not replace it. The
Jetson keeps resident fallback services and owns capture, VAD, routing, and half-duplex
gating.

## Measurements

| Spike | Measurement | Configuration decision |
|---|---|---|
| 1 | Qwen3-ASR load, N-best 5, log-probabilities, six-second latency | `asr.*` |
| 2 | FlashAttention execution and Qwen3-TTS synthesis latency | `tts.*` |
| 3 | MoE and dense LLM generation rate and time to first token | `llm.profile` |

Do not compare measurements from different hosts as one runtime result. Every retained
result must identify the target, GPU, compute capability, runtime versions, and benchmark
conditions.

## Preconditions

### Jetson AGX Orin

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
jtop
```

Discard measurements collected during thermal throttling. Keep `jtop` visible throughout
the run.

### RTX A6000

```bash
nvidia-smi
docker info --format '{{json .Runtimes}}'
docker compose -f docker/compose.remote.yaml ps
```

All four remote services must remain resident during final acceptance. An isolated model
benchmark does not prove concurrent residency.

Retain `config/remote-gpu.env` and the corresponding `nvidia-smi` inventory with every
measurement. Each configured role reservation must be at least its measured peak GPU
memory consumption, and the per-GPU safety reserve must remain free during the complete
resident workload.

### Artifacts

Run `scripts/fetch_models.sh` before either track. Spike 3 requires both AWQ model
snapshots. Use a real recording of approximately six seconds for Spike 1. Synthetic audio
provides timing data only and cannot validate transcription quality.

## ASR Measurement

### Jetson

Run the vLLM path in the configured Jetson image:

```bash
jetson-containers run ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04
python3 spikes/spike1_asr_load.py \
  --target jetson \
  --wav samples/ko_6s.wav \
  --only vllm \
  --out spikes/out/spike1.json
```

Run the Transformers fallback in its compatible r36.4.0 image when comparison is
required. Preserve both backend results in the final `spike1.json`.

### RTX A6000

Run inside the remote ASR image:

```bash
python3 spikes/spike1_asr_load.py \
  --target a6000 \
  --wav samples/ko_6s.wav \
  --only vllm \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --enforce-eager \
  --out spikes/out/a6000/spike1.json
```

Tune one variable per run. Store candidates under distinct names, then copy the accepted
result to `spike1.json` before report generation.

| Variable | Values to measure | Constraint |
|---|---|---|
| `--gpu-memory-utilization` | `0.70`, `0.80`, `0.90` | All remote services remain resident |
| `--enforce-eager` | enabled, disabled | Model load and N-best inference succeed |
| `--max-model-len` | production value | Covers the configured request |

Acceptance criteria:

| Criterion | Requirement |
|---|---|
| Model load | Succeeds |
| N-best output | Exactly five sequences |
| Log-probabilities | Available per sequence |
| Six-second transcription | 900 ms or less |

A path without five hypotheses fails the accuracy contract regardless of latency.

## TTS Measurement

### Jetson

```bash
jetson-containers run $(autotag flash-attention)
python3 spikes/spike2_flash_attn.py \
  --target jetson \
  --out spikes/out/spike2.json
```

### RTX A6000

Run inside `kotonohainterpreter-tts`. The remote image does not contain MeloTTS.

```bash
python3 spikes/spike2_flash_attn.py \
  --target a6000 \
  --skip-melo \
  --out spikes/out/a6000/spike2.json
```

The probe executes a FlashAttention kernel and measures Qwen3-TTS with
`flash_attention_2`, `sdpa`, and `eager`.

| Criterion | Requirement |
|---|---|
| TTS backend | Qwen3-TTS on A6000; Qwen3-TTS or MeloTTS on Jetson |
| Single-clause synthesis | 300 ms or less |
| Kernel result | Finite output with the expected shape |

## Translation Measurement

Jetson Phase 0 uses context 2048, batch 1, and 60 output tokens.

```bash
python3 spikes/spike3_llm_tokrate.py \
  --target jetson \
  --vllm-command vllm \
  --models-dir ./models/llm \
  --context 2048 \
  --gpu-memory-utilization 0.80 \
  --output-tokens 60 \
  --out spikes/out/spike3.json
```

The A6000 measurement uses the production 4096-token context:

```bash
python3 spikes/spike3_llm_tokrate.py \
  --target a6000 \
  --vllm-command vllm \
  --models-dir /models/llm \
  --context 4096 \
  --gpu-memory-utilization 0.80 \
  --output-tokens 60 \
  --runs 3 \
  --out spikes/out/a6000/spike3.json
```

The probe starts one vLLM server per profile and records time to first token and generation
rate under the production prompt shape. It terminates each server before loading the next
profile so both measurements use the same available GPU capacity.

The minimum rate is 5 tok/s. Select MoE only when it meets the threshold and is not slower
than dense 14B. Translation quality remains a separate evaluation-set decision.

## Link Measurement

Run from the Jetson after all A6000 services are healthy:

```bash
KOTONOHA__REMOTE__TOKEN=<token> \
uv run kotonoha -c config/performance.yaml netcheck --samples 20 --seconds 6
```

`netcheck` measures service RTT and binary PCM upload time. Use `remote` placement only
when every remote role is reachable and estimated link overhead consumes no more than 25%
of the end-of-utterance latency budget. Use `hybrid` when audio must remain on the Jetson
or measured overhead exceeds the limit.

## Concurrent Residency

Start the complete A6000 stack after selecting candidate settings:

```bash
set -a
source .env
source config/remote-gpu.env
set +a
docker compose -f docker/compose.remote.yaml up -d
docker compose -f docker/compose.remote.yaml ps
docker compose -f docker/compose.remote.yaml config | sed -n '/device_ids:/,+2p'
nvidia-smi
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8002/health
curl -fsS http://127.0.0.1:8003/health
curl -fsS http://127.0.0.1:8004/health
```

Reject a configuration that passes isolated measurements but causes an out-of-memory
restart when all services are resident.

Record the GPU UUID assigned to each role. Repeat concurrent-residency and latency
measurements after every `--reallocate-gpus` operation because placement changes the
available memory and contention profile.

## Reporting

### Jetson Phase 0

```bash
python3 spikes/report.py \
  --target jetson \
  --dir spikes/out \
  --md spikes/out/PHASE0.md \
  --patch spikes/out/local.yaml
```

### RTX A6000

```bash
python3 spikes/report.py \
  --target a6000 \
  --dir spikes/out/a6000 \
  --md spikes/out/a6000/PERFORMANCE.md \
  --patch spikes/out/a6000/remote-server.local.yaml
```

| Output | Content |
|---|---|
| `PHASE0.md` | Jetson compatibility verdicts and latency reconciliation |
| `local.yaml` | Jetson configuration overlay |
| `PERFORMANCE.md` | A6000 server-stage measurements and selected settings |
| `remote-server.local.yaml` | Remote service configuration overlay |

The A6000 report excludes network overhead. Attach `netcheck` output, `nvidia-smi`, and
service health evidence before approving high-performance mode.

## Batch Execution

The harness interface is documented in [Hardware Spike Harness](../../spikes/README.md).

```bash
bash spikes/run_all.sh jetson
bash spikes/run_all.sh a6000
```

The A6000 run writes only under `spikes/out/a6000` and cannot overwrite Jetson results.
