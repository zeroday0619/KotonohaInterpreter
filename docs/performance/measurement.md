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
| 2 | FlashAttention execution and vLLM-Omni Qwen3-TTS PCM streaming | `tts.*` |
| 3 | Target TranslateGemma WebSocket rate, TTFT, and one-pass marker | `llm.profile` |

Do not compare measurements from different hosts as one runtime result. Every retained
result must identify the target, GPU, compute capability, runtime versions, and benchmark
conditions.

## Applied vLLM Profiles

The service passes optimization arguments only when the selected device profile supports
them. PagedAttention remains an internal vLLM implementation and requires no project
setting.

| Feature | Jetson AGX Orin | RTX A6000 | Reason |
|---|---|---|---|
| Accelerator profile | `nvidia.jetson.agx-orin` | `nvidia.rtx.a6000` | Hardware identity selects the device-specific service defaults |
| KV cache dtype | FP8 | Automatic | Jetson memory pressure requires a smaller KV cache; A6000 retains model-native dtype selection |
| GPU memory utilization | 0.35 | 0.90 | Jetson shares unified memory with the host; A6000 reserves headroom for the dedicated translation engine |
| Prefix caching | Disabled | Enabled | A6000 has capacity for repeated system-prompt reuse; Jetson avoids cache residency growth |
| Chunked-prefill budget | 2048 tokens | 4096 tokens | Jetson prioritizes inter-token latency and cache pressure; A6000 receives a larger prefill budget |
| Sequence limit | 1 | 1 | The orchestrator processes one consecutive turn at a time |
| Compilation | Eager execution | Mode 2 with CUDA graphs for `[1, 2, 4]` | Jetson target compatibility remains the priority; A6000 limits graph capture to the service concurrency |
| Compilation cache | Disabled | `/models/vllm-compile-cache` | Reuse compiled artifacts across resident-service restarts |
| Speculative decoding | Not configured | Not configured | No validated draft model is part of the translation deployment |
| Pipeline or disaggregated prefill | Not configured | Not configured | The 4B and 12B services fit the assigned GPU and run as one resident engine |

These values are initial device-specific settings, not performance results. Spike 3 must
measure time to first token, generation rate, cache usage, and resident memory before an
optimization is accepted. vLLM 0.24.0 defines `max_num_batched_tokens`, prefix caching,
and `compilation_config` as engine arguments, and its optimization guide documents the
trade-off between chunked-prefill budget, inter-token latency, and time to first token.
[Engine arguments](https://docs.vllm.ai/en/v0.24.0/configuration/engine_args/),
[optimization guide](https://docs.vllm.ai/en/v0.24.0/configuration/optimization/)

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

Run `bash scripts/manage.sh models fetch` before either track. Spike 3 requires both AWQ
model snapshots. Use a real recording of approximately six seconds for Spike 1. Synthetic
audio provides timing data only and cannot validate transcription quality.

## ASR Measurement

The WebSocket probe follows the
[vLLM Speech to Text API](https://docs.vllm.ai/en/latest/serving/online_serving/speech_to_text/):
base64 PCM16, mono, 16 kHz audio with `session.update`, buffer commit and append events,
then transcription delta and done events. The service wraps vLLM's connection class in
the resident FastAPI process instead of starting a nested vLLM server.

### Jetson

Run the vLLM path through the hardware spike container:

```bash
ASR_ONLY=vllm bash scripts/manage.sh benchmark jetson --only 1
```

The selected `nvcr.io/nvidia/vllm:26.07-py3` image provides an arm64 manifest with CUDA
13.3.1 and vLLM `0.24.0+092c4842`. Its architecture list does not include Orin sm_87.
A successful image pull validates only registry access and manifest selection. Spike 1
must execute CUDA kernels on sm_87; reject the image if the container cannot use the
host driver or reports an unsupported compute capability or no compatible kernel image.

The Jetson probe loads `Qwen/Qwen3-ASR-0.6B` with vLLM's
`Qwen3ASRRealtimeGeneration` override. It must exercise both N-best five beam search and
the embedded `/v1/realtime` protocol before the target path is accepted.

Run the Transformers fallback in a separately validated JetPack 7.2 Arm64 image when
comparison is required. Preserve both backend results in the final `spike1.json`.

### RTX A6000

Run through the x86_64 vLLM container on the A6000 host:

```bash
ASR_ONLY=vllm \
VLLM_GPU_MEMORY_UTILIZATION=0.80 \
VLLM_MAX_MODEL_LEN=4096 \
VLLM_ENFORCE_EAGER=1 \
bash scripts/manage.sh benchmark a6000 --only 1
```

The harness never installs the PyPI vLLM wheel into the host environment. The selected
`nvcr.io/nvidia/vllm:26.07-py3` container supplies the CUDA-matched runtime. The image
metadata reports vLLM `0.24.0+092c4842`; the benchmark must confirm that the project ASR
and translation interfaces remain compatible with that runtime.

The A6000 probe loads `mistralai/Voxtral-Mini-4B-Realtime-2602` in BF16 and exercises
the same in-process N-best and WebSocket paths. The model selection does not establish
latency, memory use, or compatibility until this target run completes.

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
| Realtime transcription | At least one delta and one final event |
| Six-second transcription | 900 ms or less |

A path without five hypotheses fails the accuracy contract regardless of latency.

## TTS Measurement

### Jetson

```bash
bash scripts/manage.sh benchmark jetson --only 2
```

The TTS probe image derives from `vllm/vllm-omni:v0.26.0`, which has an arm64 manifest.
The probe must still start the Kotonoha FastAPI service on the Jetson Linux 39.2 host,
initialize CUDA, execute a FlashAttention kernel, load the local model, and stream audio
before the path is accepted.

### RTX A6000

Run the same FastAPI service and model probe through the amd64 base-image variant.

```bash
bash scripts/manage.sh benchmark a6000 --only 2
```

The probe executes the vLLM-bundled FlashAttention-2 kernel, starts
`kotonoha.services._tts_server`, waits for its in-process vLLM-Omni engine to pass
`/health`, and records 24 kHz signed 16-bit PCM streaming measurements for Korean,
English, Japanese, and Chinese. The full service and engine traceback is retained beside
the JSON result.

| Criterion | Requirement |
|---|---|
| TTS backend | vLLM-Omni Qwen3-TTS on both targets |
| First PCM packet | 300 ms or less after a warm request |
| Language output | Non-empty PCM for all four languages |
| Kernel result | FlashAttention returns finite output with the expected shape |

## Translation Measurement

Jetson Phase 0 measures TranslateGemma 4B with context 2048, batch 1, and 60 output
tokens.

```bash
LLM_CONTEXT=2048 \
LLM_GPU_MEMORY_UTILIZATION=0.80 \
LLM_OUTPUT_TOKENS=60 \
bash scripts/manage.sh benchmark jetson --only 3
```

The A6000 measurement uses TranslateGemma 12B's 2048-token context:

```bash
LLM_CONTEXT=2048 \
LLM_GPU_MEMORY_UTILIZATION=0.80 \
LLM_OUTPUT_TOKENS=60 \
BENCHMARK_RUNS=3 \
bash scripts/manage.sh benchmark a6000 --only 3
```

The probe starts the project FastAPI service, which owns the vLLM engine in-process, then
records time to first token and generation rate through `/v1/realtime`. It also records
whether the response preserves the one-pass correction-and-translation source marker.

The minimum rate is 5 tok/s. Translation accuracy and the correction behavior remain
separate evaluation-set decisions.

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

`run_all.sh` executes `report.py` in the report container after every selected probe. A
single-spike run retains the other accepted result files and regenerates the combined
target report. Remove obsolete candidate files before generating an acceptance report.

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
bash scripts/manage.sh benchmark jetson
bash scripts/manage.sh benchmark a6000
```

The A6000 run writes only under `spikes/out/a6000` and cannot overwrite Jetson results.
