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

Run `bash scripts/manage.sh models fetch` before either track. Spike 3 requires both AWQ
model snapshots. Use a real recording of approximately six seconds for Spike 1. Synthetic
audio provides timing data only and cannot validate transcription quality.

## ASR Measurement

### Jetson

Run the vLLM path through the hardware spike container:

```bash
ASR_ONLY=vllm bash scripts/manage.sh benchmark jetson --only 1
```

The selected image tag targets Jetson Linux r36.4, while the host contract is Jetson
Linux 39.2. Its build metadata targets CUDA architecture 8.7. A successful image pull
validates only registry access and the arm64/v8 manifest. Spike 1 must execute CUDA
kernels on sm_87; reject the image if the container cannot use the host driver or the
runtime reports an unsupported compute capability or no compatible kernel image.

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
bash scripts/manage.sh benchmark jetson --only 2
```

The default `vllm/vllm-omni:v0.26.0` image has an arm64 manifest. The probe must still
start it on the Jetson Linux 39.2 host, initialize CUDA, execute a FlashAttention kernel,
load the local model, and stream audio before the path is accepted.

### RTX A6000

Run the same API and model probe through the amd64 variant of the official image.

```bash
bash scripts/manage.sh benchmark a6000 --only 2
```

The probe executes a FlashAttention kernel, starts `vllm serve --omni`, waits for
`/health`, and records 24 kHz signed 16-bit PCM streaming measurements for Korean,
English, Japanese, and Chinese. The full vLLM-Omni server log is retained beside the JSON
result.

| Criterion | Requirement |
|---|---|
| TTS backend | vLLM-Omni Qwen3-TTS on both targets |
| First PCM packet | 300 ms or less after a warm request |
| Language output | Non-empty PCM for all four languages |
| Kernel result | FlashAttention returns finite output with the expected shape |

## Translation Measurement

Jetson Phase 0 uses context 2048, batch 1, and 60 output tokens.

```bash
LLM_CONTEXT=2048 \
LLM_GPU_MEMORY_UTILIZATION=0.80 \
LLM_OUTPUT_TOKENS=60 \
bash scripts/manage.sh benchmark jetson --only 3
```

The A6000 measurement uses the production 4096-token context:

```bash
LLM_CONTEXT=4096 \
LLM_GPU_MEMORY_UTILIZATION=0.80 \
LLM_OUTPUT_TOKENS=60 \
BENCHMARK_RUNS=3 \
bash scripts/manage.sh benchmark a6000 --only 3
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
