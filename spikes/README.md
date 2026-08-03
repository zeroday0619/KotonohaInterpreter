# Phase 0 — Validation Spikes

## Objective

Resolve three architectural decisions by measurement on the target hardware. Each decision
gates work that cannot proceed correctly without it.

| Spike | Question | Decision | Setting |
|---|---|---|---|
| 1 | Does the Jetson vLLM container load Qwen3-ASR, and can it produce N-best output? | ASR runtime | `asr.backend` |
| 2 | Does flash-attn build and execute on sm_87, and does Qwen3-TTS load? | TTS backend | `tts.backend` |
| 3 | What is the measured token generation rate for the 30B MoE and the dense 14B? | Translation model class | `llm.profile` |

Execution stops after Phase 0. Results are reported before Phase 1 begins.

## Preconditions

The spikes require Jetson AGX Orin hardware. They produce no meaningful data on a
development workstation: the measurements depend on aarch64, CUDA, sm_87, and the
204.8 GB/s memory bandwidth of the target.

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
jtop
```

Discard any measurement taken during thermal throttling. Confirm the state in `jtop` in a
separate terminal for the duration of each run.

Spike 3 requires both GGUF artifacts, 27.6 GB combined. Download them with
`scripts/fetch_models.sh` before starting.

A real recording of approximately six seconds improves Spike 1. Without one the harness
generates synthetic audio; timing remains valid, transcription content does not.

## Spike 1 — vLLM and Qwen3-ASR

Run the vLLM path inside the vLLM container:

```bash
jetson-containers run ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04
python3 spikes/spike1_asr_load.py --wav samples/ko_6s.wav --only vllm \
    --out spikes/out/spike1_vllm.json
```

Run the transformers path in an r36.4.0 image:

```bash
python3 spikes/spike1_asr_load.py --wav samples/ko_6s.wav --only transformers \
    --out spikes/out/spike1.json
```

Merge the two results by copying the `vllm` key into `spike1.json`.

Acceptance criteria:

| Criterion | Requirement |
|---|---|
| Model load | Succeeds |
| N-best output | Exactly 5 sequences |
| Log-probabilities | Available per sequence |
| N-best transcription time | 900 ms or less for a 6 s utterance |

A path that transcribes but cannot produce N-best fails the requirement in the design
constraints and is not eligible, regardless of its latency.

## Spike 2 — flash-attn on sm_87

```bash
jetson-containers run $(autotag flash-attention)
python3 spikes/spike2_flash_attn.py --out spikes/out/spike2.json
```

The harness does not accept a successful import as evidence. An aarch64 wheel can import
and then fail inside the kernel, so a `flash_attn_func` call is executed and its output
checked for finiteness.

Qwen3-TTS is loaded three times, with `flash_attention_2`, `sdpa`, and `eager`. The
harness records which implementations load and the synthesis time for each. When no
implementation loads, MeloTTS is measured as the fallback.

Acceptance criteria:

| Criterion | Requirement |
|---|---|
| TTS backend | At least one of Qwen3-TTS or MeloTTS loads |
| Single-clause synthesis time | 300 ms or less |

If Qwen3-TTS loads without flash-attn, no further effort is spent on building flash-attn.
If no Qwen3-TTS configuration loads, Phases 1 through 3 proceed on MeloTTS and Qwen3-TTS
becomes a separate work item.

## Spike 3 — MoE against dense 14B

```bash
python3 spikes/spike3_llm_tokrate.py \
    --bin /opt/llama.cpp/build/bin \
    --models-dir ./models/gguf \
    --out spikes/out/spike3.json
```

Measurement conditions: context 2048, batch 1, 60 output tokens.

Two figures are recorded for each profile:

| Source | Measures |
|---|---|
| `llama-bench` | Raw generation rate |
| `llama-server` with a representative translation prompt | Time to first token and generation rate under production prompt shape |

The second figure governs the decision. Prompt processing time falls inside the 700 ms
allocated to correction and translation up to the first clause.

Acceptance criterion: 5 tok/s. Below that, clause-level streaming does not sustain
playback and the profile reverts to the dense 14B. The MoE is selected only when it meets
the threshold and is not slower than the dense model.

The MoE reads only active parameters, which favors it on a bandwidth-limited device.
Routing changes the set of experts touched per token, which reduces locality. The net
effect at Orin bandwidth is not predictable from these two properties, which is why it is
measured.

## Reporting

```bash
python3 spikes/report.py --dir spikes/out \
    --md spikes/out/PHASE0.md --patch spikes/out/local.yaml
cp spikes/out/local.yaml config/local.yaml
```

`report.py` uses only the standard library and may also be run on the development
workstation with `uv run`.

Outputs:

| File | Content |
|---|---|
| `spikes/out/PHASE0.md` | Result tables, verdicts, and a latency budget reconciliation |
| `spikes/out/local.yaml` | Configuration patch applying the three decisions |

`config/local.yaml` is the third configuration layer and overrides `config/default.yaml`.
Applying Phase 0 results requires no code change.

## Batch execution

`spikes/run_all.sh` executes whichever spikes the current container supports and then
generates the report. Because the three spikes require different images, running it once
per image and collecting results in `spikes/out` is the expected workflow.

```bash
LLAMA_BIN=/opt/llama.cpp/build/bin bash spikes/run_all.sh
```
