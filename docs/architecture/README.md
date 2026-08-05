# System Architecture

## Processing Model

Kotonoha processes completed utterances. It does not implement simultaneous
interpretation. The audio frontend and orchestrator run on the Jetson, and model services
remain resident across turns.

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
    |-- :8003 translation LLM   vLLM, correction and translation
    `-- :8004 TTS               Qwen3-TTS 0.6B or MeloTTS
    |
    v
[speaker and terminal UI]
```

## Turn Workflow

| Stage | Contract |
|---|---|
| Capture | PortAudio captures 48 kHz mono audio |
| Segmentation | VAD includes 200-300 ms preroll and closes after 800 ms silence |
| Primary ASR | ASR returns five hypotheses, average log-probability, and a language label |
| Language decision | Short or low-confidence input inherits the previous language |
| Verification | Verification runs only when the quality gate activates on the Jetson |
| Translation | One LLM pass corrects the transcription and translates directly |
| Streaming | Complete clauses reach TTS before LLM completion |
| Playback | The microphone remains closed until the TTS queue is empty |

The state machine permits these transitions:

```text
IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE
  `----------------> PROCESSING                 typed input
```

`Orchestrator._on_state_change` owns half-duplex microphone gating. TTS output cannot
re-enter capture while the state is `SPEAKING`.

## Audio Transport

Local ASR services receive an `AudioRef` containing `{name, slot, seq, frames}` and read
PCM from POSIX shared memory. Remote ASR services receive binary multipart PCM. Neither
path base64-encodes audio.

The orchestrator publishes both representations in `AudioPayload`. Client routing selects
the representation, so role placement does not branch the processing pipeline.

## Accuracy Contracts

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
enforce Taiwanese vocabulary, including `軟體`, `影片`, `資訊`, and `滑鼠`.

## Runtime Placement

`perf_mode` controls service placement.

| Mode | ASR | Verification | LLM | TTS | Audio leaves Jetson |
|---|---|---|---|---|---|
| `onboard` | Jetson | Jetson | Jetson | Jetson | No |
| `hybrid` | Jetson | Jetson | A6000 | Jetson | No |
| `remote` | A6000 | A6000 | A6000 | A6000 | Yes |

Each remote role retains a resident Jetson fallback. Transport failures retry locally.
HTTP 4xx application errors do not activate failover. Streaming roles fail over only
before the first chunk because an active stream cannot be rewound.

The ASR and translation services run separate vLLM engines. Their GPU memory utilization
limits are independent and must leave capacity for verification ASR and TTS. The checked-in
split is provisional until concurrent-residency measurements pass on each target.

Remote services require a bearer token when `KOTONOHA_SERVICE_TOKEN` is set. Plain HTTP
does not provide confidentiality. Restrict service ports to a trusted network or place a
TLS reverse proxy in front of them.

## Model Identifiers

| Component | Identifier |
|---|---|
| Primary ASR, vLLM | `Qwen/Qwen3-ASR-1.7B` |
| Primary ASR, Transformers fallback | `Qwen/Qwen3-ASR-1.7B-hf` |
| TTS | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| MoE translation model | `ELVISIO/Qwen3-30B-A3B-Instruct-2507-AWQ` |
| Dense translation model | `Qwen/Qwen3-14B-AWQ` |
| Jetson vLLM image | `ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04` |
| A6000 vLLM image | `vllm/vllm-openai:v0.19.1` |

## Scope Exclusions

- Cloud ASR, translation, or TTS APIs
- Simultaneous interpretation policies
- English-pivot translation
- Browser microphone capture
- Per-request model loading
- Vector databases or embedding models for glossary lookup
- Unvalidated JetPack, CUDA, L4T, or base-image upgrades
