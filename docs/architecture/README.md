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
    |-- :8001 primary ASR       target vLLM model, N-best 5 and realtime WS
    |-- :8002 verification ASR  faster-whisper large-v3, conditional
    |-- :8003 translation LLM   vLLM, correction and translation
    `-- :8004 TTS               vLLM-Omni, Qwen3-TTS 0.6B
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

On a multi-GPU A6000 host, `scripts/allocate_gpus.py` reads GPU UUID, model name, total
memory, and free memory from `nvidia-smi`. It applies per-role memory reservations and
selects the placement with the lowest projected maximum memory utilization. Compose pins
each service to one generated GPU UUID through `device_ids`. Cached UUID assignments avoid
placement changes across routine service restarts.

Remote services require a bearer token when `KOTONOHA_SERVICE_TOKEN` is set. Plain HTTP
does not provide confidentiality. Restrict service ports to a trusted network or place a
TLS reverse proxy in front of them.

## Model Identifiers

| Component | Identifier |
|---|---|
| Jetson primary ASR, vLLM | `Qwen/Qwen3-ASR-0.6B` |
| Jetson ASR, Transformers fallback | `Qwen/Qwen3-ASR-0.6B-hf` |
| A6000 primary ASR, vLLM | `mistralai/Voxtral-Mini-4B-Realtime-2602` |
| TTS | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| Jetson translation model | `google/translategemma-4b-it` |
| A6000 translation model | `google/translategemma-12b-it` |
| Translation runtime | In-process vLLM and FastAPI `/v1/realtime` WebSocket |
| Jetson vLLM image | `nvcr.io/nvidia/vllm:26.07-py3` |
| A6000 vLLM image | `nvcr.io/nvidia/vllm:26.07-py3` |
| TTS service image | `kotonohainterpreter-tts:latest` |
| TTS runtime base | `vllm/vllm-omni:v0.26.0` |

The TTS FastAPI service owns `AsyncOmni` and `OmniOpenAIServingSpeech` in-process. It
exposes their speech stream directly and does not run a nested vLLM-Omni HTTP server.

The ASR FastAPI service similarly owns one vLLM `AsyncLLM` engine in-process. The same
resident engine serves N-best five batch transcription and the vLLM realtime WebSocket
protocol at `/v1/realtime`; no nested vLLM HTTP server or per-request model load is used.
The wrapper accepts both the vLLM 0.19 OpenAI module layout and the current
`speech_to_text` layout because these are internal, version-sensitive interfaces.

The translation FastAPI service owns a separate vLLM asynchronous engine and applies
TranslateGemma's structured chat template with source and target language codes. Its
`/v1/realtime` WebSocket emits translation deltas from the same resident engine; the
compatibility HTTP stream does not start another model server.

## Scope Exclusions

- Cloud ASR, translation, or TTS APIs
- Simultaneous interpretation policies
- English-pivot translation
- Browser microphone capture
- Per-request model loading
- Vector databases or embedding models for glossary lookup
- Unvalidated JetPack, CUDA, Jetson Linux, or base-image upgrades
