# System Architecture

The [configuration architecture](configuration.md) defines SQLite persistence, hardware
profile selection, reset behavior, and legacy YAML migration.

## Processing Model

Kotonoha processes completed utterances. It does not implement simultaneous
interpretation. The audio frontend and orchestrator run in the selected client process,
and model services remain resident across turns. A browser session connects to the Web
process deployed with the model stack on the selected host.

```text
[microphone or browser MediaStream]
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
[speaker or browser Web Audio]
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

Browser sessions replace PortAudio with Web Audio adapters. Each connection owns an
orchestrator and unique shared-memory name. Browser playback acknowledges consumed
samples before the state machine reopens capture.

## Monitoring Path

Every resident model service exposes Prometheus metrics on its service port. The Web
process polls the four endpoints selected by the active hardware profile. It merges those
samples with Web and turn metrics, then exposes one deployment-wide `/metrics` endpoint
and bounded JSON history at `/api/monitoring`.

Hardware-profile labels identify every aggregated sample. A separate port `9091`
receiver remains available for headless deployments that do not run the Web process.

## Audio Transport

Co-located ASR services receive an `AudioRef` containing `{name, slot, seq, frames}` and
read PCM from POSIX shared memory. Browser capture and HTTP upload paths use binary PCM.
Neither path base64-encodes audio.

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

## Service Topology

Every deployment runs one Web process and one resident service for each model role. The
active hardware profile supplies the ASR, verification ASR, translation, and TTS
endpoints. The orchestrator calls each endpoint directly and does not maintain placement
modes or resident fallback clients.

Jetson and RTX A6000 deployments use host IPC for co-located shared-memory audio. Browser
traffic remains binary WebSocket or HTTP data. A service transport failure fails the
affected turn and leaves the state machine available for the next turn.

The ASR and translation services run separate vLLM engines. Their GPU memory utilization
limits are independent and must leave capacity for verification ASR and TTS. The checked-in
split is provisional until concurrent-residency measurements pass on each target.

## Hardware Profiles

Hardware identity and engine tuning are defined by hardware profiles under
`config/profiles/accelerators/<vendor>/<family>/<model>.yaml`. Profile identifiers use the
stable `<vendor>.<family>.<model>` format.

| Profile | Runtime | Current use |
|---|---|---|
| `nvidia.jetson.agx-orin` | CUDA, `sm_87` | Unified-memory appliance deployment |
| `nvidia.rtx.a6000` | CUDA, `sm_86` | Dedicated-memory server deployment |

The configuration loader synchronizes packaged base defaults and hardware profile
defaults into SQLite. Operator overrides form the final persistent layer. vLLM argument
construction consumes typed profile values and does not branch on product names. This
allows future CUDA, ROCm, Metal, and accelerator runtime profiles to add measured
settings without changing the translation service.

Profile selection uses `accelerator.profile`. `KOTONOHA_INITIAL_HARDWARE_PROFILE` seeds a
new database, and `KOTONOHA__ACCELERATOR__PROFILE` provides an explicit process-level
override. The Web configuration editor exposes registered profile identifiers.

On a multi-GPU A6000 host, `scripts/py/allocate_gpus.py` reads GPU UUID, model name, total
memory, and free memory from `nvidia-smi`. It applies per-role memory reservations and
selects the placement with the lowest projected maximum memory utilization. Compose pins
each service to one generated GPU UUID through `device_ids`. Cached UUID assignments avoid
placement changes across routine service restarts.

Network-accessible services require a bearer token when `KOTONOHA_SERVICE_TOKEN` is set. Plain HTTP
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
- Per-request model loading
- Vector databases or embedding models for glossary lookup
- Unvalidated JetPack, CUDA, Jetson Linux, or base-image upgrades
