# Implementation Plan

## Objective

This plan defines the implementation and acceptance sequence for the consecutive
four-language interpreter. Phase completion requires measured evidence from the named
execution environment. Source implementation alone does not satisfy a phase gate.

## Governance

1. Execute one phase at a time.
2. Complete Phase 0 first and obtain explicit approval before accepting Phase 1.
3. Produce a phase report that states what works, what fails, and what remains unverified.
4. Do not advance when a required measurement, artifact, or approval is missing.
5. Record hardware compatibility and performance only from the target host.
6. Preserve accuracy contracts while optimizing latency or resource use.

Implementation can exist ahead of the accepted phase. Such code remains provisional
until the corresponding phase exit criteria pass on the target.

## Current Status

Status reflects repository inspection on 2026-08-04. No target acceptance result is
inferred from workstation tests.

| Phase | Source state | Acceptance state | Primary gap |
|---|---|---|---|
| Phase 0 | Spike harness and report generator implemented | Pending | Jetson target execution and approval |
| Phase 1 | Frontend, PTT, ASR, LID, and evaluation tools implemented | Pending | Real microphone, room, and Jetson measurements |
| Phase 2 | Translation, clause streaming, TTS, and timing records implemented | Pending | End-of-utterance to first-audio measurement |
| Phase 3 | State machine, half-duplex gate, automatic mode, and fallbacks implemented | Pending | Device-level loop and failure-path validation |
| Phase 4 | Four-language routing, Traditional Chinese conversion, and three routing modes implemented | Pending | Twelve-direction language acceptance |
| Phase 5 | N-best correction, conditional verification, glossary, and six-turn context implemented | Partial | Conditional back-translation and evaluation results |

## Phase Sequence

```text
Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5
   |
   `-> A6000 performance workstream, without replacing Jetson acceptance
```

## Phase 0: Hardware Validation

### Objective

Resolve hardware-dependent runtime decisions before feature acceptance begins.

### Scope

- Load Qwen3-ASR 0.6B through Jetson vLLM and return five scored hypotheses.
- Load Voxtral Mini 4B Realtime through A6000 vLLM.
- Execute the embedded vLLM realtime WebSocket path on both targets.
- Measure the Transformers fallback when vLLM fails or does not satisfy N-best.
- Execute FlashAttention on sm_87 and measure vLLM-Omni Qwen3-TTS PCM streaming.
- Measure MoE 30B-A3B and dense 14B AWQ generation through vLLM under the specified
  conditions.
- Select configuration values from measured results.

### Deliverables

| Artifact | Purpose |
|---|---|
| `spikes/out/spike1.json` | ASR compatibility, N-best, realtime events, scores, and latency |
| `spikes/out/spike2.json` | FlashAttention and TTS backend evidence |
| `spikes/out/spike3.json` | MoE and dense LLM measurements |
| `spikes/out/PHASE0.md` | Consolidated verdict and unresolved failures |
| `spikes/out/local.yaml` | Measured Jetson configuration decisions |

### Exit Criteria

- Every spike records the target, runtime versions, conditions, result, and failure text.
- ASR provides exactly five hypotheses or the approved fallback is documented.
- TTS has an executable backend and a measured synthesis result.
- The selected LLM path sustains at least 5 tok/s under the Phase 0 conditions.
- Thermal throttling is absent during retained measurements.
- The phase report receives explicit approval.

Use [Performance Measurement](../performance/measurement.md) for commands, thresholds,
and report generation. Stop after reporting Phase 0 until approval is recorded.

## Phase 1: English-Korean Minimum Path

### Objective

Validate utterance capture, end-of-utterance detection, primary ASR, and language
identification before translation and speech synthesis affect diagnosis.

### Scope

- Use push-to-talk as the initial interaction mode.
- Run the audio frontend, primary ASR, and LID only for acceptance measurements.
- Compare captures with and without 200-300 ms preroll.
- Record end-of-utterance false closure, missed closure, and first-syllable clipping.
- Begin the four-language evaluation data set with the production microphone and room.

### Deliverables

- English and Korean capture samples with reference transcripts.
- Preroll comparison evidence for Korean stop onsets and equivalent English samples.
- ASR CER, LID decisions, LID confidence, audio duration, and ASR latency.
- A Phase 1 report listing end-of-utterance and LID failure patterns.

### Exit Criteria

- Push-to-talk completes repeated turns without losing the onset frame.
- The retained configuration preserves at least 200 ms of preroll.
- Empty ASR output returns to `IDLE` without playback.
- LID fallback inherits the previous language for sub-second or low-confidence input.
- Evaluation manifests are reproducible and contain references for recorded samples.

## Phase 2: Clause-Streaming Chain

### Objective

Connect correction, direct translation, clause-level handoff, TTS, and playback while
meeting the first-audio latency objective.

### Scope

- Perform transcription correction and translation in one LLM request.
- Stream complete clauses to TTS before LLM completion.
- Record all five turn timestamps.
- Measure each model stage and the complete end-of-utterance path.

### Deliverables

- Full English-Korean interpreted turns in both directions.
- Turn records containing EOU, ASR completion, first clause, first audio, and queue drain.
- Stage-level overrun evidence for every turn that misses the latency objective.
- A Phase 2 report with measured first-audio latency.

### Exit Criteria

- First audio begins within 3.0 seconds of end-of-utterance under accepted test conditions.
- LLM generation sustains playback without clause starvation at the selected speech rate.
- TTS receives clauses before the complete translation finishes.
- A timeout produces transcript-only output and returns to `IDLE`.

## Phase 3: Gating and State Machine

### Objective

Validate unattended turn control and failure recovery without microphone feedback loops.

### Scope

- Enforce `IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE`.
- Keep half-duplex microphone gating in `Orchestrator._on_state_change`.
- Transition from push-to-talk acceptance to automatic VAD operation.
- Exercise ASR, LLM, TTS, GPU OOM, transport, and playback failure paths.

### Deliverables

- State transition logs for normal and failed turns.
- Playback-loop test evidence using the production microphone and speaker placement.
- Timeout, fallback, service restart, and discarded-turn records.
- A Phase 3 report covering every defined failure condition.

### Exit Criteria

- Capture remains blocked for the entire `SPEAKING` state.
- TTS output does not create a new utterance or interpretation loop.
- Automatic mode detects speech and closes the turn after 800 ms silence.
- Every failure path returns to `IDLE` or restarts the failed service without deadlock.
- A failed TTS path activates the configured fallback before the first audio chunk.

## Phase 4: Four-Language Expansion

### Objective

Accept Korean, English, Traditional Chinese for Taiwan, and Japanese across all direct
translation directions and routing modes.

### Scope

- Validate all 12 direct translation directions without an English pivot.
- Apply a Traditional Chinese ASR prompt and OpenCC `s2twp` to ASR and translation output.
- Enforce Taiwanese glossary terms such as `軟體`, `影片`, `資訊`, and `滑鼠`.
- Validate pair, fixed-target, and broadcast routing.
- Display the resolved source and target languages in the TUI.

### Deliverables

- Direction-by-direction ASR, translation, and TTS results.
- Traditional Chinese conversion and Taiwanese terminology evidence.
- Routing results for pair, fixed-target, and broadcast sessions.
- A Phase 4 report listing language-specific defects and accepted exceptions.

### Exit Criteria

- Every direct language direction completes without an English-pivot request.
- Traditional Chinese output passes `s2twp` and the configured Taiwan vocabulary rules.
- Short-utterance LID fallback works across all four languages.
- Broadcast produces one output per configured target without reprocessing ASR.

## Phase 5: Accuracy Tuning

### Objective

Improve transcription correction and translation accuracy without hiding regressions
behind larger models.

### Scope

- Supply ASR N-best 5, optional verification output, glossary, and six-turn history to one
  correction-and-translation pass.
- Invoke verification ASR only below the configured confidence threshold or when N-best
  disagreement activates the quality gate.
- Add back-translation validation only for turns containing numbers, dates, or proper
  names.
- Tune the frontend, prompt, context, and correction policy before changing model size.
- Complete the four-language evaluation set with 100 utterances per language.

### Deliverables

- Versioned ASR CER results for each language.
- Versioned COMET results for all translation directions under evaluation.
- Verification activation rate, divergence rate, and latency cost.
- Error analysis for numbers, dates, proper names, homophones, and Traditional Chinese.
- A Phase 5 report comparing each change against the accepted baseline.

### Exit Criteria

- Every accepted change preserves or improves the recorded evaluation baseline.
- Verification remains conditional on the Jetson and reports its activation reason.
- Number, date, and proper-name turns record the back-translation decision and result.
- COMET runs offline on the development workstation, not on the Orin.
- Accuracy changes do not violate the Phase 2 latency acceptance without explicit review.

## A6000 Performance Workstream

The external server is a parallel deployment workstream. It does not replace Jetson
fallbacks or phase acceptance.

| Gate | Required evidence |
|---|---|
| Model-stage performance | A6000 ASR, TTS, MoE, and dense measurements |
| Concurrent residency | All four services healthy with `nvidia-smi` evidence |
| Link acceptance | Jetson `netcheck` RTT and six-second PCM upload results |
| Placement | Accepted `onboard`, `hybrid`, or `remote` configuration |
| Recovery | Transport failures complete through resident Jetson fallbacks |
| Security | Bearer authentication and restricted service ports |

Store A6000 results under `spikes/out/a6000`. Do not combine server inference and network
overhead into one unlabelled measurement.

## Reporting Contract

Every phase report contains:

- Source revision and configuration overlays
- Host, GPU, runtime, and model identifiers
- Test data and measurement conditions
- Measured results separated from unverified behavior
- Passed and failed exit criteria
- Known defects, fallback decisions, and deferred work
- Approval state and the next permitted phase

The report must state that a phase remains pending when required target evidence is
missing, even when workstation tests pass.
