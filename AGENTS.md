# AGENTS.md

Instructions for AI coding agents working in this repository. Applies to the entire tree.

## Project

Consecutive four-language offline speech interpreter for Jetson AGX Orin 64GB, with an
optional high-performance mode backed by an external RTX A6000. See `README.md` for the
architecture and `spikes/README.md` for the Phase 0 procedure.

The target hardware is not available from the development workstation. Device behavior is
verified by measurement on the Orin, not by inspection.

## Commands

| Task | Command |
|---|---|
| Install | `uv sync` |
| Test | `uv run pytest -q` |
| Lint | `uv run ruff check .` |
| Autofix | `uv run ruff check --fix .` |
| Environment report | `uv run kotonoha doctor` |
| Pipeline without a microphone | `uv run kotonoha replay <wav> --seconds 12` |
| Link measurement | `uv run kotonoha netcheck` |

`uv run ruff check .` and `uv run pytest -q` must both pass before any change is reported
as complete. Current baseline: 53 tests, zero lint findings.

Dependencies are managed with uv. Use `uv add`, `uv add --group dev`, and
`uv lock --upgrade-package`. Do not edit `uv.lock` by hand. Do not invoke `pip`.

## Language conventions

| Artifact | Language |
|---|---|
| Code comments and docstrings | English |
| Developer-facing exception and diagnostic strings | English |
| Commit messages | English |
| Documentation, including this file | English |
| TUI labels, CLI help text, CLI printed output | Korean |
| Generated report text in `spikes/report.py` and spike `verdict` fields | Korean |
| Glossary data and CJK test fixtures | Source language |

The split is deliberate: code is read by whoever edits it, product text is read by the
operator. Do not translate the Korean product text into English.

## Commit policy

Do not run `git commit`. Produce the commit message text; the repository owner applies it.
Do not run `git push`.

## Hard constraints

These originate in the project specification and are not optimization targets. Changing
any of them requires explicit instruction.

| Constraint | Location | Consequence of violation |
|---|---|---|
| VAD preroll 200-300 ms | `src/kotonoha/audio/vad.py` | Korean tense-stop onsets and the pause before a Japanese sokuon are clipped. Presents as an ASR quality defect. |
| ASR N-best 5 | `src/kotonoha/services/asr_server.py` | Removes the input the correction pass depends on. |
| Correction and translation in one LLM pass | `src/kotonoha/prompts/translate.py` | The translation stage amplifies correction-stage errors. |
| Clause-level streaming handoff | `src/kotonoha/core/clauses.py` | First audio waits for the complete LLM output. |
| Conditional cross-verification on the Orin | `src/kotonoha/core/quality.py` | Adds 0.8 s per turn. |
| Half-duplex gating in exactly one place | `Orchestrator._on_state_change` | TTS output re-enters the microphone and loops without bound. |
| Audio not base64-encoded | `src/kotonoha/shmring.py`, `src/kotonoha/transport.py` | 100-200 ms lost per turn. |

Prohibited without instruction:

- Cloud APIs for ASR, translation, or TTS
- Simultaneous interpretation policies such as AlignAtt or LocalAgreement
- Vector databases or embedding models; the glossary is a prompt prefix
- English-pivot translation
- Browser-based microphone capture
- Loading a model per request
- Raising JetPack, CUDA, or container base image versions

Accuracy work proceeds in this order: frontend, prompt and context, N-best correction,
model size. Do not begin with model size.

## Verification before implementation

Do not implement against an assumed API. Confirm the library interface, then write the
code.

`VllmBackend` in `src/kotonoha/services/asr_server.py` raises `NotImplementedError` and
must stay that way until Spike 1 runs on the target. Whether vLLM loads Qwen3-ASR on sm_87
and exposes N-best is the question the spike answers.

Do not report measured values that were not measured. Where a number is required and no
measurement exists, state that it is missing.

Confirmed as of 2026-08:

| Component | Identifier |
|---|---|
| Primary ASR | `Qwen/Qwen3-ASR-1.7B-hf`, `AutoProcessor.apply_transcription_request` |
| TTS | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, `qwen_tts.Qwen3TTSModel.generate_custom_voice` |
| MoE GGUF | `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` |
| Dense GGUF | `unsloth/Qwen3-14B-GGUF` |
| vLLM container | `ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04` |

## Configuration

Decisions pending measurement are expressed as configuration, not code branches. Three
layers merge in order, each overriding the previous: `config/default.yaml`, the file given
to `--config`, then `config/local.yaml`. Environment variables override all three and use
a double underscore for nesting.

When adding a setting:

1. Add the field to the pydantic model in `src/kotonoha/config.py`.
2. Add it to `config/default.yaml` with a comment explaining the value.
3. Overlay files state only differences. Do not duplicate the baseline.

## Code style

| Item | Rule |
|---|---|
| Line length | 100 |
| Target version | py310, enforced by ruff `target-version` |
| Lint rules | `E`, `F`, `I`, `UP`, `B` |
| Type annotations | `from __future__ import annotations` at the top of every module |
| Logging | structlog, event name first: `log.info("turn.finished", turn_id=...)` |

The device runs Python 3.10. The workstation runs 3.12. Do not use syntax unavailable in
3.10.

Comments state why, not what. A comment that restates the code is removed during review.

## Tests

`tests/` runs without models, without a microphone, and without network access. Preserve
that property.

| File | Covers |
|---|---|
| `test_vad_segmenter.py` | Preroll presence, end-of-utterance timing, short-noise rejection, PTT path |
| `test_clauses.py` | Clause boundaries, marker handling across stream deltas |
| `test_shmring.py` | Ring publish and read, overwrite detection, truncation |
| `test_routing_and_quality.py` | Language normalization, inheritance fallback, routing modes, verification gate |
| `test_state_and_metrics.py` | State transitions, five-point instrumentation, budget overrun reporting |
| `test_remote_mode.py` | Role placement, PCM encoding, failover behavior |

A change to any hard constraint requires a test that fails when the constraint is removed.

## Manual verification

`kotonoha replay` runs the full pipeline from a WAV file. With no services running, the
turn completes through the empty-ASR path and writes a turn record.

```bash
KOTONOHA__FRONTEND__VAD__BACKEND=energy uv run kotonoha replay probe.wav --seconds 12
```

`energy` is a development-only VAD fallback for workstations without `onnxruntime`. It is
not valid on the target.

No mock service harness is committed. Verifying a complete turn requires either the real
services or a locally written mock exposing `/health`, `/transcribe`,
`/transcribe/upload`, `/echo`, `/v1/chat/completions`, and `/synthesize`.

Inspect results in `data/logs/turns.jsonl`. Application logs are in
`data/logs/kotonoha.jsonl`. The files are separate so the turn log parses without
filtering.

## Known pitfalls

Each of these was encountered and fixed. Reintroducing them is a regression.

| Area | Pitfall |
|---|---|
| `shmring.py` | `np.frombuffer` returns a read-only array. Writable views require `np.ndarray(buffer=...)`. |
| `metrics.py` | `TurnLog.write` returns a record without an `event` key. Adding one collides with structlog in `log.info("turn", **rec)`. |
| `config.py` | pydantic-settings ranks initialization values above environment variables by default. `settings_customise_sources` reverses this. Do not remove it. |
| `config.py` | Overlay files are incomplete by design. `load_settings` must merge them over `config/default.yaml` or required fields are missing. |
| `audio/vad.py` | Preroll uses `math.ceil` plus one slot. The extra slot holds the frame that triggered onset, which belongs to the utterance. |
| `cli.py` | `replay` forces automatic mode. Push-to-talk has no key to press when reading a file. |
| `clients/router.py` | Lambdas passed to the router may be re-invoked on the fallback. Bind loop variables explicitly, for example `lambda c, text=clause: ...`. |
| Logging | The turn log and the application log must not share a path. |

## High-performance mode

`perf_mode` selects the role placement: `onboard`, `hybrid`, or `remote`. `hybrid` moves
only the LLM and keeps utterance audio on the device; prefer it when audio must not leave
the Orin.

When modifying this path:

- The orchestrator must not branch on placement. `AudioPayload` carries both the
  shared-memory reference and the PCM buffer, and the client selects.
- Every remote role keeps a loaded on-board counterpart. A transport failure retries the
  same call locally so the turn completes.
- Streams fail over only before the first chunk. After audio has started there is no
  rewind.
- Application errors, 4xx, are not transport failures and are not retried elsewhere.
- Record `placement` and `failovers` in the turn log. Without them, a turn served by the
  fallback is indistinguishable from one served by the A6000.

Services exposed on a network enforce a bearer token when `KOTONOHA_SERVICE_TOKEN` is set,
and log `auth.disabled` at startup when it is not. Do not remove that warning.

## Reporting work

State what was verified and how. Distinguish measured results from unverified code paths.
When a task is partially blocked, complete the remainder and state explicitly what was not
done and why.
