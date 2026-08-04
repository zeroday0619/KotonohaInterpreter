# AGENTS.md

Repository-wide instructions for coding agents. More specific instructions in a child
directory take precedence for that directory.

## Mission

Kotonoha Interpreter is a consecutive four-language offline speech interpreter for an
NVIDIA Jetson AGX Orin 64GB. An external RTX A6000 provides an optional high-performance
mode. `README.md` defines the architecture and operator interface. `spikes/README.md`
defines Phase 0 hardware validation.

| Environment | Constraint |
|---|---|
| Development workstation | macOS, Python 3.12, no target GPU |
| Target device | JetPack 6.2, L4T r36.4.x, CUDA 12.6, aarch64, Python 3.10 |
| External server | RTX A6000, x86_64 |
| Runtime | Offline for ASR, translation, and TTS |

Hardware compatibility and performance require measurement on the target. Source review
does not verify Jetson behavior.

## Operating contract

- Inspect the current implementation and library interface before changing code.
- Do not infer hardware support, model compatibility, latency, throughput, or memory use.
- Preserve unrelated working-tree changes. Do not overwrite user-owned modifications.
- Ask before adding a production dependency.
- Manage dependencies with uv. Do not invoke `pip` or edit `uv.lock` manually.
- Do not run `git commit` or `git push`. Provide a one-line English Conventional Commit
  message when requested.
- Do not use `--no-verify` for any repository operation.
- Complete safe in-scope work before reporting any remaining blocker.
- Report the exact verification commands and results. Separate measured results from
  unverified code paths.

## Commands

| Task | Command |
|---|---|
| Install | `uv sync` |
| Test | `uv run pytest -q` |
| Lint | `uv run ruff check .` |
| Autofix | `uv run ruff check --fix .` |
| Environment report | `uv run kotonoha doctor` |
| Integrated TUI | `uv run kotonoha tui` |
| Configuration editor | `uv run kotonoha config` |
| History browser | `uv run kotonoha history browse` |
| Translation catalog check | `uv run python scripts/i18n.py check` |
| Translation catalog compile | `uv run python scripts/i18n.py compile` |
| Typed turn | `uv run kotonoha text "<utterance>"` |
| WAV replay | `uv run kotonoha replay <wav> --seconds 12` |
| External link measurement | `uv run kotonoha netcheck` |
| Deploy Jetson | `bash scripts/deploy.sh jetson` |
| Deploy A6000 | `bash scripts/deploy.sh a6000` |
| Uninstall Jetson | `bash scripts/deploy.sh uninstall jetson` |
| Uninstall A6000 | `bash scripts/deploy.sh uninstall a6000` |

Completion requires both commands:

```bash
uv run ruff check .
uv run pytest -q
```

Current baseline: 201 tests and zero lint findings.

## Non-negotiable constraints

Changing these constraints requires explicit user instruction and a regression test.

| Constraint | Authority | Failure mode |
|---|---|---|
| VAD preroll remains 200-300 ms | `src/kotonoha/audio/vad.py` | Clips Korean stop onsets and Japanese sokuon context |
| Primary ASR returns N-best 5 | `src/kotonoha/services/asr_server.py` | Removes correction-pass evidence |
| One LLM pass performs correction and translation | `src/kotonoha/prompts/translate.py` | Compounds transcription errors |
| LLM output reaches TTS by clause | `src/kotonoha/core/clauses.py` | Delays first audio until full completion |
| Cross-verification remains conditional on Orin | `src/kotonoha/core/quality.py` | Adds approximately 0.8 seconds to every turn |
| Half-duplex gating remains in `Orchestrator._on_state_change` | `src/kotonoha/core/orchestrator.py` | TTS re-enters the microphone and loops |
| Audio transport remains binary or shared memory | `src/kotonoha/shmring.py`, `src/kotonoha/transport.py` | Base64 adds avoidable latency and allocation |

Do not introduce:

- Cloud ASR, translation, or TTS APIs
- Simultaneous interpretation policies such as AlignAtt or LocalAgreement
- English-pivot translation
- Browser microphone capture
- Per-request model loading
- Vector databases or embedding models for the glossary
- Unvalidated JetPack, CUDA, L4T, or base-image upgrades

Apply accuracy work in this order: audio frontend, prompt and context, N-best correction,
then model size.

## Confirmed model interfaces

Identifiers confirmed as of 2026-08:

| Component | Identifier |
|---|---|
| Primary ASR | `Qwen/Qwen3-ASR-1.7B-hf` |
| ASR request API | `AutoProcessor.apply_transcription_request` |
| TTS | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| TTS API | `qwen_tts.Qwen3TTSModel.generate_custom_voice` |
| MoE GGUF | `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` |
| Dense GGUF | `unsloth/Qwen3-14B-GGUF` |
| Jetson vLLM image | `ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04` |

`VllmBackend` in `src/kotonoha/services/asr_server.py` must continue raising
`NotImplementedError` until Spike 1 verifies model loading and N-best output on sm_87.

## Python and source standards

| Area | Rule |
|---|---|
| Compatibility | Python 3.10 syntax and APIs |
| Line length | 100 |
| Ruff rules | `E`, `F`, `I`, `UP`, `B`, `TID252` |
| Module preamble | `from __future__ import annotations` |
| I/O | Async-first; use `async` and `await` for I/O |
| Imports | Use absolute imports; do not add parent-relative imports |
| Re-exports | Import implementation modules directly; avoid new `__init__.py` re-exports |
| Class state | Declare instance fields with types at class scope, then assign them in `__init__` |
| Logging | structlog with event first, for example `log.info("turn.finished", turn_id=...)` |

- Use full English words for identifiers. Do not introduce abbreviations such as `cfg`,
  `res`, `tmp`, `val`, or `btn`.
- Write comments and docstrings in English complete sentences.
- Comments explain rationale or constraints, not the code operation.
- Write developer diagnostics, exceptions, log events, and structured log fields in
  English.
- Do not add `BUILD` files under `src/`.
- Add a `BUILD` file when creating a new test directory. Use `python_tests()` for test
  modules and `python_testutils()` for test utilities.

## Configuration

Settings merge in this order, with later sources taking precedence:

1. `config/default.yaml`
2. The file passed through `--config`
3. `config/local.yaml`
4. Environment variables using `KOTONOHA__SECTION__FIELD`

Pending hardware decisions remain configuration values, not hard-coded branches.

When adding a setting:

1. Add the typed field to `src/kotonoha/config.py`.
2. Add the baseline value and rationale to `config/default.yaml`.
3. Keep overlay files limited to differences from the baseline.
4. Confirm that the local configuration TUI exposes the new leaf.
5. Add a specific field description only when the generic type description is
   insufficient.

The configuration editor writes local changes to `config/local.yaml`. Remote changes use
the authenticated `/admin/config` endpoint and write
`config/remote-server.local.yaml`. Neither path writes baseline or selected overlay
files.

The remote allowlist contains only settings consumed by resident model services. Do not
expose credentials, client policy, audio devices, or local storage. Remote model changes
take effect after service restart. llama.cpp settings are mirrored to
`config/remote-llm.env`.

## Localization

Operator-facing CLI and TUI text uses gettext with English source strings as message ids.
Supported locales are `en`, `ko`, `ja`, and `zh-TW`.

| Path | Purpose |
|---|---|
| `src/kotonoha/i18n.py` | Locale resolution and `_`, `N_`, `pgettext` |
| `src/kotonoha/locale/kotonoha.pot` | Generated extraction template |
| `src/kotonoha/locale/<lang>/LC_MESSAGES/kotonoha.po` | Translation source |
| `src/kotonoha/locale/<lang>/LC_MESSAGES/kotonoha.mo` | Generated, gitignored catalog |
| `scripts/i18n.py` | Extract, update, compile, and check workflow |

Locale resolution order is `KOTONOHA_LANG`, `ui.language`, system locale, then English.

For a new operator-facing string:

1. Add the English source inline through `_()`.
2. Use `N_()` for strings stored in import-time tables, then call `_()` when rendering.
3. Run `uv run python scripts/i18n.py extract` and
   `uv run python scripts/i18n.py update`.
4. Translate every `.po`; use Taiwanese vocabulary in `zh_TW`.
5. Run `uv run python scripts/i18n.py compile` and
   `uv run python scripts/i18n.py check`.
6. Commit `.po` files only. The build compiles `.mo` files.

Do not localize dotted configuration paths, identifiers shared with YAML, log event names,
or structured log fields. Spike verdicts remain Korean in `PHASE0.md` and
`spikes/report.py`.

## Runtime contracts

### Text input

`session.mode` supports `push_to_talk`, `auto`, and `text`.

- Text mode closes the microphone gate and accepts keyboard input.
- Text turns move directly from `IDLE` to `PROCESSING`.
- `decide_typed_language` resolves explicit language, script detection, then inherited
  language.
- Text mode skips capture, ASR, and cross-verification.
- Voice and text paths join at `Orchestrator._route_and_translate` and share translation,
  TTS, and `_finish`.
- `TurnMetrics.input_mode` records `voice` or `text`; text turns record null
  `audio_seconds` and a zero ASR stage.

Do not duplicate the translation and TTS tail for text mode.

### High-performance mode

`perf_mode` controls role placement:

| Mode | Placement |
|---|---|
| `onboard` | All roles on Jetson |
| `hybrid` | LLM remote; audio roles remain on Jetson |
| `remote` | ASR, verification ASR, LLM, and TTS remote |

Preserve these contracts:

- The orchestrator does not branch on placement. Clients select from `AudioPayload`,
  which carries shared-memory and PCM representations.
- Every remote role retains a loaded onboard fallback.
- Transport failures retry locally. HTTP 4xx application errors do not.
- Streaming failover occurs only before the first emitted chunk.
- Turn logs record `placement` and `failovers`.
- Network services enforce bearer authentication when `KOTONOHA_SERVICE_TOKEN` is set.
- Services log `auth.disabled` when authentication is not configured.

## Tests

The test suite must run without models, microphones, target hardware, or network access.
`tests/conftest.py` sets `KOTONOHA_SKIP_LOCAL_CONFIG` so device endpoints and tokens from
`config/local.yaml` never enter tests.

| Test module | Contract |
|---|---|
| `test_vad_segmenter.py` | Preroll, EOU timing, noise rejection, PTT |
| `test_clauses.py` | Clause boundaries and streamed marker handling |
| `test_shmring.py` | Publish, read, overwrite detection, truncation |
| `test_routing_and_quality.py` | Language routing and verification gate |
| `test_state_and_metrics.py` | State transitions and five-point timing |
| `test_remote_mode.py` | Placement, transport encoding, failover |
| `test_config_admin.py` | Remote authorization, allowlist, validation, persistence |
| `test_deploy_script.py` | Deployment interface and host templates |
| `test_i18n.py` | Catalog completeness, placeholders, locale resolution |
| `test_history.py` | History filters, escaping, panel, browser |
| `test_text_mode.py` | Script detection, typed routing, text input |
| `test_tui.py` | Composition, bindings, localized labels |
| `test_tui_logging.py` | JSON buffering, formatting, file preservation |
| `test_tui_license.py` | Project and dependency license discovery |
| `test_tui_rendering.py` | Frame coalescing and level interpolation |
| `test_tui_tools.py` | Operations command construction and validation |
| `test_tui_workflow.py` | Control-center sequencing and settings reload |

## Target verification

Use `kotonoha replay` for the workstation regression path:

```bash
KOTONOHA__FRONTEND__VAD__BACKEND=energy \
  uv run kotonoha replay probe.wav --seconds 12
```

The `energy` backend is workstation-only and does not validate the target VAD.

A complete turn requires real services or a local harness implementing `/health`,
`/transcribe`, `/transcribe/upload`, `/echo`, `/v1/chat/completions`, and `/synthesize`.
No mock service harness is committed.

| Output | Path |
|---|---|
| Application log | `data/logs/kotonoha.jsonl` |
| Turn metrics | `data/logs/turns.jsonl` |

Do not mix application logs and turn metrics in one file.

## Regression traps

| Area | Required behavior |
|---|---|
| Shared memory | Use `np.ndarray(buffer=...)` for writable views; `np.frombuffer` is read-only |
| Metrics | Do not add `event` to `TurnLog.write`; it collides with structlog |
| Settings sources | Keep environment values above initialization values |
| YAML overlays | Merge incomplete overlays over `config/default.yaml` |
| VAD preroll | Keep `math.ceil` plus the onset-trigger frame slot |
| Replay | Force automatic mode because files have no PTT key event |
| Client fallback | Bind loop values passed through retryable lambdas |
| Textual bindings | Replace `BindingsMap`; do not mutate the shared class map with `bind()` |
| Hidden input | Remove focusability when hidden or it consumes global bindings |
| Text input | Keep the exit binding at priority because focused inputs consume keys |
| Typer localization | Help is rendered at import time; use `KOTONOHA_LANG` for localized help |
| gettext catalogs | Compile after `.po` changes; stale `.mo` files serve old text |
| History search | Escape `%` and `_` before SQL LIKE matching |
| Test isolation | Keep `KOTONOHA_SKIP_LOCAL_CONFIG`; local config contains device secrets |

## Documentation and reporting

- Write documentation, comments, docstrings, PR text, and commit messages in English.
- Follow the existing repository document style. Use concise contract-level descriptions.
- Lead with the result. Avoid repeating implementation details already evident in code.
- Use tables or lists when they improve scanning.
- State verified commands and results at handoff.
- Identify target-only behavior that remains unmeasured.
