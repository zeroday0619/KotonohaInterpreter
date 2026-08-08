# Operator Guide

## Commands

Install the environment before running workstation commands:

```bash
uv sync
```

| Command | Function |
|---|---|
| `uv run kotonoha web` | Open the multi-session browser control center |
| `uv run kotonoha text "<utterance>"` | Interpret typed text |
| `uv run kotonoha replay <wav> --seconds 12` | Replay a WAV through the voice pipeline |
| `uv run kotonoha devices` | List audio devices |
| `uv run kotonoha doctor` | Report environment and service health |
| `uv run kotonoha netcheck` | Measure configured service latency and upload throughput |
| `uv run kotonoha serve <service>` | Start a Python ASR service |

`replay --seconds` accepts 0.1-600 seconds. The WAV loader reads only that duration and
feeds a bounded frame queue, so an oversized recording cannot allocate an unbounded
replay backlog.

## Web UI

The Web UI exposes interpreter, monitoring, configuration, history, operations, and
license areas. It also streams structured application logs into the interpreter page.
Live ASR and translation panes clear when a new turn starts. Completed turns remain in a
message-style conversation timeline below the live panes.

The interpreter page preserves the former keyboard workflow:

| Key | Action |
|---|---|
| `Space` | Toggle push-to-talk |
| `a` | Cycle push-to-talk, automatic, and text modes |
| `r` | Select the next target language |
| `c` | Clear the active source and translation |
| `h` | Show or hide recent turns |
| `t` | Enter or leave text mode |
| `Escape` | Leave text mode while an input has focus |
| `q` | Connect or disconnect the browser session |

The active-turn diagnostics show the four latency-budget comparisons, service readiness,
active hardware profile, and service endpoints. The
operations page runs replay, device, service, diagnostic, network, glossary, and shell
completion commands. It supports bounded file uploads, live output, cancellation, and
output clearing.

The browser owns microphone capture and audio playback. Each browser connection owns an
isolated orchestrator session and shared-memory ring. The resident ASR, verification ASR,
translation, and TTS services remain shared across sessions.

Theme selection supports system synchronization, explicit light mode, and explicit dark
mode. The browser stores the preference locally and sends no theme data to the server.
Input and output selectors use the browser media-device inventory. The audio test opens
the selected microphone, updates the level meter, and plays a low-volume tone through the
selected output when the browser implements `AudioContext.setSinkId`.

The monitoring page polls the Web metrics API every five seconds. It displays resident
service readiness, host and accelerator identity, CPU load, system and accelerator
memory, root-filesystem use, turn counters, latency-budget violations, and
first-audio p95 latency. Charts retain up to 60 minutes of in-process samples. Restarting
the Web process clears chart history but does not reset Prometheus counters in resident
services.

## Text Input

Text mode closes the microphone and enters `PROCESSING` directly from `IDLE`. It skips
capture, ASR, and cross-verification, then rejoins the voice path at correction and
translation.

Source-language selection follows this order:

1. Explicit `--from` or `session.text_source_language`
2. Unicode script detection
3. Previous language inheritance

## History

SQLite stores source text, translation, language decision provenance, ASR confidence,
verification status, timing data, hardware profile, service endpoints, and outcome. The Web interface
separates the current turn from completed history and provides filtering, pagination,
diagnostic details, JSONL export, and deletion.

## Configuration

Configuration layers resolve in this order, with later values taking precedence:

1. Base defaults synchronized from `config/default.yaml`
2. The selected hardware-profile defaults synchronized from `config/profiles/accelerators/`
3. Operator overrides stored in `data/kotonoha.db`
4. `KOTONOHA__*` environment variables

Set `accelerator.profile` using `<vendor>.<family>.<model>` naming. The Web configuration
page persists profile selection. `KOTONOHA_INITIAL_HARDWARE_PROFILE` seeds a new database.
`KOTONOHA__ACCELERATOR__PROFILE` forces a process-local selection without changing SQLite.

Nested environment keys use a double underscore:

```bash
KOTONOHA__ACCELERATOR__PROFILE=nvidia.rtx.a6000 \
KOTONOHA__SERVICES__LLM=http://127.0.0.1:8003 \
uv run kotonoha doctor
```

The Web configuration page exposes every editable `Settings` leaf. A field displays its
effective value, active profile default, and modified state. Saving validates the complete
candidate and stores only operator overrides in SQLite. Reset restores one field or every
field to the active profile defaults. Reset all preserves the selected profile.

The Web process applies runtime changes immediately and invokes `/admin/reload` for each
affected resident model service. Service administration endpoints accept only settings
owned by that service.

Qwen3-TTS voice selection is exposed as four local settings under `tts.voices`. The
configuration editor presents only the presets native to each target language:

| Voice | Character | Native language | Setting |
|---|---|---|---|
| `Vivian` | Bright young female voice | Chinese | `tts.voices.zh_tw` |
| `Serena` | Warm, gentle young female voice | Chinese | `tts.voices.zh_tw` |
| `Uncle_Fu` | Seasoned male voice with a mellow timbre | Chinese | `tts.voices.zh_tw` |
| `Dylan` | Youthful Beijing male voice | Chinese, Beijing | `tts.voices.zh_tw` |
| `Eric` | Lively Chengdu male voice | Chinese, Sichuan | `tts.voices.zh_tw` |
| `Ryan` | Dynamic male voice with rhythm | English | `tts.voices.en` |
| `Aiden` | Sunny American male voice | English | `tts.voices.en` |
| `Ono_Anna` | Playful female voice | Japanese | `tts.voices.ja` |
| `Sohee` | Warm female voice | Korean | `tts.voices.ko` |

Defaults are Sohee for Korean, Ryan for English, Ono_Anna for Japanese, and Vivian for
Traditional Chinese. Voice selection is Web policy and remains outside model-service
administration allowlists.

## Localization

English message identifiers are the source text. Korean, Japanese, and Traditional
Chinese translations live under `src/kotonoha/locale/`. Locale resolution order is:

1. `KOTONOHA_LANG`
2. `ui.language`
3. `LC_ALL`, `LC_MESSAGES`, or `LANG`
4. English

Typer renders command help during import. Set `KOTONOHA_LANG` to localize help output.
`--lang` controls command output after parsing.

Catalog maintenance:

```bash
uv run python scripts/py/i18n.py extract
uv run python scripts/py/i18n.py update
uv run python scripts/py/i18n.py compile
uv run python scripts/py/i18n.py check
```

Commit `.po` files. The build hook compiles `.mo` files during installation.
