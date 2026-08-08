# Operator Guide

## Commands

Install the environment before running workstation commands:

```bash
uv sync
```

| Command | Function |
|---|---|
| `uv run kotonoha tui` | Open the integrated control center |
| `uv run kotonoha run` | Open the interpreter directly |
| `uv run kotonoha config` | Edit local or remote configuration |
| `uv run kotonoha history browse` | Browse completed turns |
| `uv run kotonoha text "<utterance>"` | Interpret typed text |
| `uv run kotonoha replay <wav> --seconds 12` | Replay a WAV through the voice pipeline |
| `uv run kotonoha devices` | List audio devices |
| `uv run kotonoha doctor` | Report environment and service health |
| `uv run kotonoha netcheck` | Measure remote latency and upload throughput |
| `uv run kotonoha serve <service>` | Start a Python ASR service |

`replay --seconds` accepts 0.1-600 seconds. The WAV loader reads only that duration and
feeds a bounded frame queue, so an oversized recording cannot allocate an unbounded
replay backlog.

## Integrated TUI

The integrated TUI provides interpreter, configuration, history, operations, and license
views. Structured JSON logs render as bounded, human-readable records in the interpreter
footer without writing terminal control output to the application log.

Live ASR and translation panes clear when a new turn starts. Completed turns remain in a
message-style conversation timeline below the live panes.

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
verification status, timing data, placement, failovers, and outcome. The TUI separates
the current turn from completed history. Each history entry renders the source and
translation as distinct messages in chronological order.

## Configuration

Configuration layers merge in this order, with later values taking precedence:

1. `config/default.yaml`
2. The selected accelerator profile under `config/profiles/accelerators/`
3. The file selected with `--config` or `KOTONOHA_CONFIG`
4. `config/local.yaml`, or `KOTONOHA_LOCAL_CONFIG`
5. `KOTONOHA__*` environment variables

Set `accelerator.profile` using `<vendor>.<family>.<model>` naming. The environment
variable `KOTONOHA__ACCELERATOR__PROFILE` selects a profile without editing YAML.

Nested environment keys use a double underscore:

```bash
KOTONOHA__PERF_MODE=hybrid \
KOTONOHA__REMOTE__ENABLED=true \
KOTONOHA__REMOTE__SERVICES__LLM=http://a6000.internal:8003 \
uv run kotonoha doctor
```

Use `custom` mode to select each model role independently:

```yaml
perf_mode: custom
placement:
  asr: remote
  asr_verify: local
  llm: remote
  tts: local
```

Set `remote.enabled: true` when any role uses the remote placement. Unspecified roles
inherit the local placement.

`kotonoha config` exposes every local `Settings` leaf. Local changes are validated and
written atomically to `config/local.yaml`. Remote changes use the authenticated
`/admin/config` endpoint and only accept settings consumed by resident model services.
Remote model changes require a service restart.

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
Traditional Chinese. Voice selection is client policy, so it remains in the Jetson local
configuration rather than the remote model-service allowlist.

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
