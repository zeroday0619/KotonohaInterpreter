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
| `uv run kotonoha serve <service>` | Start a Python model service |

## Integrated TUI

The integrated TUI provides interpreter, configuration, history, operations, and license
views. Structured JSON logs render as bounded, human-readable records in the interpreter
footer without writing terminal control output to the application log.

Live ASR and translation panes clear when a new turn starts. Completed turns remain
available in history.

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
the current turn from completed history.

## Configuration

Configuration layers merge in this order, with later values taking precedence:

1. `config/default.yaml`
2. The file selected with `--config` or `KOTONOHA_CONFIG`
3. `config/local.yaml`, or `KOTONOHA_LOCAL_CONFIG`
4. `KOTONOHA__*` environment variables

Nested environment keys use a double underscore:

```bash
KOTONOHA__PERF_MODE=hybrid \
KOTONOHA__REMOTE__ENABLED=true \
KOTONOHA__REMOTE__SERVICES__LLM=http://a6000.internal:8003 \
uv run kotonoha doctor
```

`kotonoha config` exposes every local `Settings` leaf. Local changes are validated and
written atomically to `config/local.yaml`. Remote changes use the authenticated
`/admin/config` endpoint and only accept settings consumed by resident model services.
Remote model changes require a service restart.

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
uv run python scripts/i18n.py extract
uv run python scripts/i18n.py update
uv run python scripts/i18n.py compile
uv run python scripts/i18n.py check
```

Commit `.po` files. The build hook compiles `.mo` files during installation.
