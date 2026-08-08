# Development

## Workstation Setup

Install dependencies with uv:

```bash
bash scripts/manage.sh setup workstation
```

The Jetson and A6000 NGC service images use Python 3.12. Source and wheel metadata remain
compatible with Python 3.10. Tests run without models, microphones, target hardware, or
network access.

The runtime uses aiotools 2.x for periodic task scheduling and cancellation on Python
3.11 and later. Python 3.10 uses the compatibility implementation in
`src/kotonoha/_async_tools.py`, so the minimum-version import contract remains unchanged.
The integration uses `create_timer()` with cancellation-on-delay policy for health probes
and `cancel_and_wait()` for service shutdown. See the
[aiotools documentation](https://aiotools.readthedocs.io/en/latest/) for the upstream API.

## Quality Gates

```bash
bash scripts/manage.sh check
```

Repository-specific coding, localization, test, and commit rules are defined in
[AGENTS.md](../../AGENTS.md).

## Continuous Integration

`.github/workflows/ci.yml` runs on every push to `main` and on every pull request. Each
gate is an independent job, so one failure does not hide the others.

| Job | Gate |
|---|---|
| `ruff` | `ruff check` over the repository |
| `lint` | Shell script syntax and translation catalog completeness |
| `guard` | Lock consistency, wheel catalog compilation, device Python import parity |
| `test` | `pytest` |

The `ruff`, `lint`, and `test` jobs run the same commands as
`bash scripts/manage.sh check`. `tests/kotonoha/test_ci_workflow.py` fails when a gate is
added to the management script but not to the workflow.

The `guard` job covers three contracts a workstation run cannot reach:

- `uv lock --check` rejects a lock that no longer matches `pyproject.toml`. A drifted lock
  resolves differently inside the deployment images.
- The wheel must contain a compiled `.mo` for every committed `.po`. `.mo` files are never
  committed, so deployment depends on `hatch_build.py` compiling them during install.
- Every module must import under the supported Python 3.10 minimum. Ruff enforces 3.10
  syntax but not standard library availability.

Jobs install from the locked dependency set with `uv sync --frozen`. CI does not resolve
dependencies, download models, or contact target hardware.

English source-string requirements, locale-specific style, terminology, protected
content, and the catalog workflow are defined in the
[Localization Guide](localization.md).

## Console Logging

Console records use the kernel ring buffer layout:

```
[   12.345678] INFO    asr: asr.started model=Qwen3-ASR-0.6B n_best=5
```

The stamp is seconds since process start. `logging.console_format` selects `dmesg` or
`json`; the JSONL files under `data/logs/` stay structured either way, because the metrics
and evaluation readers parse them.

`--debug` (or `logging.debug`) raises the level to DEBUG and adds the per-stage detail
column to the interpreter's turn stage panel.

## Browser Interface

`kotonoha web` serves a browser client that captures the microphone with
`getUserMedia` and plays synthesized speech through Web Audio. The host running the
server needs no audio device.

```bash
uv run kotonoha web --host 127.0.0.1 --port 8080 --sessions 4
bash scripts/manage.sh web
```

One WebSocket carries a session both ways. Binary frames are 32-bit float PCM and
text frames are control messages, which keeps microphone blocks off the JSON path.

| Concern | Contract |
|---|---|
| Capture rate | The client reports the rate its AudioContext actually used; the server resamples |
| Half-duplex | The client reports played samples, and the microphone reopens only once playback drains |
| Sessions | One orchestrator, shared-memory ring and session identifier per browser |
| Model services | Shared by every session, so `--sessions` trades latency for simultaneous speakers |
| Logs | One reader drains the process buffer and fans records out to every session |

The interface has no authentication and drives the model services, so it binds
loopback unless `--host` says otherwise. Browsers refuse `getUserMedia` on an
insecure origin other than `localhost`, so a remote listener needs HTTPS in front
of it.

## Evaluation Data

Record evaluation data with the production microphone in the production room. The target
data set contains 100 utterances per supported language with reference transcripts and
translations.

| Stage | Metric | Execution host |
|---|---|---|
| ASR | Character error rate through `jiwer` | Development workstation |
| Translation | COMET through `unbabel-comet` | Development workstation, offline batch |

BLEU is not the project translation metric. Target-hardware performance measurements use
the separate [Performance Measurement](../performance/measurement.md) procedure.
