# Development

## Workstation Setup

Install dependencies with uv:

```bash
bash scripts/manage.sh setup workstation
```

The selected Jetson r36.4 service image uses Python 3.10. The JetPack 7.2 host and A6000
service images use Python 3.12. Source syntax remains compatible with Python 3.10. Tests
run without models, microphones, target hardware, or network access.

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
- Every module must import under Python 3.10, the generation supplied by the Jetson r36.4
  service image. Ruff enforces 3.10 syntax but not standard library availability.

Jobs install from the locked dependency set with `uv sync --frozen`. CI does not resolve
dependencies, download models, or contact target hardware.

English source-string requirements, locale-specific style, terminology, protected
content, and the catalog workflow are defined in the
[Localization Guide](localization.md).

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
