# Development

## Workstation Setup

Install dependencies with uv:

```bash
uv sync
```

JetPack 7.2 containers use Python 3.12. Source syntax remains compatible with Python 3.10
for service images that use an older interpreter. Tests run without models, microphones,
target hardware, or network access.

## Quality Gates

```bash
uv run ruff check .
uv run pytest -q
uv run python scripts/i18n.py check
```

Repository-specific coding, localization, test, and commit rules are defined in
[AGENTS.md](../../AGENTS.md).

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
