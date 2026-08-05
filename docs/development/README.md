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
