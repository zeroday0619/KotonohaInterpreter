# Observability

## Turn Timeline

Every turn records five timestamps:

```text
EOU detected -> ASR complete -> first clause -> first audio packet -> queue drained
```

The turn record also contains detected language, LID confidence, ASR average
log-probability, verification activation, audio duration, output token count, placement,
failovers, and outcome.

## Storage

| Path | Content |
|---|---|
| `data/logs/kotonoha.jsonl` | Structured application events |
| `data/logs/turns.jsonl` | One structured metrics record per turn |
| `data/kotonoha.db` | Glossary and interpretation history |

Application events and turn metrics use separate files. This separation allows turn
records to parse without filtering application events.

## Latency Analysis

The end-of-utterance latency objective is 2.9 seconds to the first audio packet. Use the
five timestamps to identify the stage that exceeded its budget. Do not attribute an
overrun to model inference without a stage measurement.

Cross-host analysis requires synchronized clocks on the Jetson and A6000. Preserve the
application log, turn record, service logs, placement, and failover fields for incident
analysis.
