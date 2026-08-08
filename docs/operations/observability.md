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

Both JSONL streams rotate at `logging.max_bytes`. The runtime retains
`logging.backup_count` prior generations with numeric suffixes. The default policy limits
each stream to one active 64 MiB file and five prior generations.

`store.maximum_turns` and `store.maximum_sessions` bound SQLite growth. Pruning runs
inside the existing write transaction and retains the newest records.

## Latency Analysis

The end-of-utterance latency objective is 2.9 seconds to the first audio packet. Use the
five timestamps to identify the stage that exceeded its budget. Do not attribute an
overrun to model inference without a stage measurement.

Cross-host analysis requires synchronized clocks on the Jetson and A6000. Preserve the
application log, turn record, service logs, placement, and failover fields for incident
analysis.

## Prometheus Metrics

Every A6000 resident FastAPI service exposes Prometheus text metrics at its existing
service port:

| Service | Endpoint |
|---|---|
| ASR | `http://127.0.0.1:8001/metrics` |
| Verification ASR | `http://127.0.0.1:8002/metrics` |
| Translation LLM | `http://127.0.0.1:8003/metrics` |
| TTS | `http://127.0.0.1:8004/metrics` |

The endpoint uses the official `prometheus-client` library. It reports HTTP request
counts and durations, service readiness, accelerator backend and memory architecture,
system and accelerator memory, configured engine limits, turn stage durations, output
tokens, generation rate, ASR log probability, conditional cross-verification, failovers,
and latency-budget violations.

The `/metrics` endpoint follows the same bearer-token middleware as other service routes.
When `KOTONOHA_SERVICE_TOKEN` is set, configure the Prometheus scrape job with the same
token. The `/health` endpoint remains unauthenticated for deployment probes.

The A6000 Compose deployment starts a dedicated metrics receiver on port `9091` when
`logging.prometheus_port` is set to a valid port. The receiver polls all four resident
service endpoints over the Docker network, caches the latest successful payload, and
exposes the service families through one unified endpoint. Aggregated samples include
bounded `role` and `source` labels, where `source` is `a6000`. A failed scrape removes the
stale payload and sets `kotonoha_remote_metrics_scrape_up` to `0`.

The receiver binds to `0.0.0.0` inside its container and publishes the configured port on
the A6000 host. `logging.prometheus_port` is required for the receiver process. Configure
Prometheus on the A6000 host or on an allowed monitoring network to scrape
`<a6000-host>:9091/metrics`.

```yaml
logging:
  prometheus_port: 9091
```

The receiver is the unified endpoint for A6000 service metrics. The Jetson orchestrator
continues to write turn metrics to `data/logs/turns.jsonl`; it does not bind port `9091`.
Direct service endpoints remain available for debugging and per-service scrape policies.

Example scrape:

```bash
curl -fsS -H "Authorization: Bearer ${KOTONOHA_SERVICE_TOKEN}" \
  http://127.0.0.1:9091/metrics
```

Metrics do not contain source speech, translated text, turn identifiers, or model prompt
content. Labels are limited to bounded service, role, source, route, outcome, mode, stage,
backend, and memory-state values.
