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

## Prometheus Metrics

Every resident FastAPI service exposes Prometheus text metrics at its existing service
port:

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

The orchestrator starts a localhost metrics receiver when `logging.prometheus_port` is
set to a valid port. It binds to `127.0.0.1` and uses the same `/metrics` exposition
format. In hybrid or remote placement, it polls the active service endpoint for every
role, caches the latest successful payload, and exposes the service families through the
same endpoint. Aggregated samples include bounded `role` and `source` labels, where
`source` is `local` or `remote`. A failed scrape removes the stale payload and sets
`kotonoha_remote_metrics_scrape_up` to `0`.

An empty `logging.prometheus_port` value disables the receiver. Configure a host-local
Prometheus instance to scrape the configured port, for example `127.0.0.1:9091`.

```yaml
logging:
  prometheus_port: 9091
```

The receiver is the unified endpoint for the orchestrator turn metrics and active local
or remote service metrics. Direct service endpoints remain available for debugging and
per-service scrape policies.

Example scrape:

```bash
curl -fsS -H "Authorization: Bearer ${KOTONOHA_SERVICE_TOKEN}" \
  http://127.0.0.1:8003/metrics
```

Metrics do not contain source speech, translated text, turn identifiers, or model prompt
content. Labels are limited to bounded service, role, source, route, outcome, mode, stage,
backend, and memory-state values.
