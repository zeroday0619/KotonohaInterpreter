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

Every resident FastAPI model service exposes Prometheus text metrics at its existing
service port on Jetson, A6000, and other supported accelerator profiles:

| Service | Endpoint |
|---|---|
| ASR | `http://127.0.0.1:8001/metrics` |
| Verification ASR | `http://127.0.0.1:8002/metrics` |
| Translation LLM | `http://127.0.0.1:8003/metrics` |
| TTS | `http://127.0.0.1:8004/metrics` |

The endpoint reports HTTP request counts and durations, service readiness, operating
system and kernel identity, accelerator backend and memory architecture, CPU load,
system and accelerator memory, root-filesystem use, configured engine limits, turn stage
durations, output tokens, generation rate, ASR log probability, conditional
cross-verification, failovers, and latency-budget violations.

The `/metrics` endpoint follows the same bearer-token middleware as other service routes.
When `KOTONOHA_SERVICE_TOKEN` is set, configure the Prometheus scrape job with the same
token. The `/health` endpoint remains unauthenticated for deployment probes.

The Web process always runs a unified metrics collector. It polls the active endpoint for
each model role and any distinct onboard fallback endpoint, refreshes service health and
resource gauges every ten seconds, caches successful metrics payloads, and merges them
with Web and turn metrics. Aggregated samples include bounded `role` and `source` labels.
A failed scrape removes the stale payload and sets
`kotonoha_remote_metrics_scrape_up` to `0`.

Use these endpoints on every placement mode:

| Endpoint | Function |
|---|---|
| `<web-host>:8080/metrics` | Unified Prometheus exposition |
| `<web-host>:8080/api/monitoring` | Chart-ready monitoring summary and bounded history |

The Web dashboard samples every five seconds and retains 720 samples in memory. The API
accepts `window_seconds` from 60 through 3600. Restarting the Web process clears this
history. Resident service counters remain available from their service endpoints.

The Web routes do not implement bearer authentication. Bind the Web service to loopback,
or place an authenticated TLS reverse proxy in front of it before exposing the dashboard,
JSON API, or unified metrics endpoint to a network.

The A6000 Compose deployment can additionally start a dedicated receiver on port `9091`
for headless operation without the Web service. `logging.prometheus_port` is required
only for that receiver. It binds to `0.0.0.0` inside its container and publishes the
configured port on the host.

```yaml
logging:
  prometheus_port: 9091
```

The Web endpoint is the primary unified endpoint. The headless receiver remains an
optional A6000 deployment component. Direct service endpoints remain available for
debugging and per-service scrape policies.

Example scrape:

```bash
curl -fsS http://127.0.0.1:8080/metrics
```

Metrics do not contain source speech, translated text, turn identifiers, or model prompt
content. Labels are limited to bounded service, role, source, route, outcome, mode, stage,
backend, and memory-state values.
