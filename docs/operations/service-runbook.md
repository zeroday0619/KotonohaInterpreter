# Service Runbook

## Status

Jetson services:

```bash
docker compose -f docker/compose.yaml ps
docker compose -f docker/compose.yaml logs --tail=200
```

A6000 services:

```bash
docker compose -f docker/compose.remote.yaml ps
docker compose -f docker/compose.remote.yaml logs --tail=200
nvidia-smi
```

## Stop Services

Stop containers without deleting local configuration, model files, or logs:

```bash
docker compose -f docker/compose.yaml stop
docker compose -f docker/compose.remote.yaml stop
```

## Uninstall Service Containers

Use `scripts/manage.sh uninstall` when the deployment must be removed from a host. The
command performs `docker compose down --remove-orphans` and remains usable when GPU or
model preflight checks fail. Add `--remove-images` only when locally built project images
must also be deleted.

## Start Existing Services

```bash
docker compose -f docker/compose.yaml start asr asr-verify llm tts
docker compose -f docker/compose.remote.yaml start asr asr-verify llm tts
```

## Restart After Configuration Changes

Local `config/local.yaml` changes apply when the orchestrator or affected model service
starts again. Restart only the affected resident model service where possible. Frontend,
audio, routing, storage, and UI changes apply on the next orchestrator invocation.

## Reallocate A6000 GPUs

Routine deployment reuses `config/remote-gpu.env` to preserve stable GPU UUID assignments.
Recalculate placement after adding, removing, or repurposing an A6000:

```bash
bash scripts/manage.sh deploy a6000 --reallocate-gpus
```

The command stops all four remote services before reading free memory. It allocates the
largest reservations first, writes the replacement mapping atomically, recreates the
services, and waits for health checks. Retain the generated mapping and `nvidia-smi`
output in the deployment record.

## Back Up Mutable State

Back up these files before source updates or configuration migrations:

| Host | Path | Content |
|---|---|---|
| Jetson | `config/local.yaml` | Device, audio, and placement overrides |
| Jetson | `data/kotonoha.db` | Glossary and turn history |
| Jetson | `data/logs/` | Application and turn metrics |
| A6000 | `config/remote-server.local.yaml` | Remote model-service overrides |
| A6000 | `config/remote-llm.env` | Generated vLLM translation startup values |
| A6000 | `config/remote-gpu.env` | Generated stable GPU UUID assignments |
| Both | `.env` | Deployment variables and secret token, when present |

Use an access-controlled backup location. Do not commit these files.

## Update Source

1. Stop the interactive orchestrator.
2. Record the current commit with `git rev-parse HEAD`.
3. Back up mutable state.
4. Fetch and check out the approved commit on both hosts.
5. Confirm both hosts report the same commit.
6. Rebuild changed images.
7. Start model services and run all health checks.
8. Run `doctor`, `netcheck` when applicable, and one WAV replay.
9. Start an operator session only after the checks pass.

Do not run `uv lock` or upgrade dependency versions on a deployment host.

## Roll Back

1. Stop affected containers.
2. Restore the previously recorded source commit on both hosts.
3. Restore host-specific configuration and SQLite data from backup when their formats
   changed.
4. Rebuild the affected images from the previous commit.
5. Start services and repeat health checks.
6. Record the failed commit, service logs, health responses, and rollback time.

The repository does not provide an automated database migration or rollback command.
SQLite backup is mandatory before changes that affect storage models.

## Security Controls

- Restrict A6000 ports 8001-8004 to the Jetson and administrative network.
- Restrict Jetson ports 8001-8004 to local or explicitly approved traffic.
- Generate `KOTONOHA_SERVICE_TOKEN` from a cryptographically secure random source.
- Store `.env`, `config/local.yaml`, and `config/remote-server.local.yaml` with mode 600.
- Keep secrets out of Git, logs, screenshots, and support bundles.
- Block outbound network access after container images and model artifacts are staged if
  full offline operation is required.
- Use `hybrid` mode when utterance audio must remain on the Jetson.
- Add operator-managed TLS before routing service traffic across an untrusted network.

### Authentication Boundaries

| Endpoint | Authentication behavior |
|---|---|
| Jetson Python services | Authentication disabled by the default Compose environment |
| Python service `/health` | Open for health monitoring |
| Python service inference endpoints | Bearer token when `KOTONOHA_SERVICE_TOKEN` is set |
| ASR `/admin/config` | Bearer token when `KOTONOHA_SERVICE_TOKEN` is set |
| vLLM port 8003 | Bearer token enforced through the vLLM `--api-key` option |
| vLLM-Omni port 8004 | Bearer token enforced through the vLLM `--api-key` option |

An `auth.disabled` startup warning means a Python service is accepting unauthenticated
requests. Treat this as a deployment failure on the A6000.

## Troubleshooting

| Symptom | Inspection | Corrective action |
|---|---|---|
| Service returns `ok: false` | Service log and `error` field | Correct model path, dependency, CUDA, or memory failure; restart the service |
| Python services restart with missing `pydantic_settings` | Inspect x86_64 markers in `uv.lock` and the common image build check | Regenerate the lock with Linux x86_64 support and rebuild the ASR Python images |
| LLM reports that `vllm` is unavailable | Inspect the selected LLM image | Restore the pinned vLLM image and recreate the container |
| Jetson ASR cannot find the model offline | Inspect `asr.vllm_model_id` | Set `/models/Qwen3-ASR-0.6B` in the Jetson override |
| A6000 ASR cannot find the model offline | Inspect `asr.vllm_model_id` | Set `/models/Voxtral-Mini-4B-Realtime-2602` in the remote override |
| Realtime ASR WebSocket fails | Inspect `/v1/realtime`, service logs, and the configured realtime architecture | Restore the target model and architecture pair; rerun Spike 1 |
| Verification downloads `large-v3` | Inspect `asr_verify.model_id` | Set `/models/faster-whisper-large-v3` |
| LLM reports an incomplete model snapshot | Inspect `/models/llm` and `remote-llm.env` | Correct `LLM_MODELS_DIR`, profile, or model directory; restart `llm` |
| vLLM-Omni TTS does not start | Inspect the selected image, CUDA initialization, model path, and Spike 2 log | Restore the pinned Omni image or correct the offline snapshot; do not infer compatibility from the manifest |
| Speech API returns an application error | Inspect the full response and TTS service log | Correct the model, voice, or language request; HTTP 4xx responses do not activate failover |
| Remote TTS transport fails before audio | Inspect TTS health and orchestrator `failovers` | Correct the remote endpoint; the turn retries the resident Jetson vLLM-Omni service before the first PCM chunk |
| CUDA is absent in a container | Inspect image output and `torch.version.cuda` | Restore the pinned CUDA base image; do not install a CPU PyTorch wheel |
| Docker cannot select the `nvidia` driver | Inspect `docker info --format '{{json .Runtimes}}'` | Install NVIDIA Container Toolkit, configure `nvidia-ctk`, and restart Docker |
| GPU allocation reports insufficient capacity | Inspect `.env` reservations, `config/remote-gpu.env`, and `nvidia-smi` | Stop competing workloads, correct reservations, then run `bash scripts/manage.sh deploy a6000 --reallocate-gpus` |
| Service starts on the wrong GPU | Source `.env` and `config/remote-gpu.env`, then inspect rendered `device_ids` | Restore the generated UUID mapping and recreate the affected service |
| Shared-memory errors | Inspect `ipc: host` and `/dev/shm` | Restore host IPC for Jetson services; do not use this path across hosts |
| No capture device | Run `devices`, inspect `/dev/snd`, and check group membership | Set the device and restart the orchestrator login session |
| Remote request returns 401 | Compare A6000 and Jetson token values | Correct the token without logging it; restart affected clients or services |
| Remote config returns 422 | Read the response detail | Edit only server-owned paths exposed by the remote TUI |
| Remote role repeatedly fails over | Inspect `netcheck`, service logs, and turn `failovers` | Correct health or networking; use `hybrid` or `onboard` until stable |
| A6000 OOM during startup | Inspect `nvidia-smi` and all service logs | Record the failure and revise measured placement |
| Latency exceeds 2.9 s | Inspect five-point turn timestamps | Identify the stage overrun before assigning a cause |
| Thermal throttling | Inspect `jtop` during the same interval | Correct cooling and rerun the complete measurement |
