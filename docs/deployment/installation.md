# Installation and Deployment

## Scope

Kotonoha supports complete Web and model-service deployments on Jetson AGX Orin and RTX
A6000 hosts. Each host runs the Web process, ASR, verification ASR, translation, and TTS.
Hardware profiles replace runtime placement modes.

| Target | Hardware profile | Compose files |
|---|---|---|
| Jetson AGX Orin | `nvidia.jetson.agx-orin` | `docker/compose.yaml` |
| RTX A6000 | `nvidia.rtx.a6000` | `docker/compose.yaml`, `docker/compose.a6000.yaml`, `docker/compose.web.a6000.yaml` |

The Web interface is the supported operator interface. Ports 8001-8004 host resident
model services. Port 8080 hosts the Web application. Port 9091 provides the optional
headless metrics receiver.

## Requirements

### Common

- Git
- Docker Engine with Compose v2
- NVIDIA Container Toolkit
- At least 300 GiB free model and image storage
- A repository checkout and staged offline model snapshots
- A 32-byte or longer random service token for network-accessible deployments

### Jetson

| Component | Required value |
|---|---|
| Device | NVIDIA Jetson AGX Orin 64GB |
| OS | Ubuntu 24.04, Jetson Linux 39.2 |
| SDK | JetPack 7.2 |
| CUDA reported by host | 13.2.x |
| Architecture | aarch64 |

Target execution must validate the selected vLLM image on `sm_87`. Image metadata alone
does not establish kernel compatibility.

### RTX A6000

| Component | Required value |
|---|---|
| GPU | NVIDIA RTX A6000, 48 GiB |
| Architecture | x86_64 |
| Driver | Compatible with the selected NGC vLLM image |
| GPU access | `docker run --gpus all` succeeds |

## Environment Preparation

Create the deployment environment file:

```bash
cp .env.example .env
chmod 600 .env
python3 -c 'import secrets; print(secrets.token_hex(32))'
```

Write the generated token to `KOTONOHA_SERVICE_TOKEN` in `.env`. Set `MODELS_DIR` to the
host model directory. Keep `TRANSFORMERS_OFFLINE=0` only while staging artifacts; set it
to `1` before offline acceptance.

The deployment creates `data/kotonoha.db`. The database contains configuration,
credentials stored as settings, glossary entries, and interpretation history. Apply mode
`0600`, include the database in backups, and never commit it.

## Managed Deployment

The management script detects the host, requests confirmation, obtains Docker privileges
through `sudo` when required, and delegates to the target deployment workflow.

```bash
bash scripts/manage.sh detect
bash scripts/manage.sh setup
bash scripts/manage.sh models fetch
bash scripts/manage.sh models verify
bash scripts/manage.sh benchmark
bash scripts/manage.sh deploy
bash scripts/manage.sh web
```

Pass an explicit target when detection is unavailable:

```bash
bash scripts/manage.sh setup jetson
bash scripts/manage.sh deploy jetson
bash scripts/manage.sh web jetson
```

```bash
bash scripts/manage.sh setup a6000
bash scripts/manage.sh deploy a6000
bash scripts/manage.sh web a6000
```

Use `-y` for non-interactive execution. Use `--dry-run` to inspect delegated commands.

```bash
bash scripts/manage.sh -y --dry-run deploy a6000
```

## Model Artifacts

Fetch and verify all snapshots before enabling offline mode:

```bash
bash scripts/manage.sh models fetch
bash scripts/manage.sh models verify
```

Required snapshot names include:

| Role | Jetson | A6000 |
|---|---|---|
| ASR | `Qwen3-ASR-0.6B`, `Qwen3-ASR-0.6B-hf` | `Voxtral-Mini-4B-Realtime-2602` |
| Verification ASR | `faster-whisper-large-v3` | `faster-whisper-large-v3` |
| Translation | `translategemma-4b-it` | `translategemma-12b-it` |
| TTS | `Qwen3-TTS-0.6B` | `Qwen3-TTS-0.6B` |

TranslateGemma requires license acceptance before snapshot download. Artifact verification
checks file presence and layout; it does not validate model execution.

## Jetson Deployment

Validate the host before building images:

```bash
uname -m
cat /etc/nv_tegra_release
nvidia-smi
docker info --format '{{json .Runtimes}}'
```

Set maximum performance immediately before measurement:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
jtop
```

Build and start the stack:

```bash
bash scripts/manage.sh deploy jetson
bash scripts/manage.sh web jetson
```

The deployment selects `nvidia.jetson.agx-orin` in SQLite before running model checks.
Confirm the selected profile in the Web configuration page before accepting traffic.

Verify services:

```bash
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8002/health
curl -fsS http://127.0.0.1:8003/health
curl -fsS http://127.0.0.1:8004/health
curl -fsS http://127.0.0.1:8080/health
```

Keep `jtop` active during the first complete turn. Reject the deployment if any service
restarts, the unified-memory budget is exceeded, or thermal throttling occurs.

## A6000 Deployment

Validate container GPU access:

```bash
nvidia-smi
docker info --format '{{json .Runtimes}}'
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

Allocate GPUs before the first deployment and after hardware changes:

```bash
bash scripts/manage.sh gpu allocate
bash scripts/manage.sh deploy a6000 --reallocate-gpus
```

The allocator writes stable GPU UUID assignments to `config/a6000-gpu.env`. It reserves
memory for all four services and rejects an assignment that exceeds configured capacity.
Retain the generated file across routine restarts.

Build and start the complete stack:

```bash
bash scripts/manage.sh deploy a6000
bash scripts/manage.sh web a6000
```

The deployment selects `nvidia.rtx.a6000` in SQLite before running model checks. Confirm
the selected profile and effective model paths in the Web configuration page.

Verify service residency:

```bash
docker compose -f docker/compose.a6000.yaml ps
nvidia-smi
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8002/health
curl -fsS http://127.0.0.1:8003/health
curl -fsS http://127.0.0.1:8004/health
curl -fsS http://127.0.0.1:8080/health
```

All four services must remain resident concurrently. An isolated model benchmark does not
prove deployment capacity.

## Configuration

Open the Web configuration page at `http://127.0.0.1:8080`. The page displays the active
hardware profile, effective values, profile defaults, and modified fields.

- Save validates the complete configuration and writes operator overrides to SQLite.
- Reset restores one field to the active profile default.
- Reset all removes every operator override and preserves the selected hardware profile.
- Model setting changes request an authenticated service reload.

Environment values have higher precedence than SQLite. Remove an environment override
before expecting a Web reset to change its effective value.

Existing `--config`, `KOTONOHA_CONFIG`, `KOTONOHA_LOCAL_CONFIG`, and
`config/local.yaml` inputs are imported once. Confirm imported values, then remove the
legacy YAML files.

## Web Exposure

Browsers allow microphone capture on localhost or a secure origin. To serve other hosts,
place an authenticated HTTPS reverse proxy in front of port 8080. Do not publish ports
8001-8004 directly to untrusted networks.

Kotonoha does not provide Web-user authentication or TLS termination. The reverse proxy
must provide both controls before network exposure.

## Validation

Run diagnostics after deployment:

```bash
bash scripts/manage.sh doctor
bash scripts/manage.sh benchmark link --samples 20 --seconds 6
curl -fsS http://127.0.0.1:8080/metrics
```

Run the target hardware spikes before performance acceptance:

```bash
bash scripts/manage.sh benchmark jetson
bash scripts/manage.sh benchmark a6000
```

Performance results require target measurements. Workstation tests and Compose rendering
do not establish ASR latency, TTS intelligibility, CUDA compatibility, or memory capacity.

## Backup and Upgrade

Back up the following paths before source or schema changes:

- `.env`
- `data/kotonoha.db`
- `data/logs/`
- `config/a6000-gpu.env` on A6000 hosts
- Hardware spike output under `spikes/out/`

Stop the stack, check out the approved revision, rebuild images, and repeat health checks.
Do not run `uv lock` or dependency upgrades on a deployment host.

## Uninstall

Remove containers while preserving configuration, models, logs, and SQLite data:

```bash
bash scripts/manage.sh uninstall jetson
bash scripts/manage.sh uninstall a6000
```

Add `--remove-images` to delete only locally built `kotonohainterpreter-*` images. The
uninstall workflow does not remove upstream NVIDIA images, model artifacts, `.env`, or
`data/kotonoha.db`.

## Acceptance Checklist

- [ ] The source revision is recorded.
- [ ] `.env` and `data/kotonoha.db` have owner-only permissions.
- [ ] The selected hardware profile matches the deployment host.
- [ ] Every required model snapshot passes artifact verification.
- [ ] All five containers remain healthy during a complete turn.
- [ ] ASR returns N-best five and the target Spike 1 passes.
- [ ] TTS produces intelligible speech in all four languages.
- [ ] The translation service sustains at least 5 tokens per second under the specified benchmark.
- [ ] Unified metrics are available at port 8080.
- [ ] The Web interface is reachable only through localhost or authenticated HTTPS.
- [ ] Reset one and reset all restore active profile defaults without changing the profile.
