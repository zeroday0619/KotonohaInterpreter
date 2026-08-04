# Installation and Deployment

## Scope

This document defines installation, deployment, verification, update, and rollback
procedures for Kotonoha Interpreter. It covers three environments:

| Environment | Purpose | Inference support |
|---|---|---|
| macOS arm64 workstation | Development, linting, unit tests, evaluation | Control-plane tests only |
| Jetson AGX Orin 64GB | Audio frontend, orchestrator, and on-board model services | Required target |
| RTX A6000 server | Optional high-performance model services | Optional target |

The deployment remains a validation deployment until Phase 0 has been executed on the
Jetson. Phase 0 determines the ASR runtime, TTS backend, and translation model profile.
No latency or model-compatibility result in this document replaces target measurements.

## Deployment Topology

### On-board mode

All processes run on the Jetson. Model services share the Jetson host network and POSIX
shared memory. The orchestrator accesses the microphone and speaker through `/dev/snd`.

| Process | Port | Container | Data path |
|---|---:|---|---|
| Primary ASR | 8001 | `asr` | Shared-memory reference |
| Verification ASR | 8002 | `asr-verify` | Shared-memory reference |
| Translation LLM | 8003 | `llm` | Text over HTTP |
| TTS | 8004 | `tts` | Text and PCM over HTTP |
| Orchestrator and TUI | None | `orchestrator` | `/dev/snd`, host IPC, host network |

### High-performance mode

The audio frontend and orchestrator remain on the Jetson. The A6000 runs resident model
services. `hybrid` moves only the LLM. `remote` moves ASR, verification ASR, LLM, and
TTS. Remote ASR traffic uses multipart binary PCM; it does not use shared memory or
base64 encoding.

| Source | Destination | Ports | Required direction |
|---|---|---|---|
| Jetson | A6000 | TCP 8001-8004 | Outbound from Jetson |
| Operator workstation | Jetson | SSH only, when used | Administrative |
| Operator workstation | A6000 | SSH only, when used | Administrative |

Do not expose ports 8001-8004 to the public internet. The default deployment uses plain
HTTP. Bearer tokens protect the Python model services and remote configuration endpoint,
but they do not provide transport encryption. Network isolation or an operator-managed
TLS reverse proxy is required when traffic crosses an untrusted network.

Jetson model services use host networking and bind to `0.0.0.0`. Their default Compose
environment does not enable bearer authentication. Restrict Jetson ports 8001-8004 at the
host firewall so they are not reachable from unapproved network interfaces.

## Release Readiness

The following conditions are current repository facts:

| Item | State |
|---|---|
| macOS unit tests | Implemented |
| Jetson Phase 0 measurements | Not executed |
| Jetson vLLM ASR backend | Implemented; target execution remains pending |
| aarch64 `onnxruntime`, DeepFilterNet, and CTranslate2 validation | Pending target execution |
| Concurrent model residency on the A6000 | Pending target execution |
| Production service supervisor outside Docker Compose | Not provided |
| TLS termination | Not provided |

Deployment acceptance requires the verification checklist in this document and the
Phase 0 procedure in `spikes/README.md`.

## Version and Hardware Requirements

### Common requirements

- Git access to `git@github.com:zeroday0619/KotonohaInterpreter.git`.
- Sufficient local storage for the source tree, container layers, model artifacts, logs,
  and evaluation recordings. A complete minimum has not been measured because the TTS
  and verification repositories are not assigned fixed artifact sizes in the project.
- Access to the model repositories during artifact preparation. Runtime hosts can be
  disconnected after all models and container images are present.
- A stable system clock on both hosts. Turn metrics and cross-host incident analysis
  require comparable timestamps.

### macOS workstation

| Component | Requirement |
|---|---|
| Architecture | Apple silicon, arm64 |
| Python | 3.10 or later; repository development uses 3.12 |
| uv | 0.12.x |
| Audio | Optional for unit tests; required for local device enumeration |

### Jetson host

| Component | Required value |
|---|---|
| Device | NVIDIA Jetson AGX Orin 64GB |
| Architecture | aarch64 |
| JetPack | 6.2 |
| L4T | r36.4.x |
| CUDA | 12.6 |
| Container runtime | Docker with NVIDIA runtime |
| Power mode | MAXN during validation and operation |
| Clocks | Locked with `jetson_clocks` during measurement |

The Compose file pins the `dustynv` images to the `r36.4.0` family. Do not change the
JetPack, CUDA, L4T, or base-image family without a separate compatibility validation.

### A6000 host

| Component | Requirement |
|---|---|
| GPU | NVIDIA RTX A6000 48 GB, sm_86 |
| Host architecture | x86_64 |
| Container runtime | Docker Engine, Compose plugin, NVIDIA Container Toolkit |
| Driver | Must execute the configured CUDA 12.6 container image |
| Network | Stable route from the Jetson to TCP 8001-8004 |

The repository does not specify an exact host driver version. Installation is blocked
until `nvidia-smi` works on the host and inside a GPU-enabled container.

## Source Installation

Use the same Git commit on the workstation, Jetson, and A6000. Different source revisions
can change request schemas and configuration fields independently.

```bash
git clone git@github.com:zeroday0619/KotonohaInterpreter.git
cd KotonohaInterpreter
git rev-parse HEAD
git status --short
```

Record the commit hash in the deployment record. Continue only when `git status --short`
is empty, or when every local file is an intentional host-specific ignored file.

## Quick Deployment

The deployment script performs host validation, verifies required model artifacts,
creates missing host-specific configuration from committed templates, builds images,
starts resident model services, and waits for health checks.

On the Jetson:

```bash
bash scripts/deploy.sh jetson
```

On the A6000:

```bash
bash scripts/deploy.sh a6000
```

The first A6000 run creates a protected `.env` file with a random 32-byte service token
when no file or `KOTONOHA_SERVICE_TOKEN` value exists. Transfer that token to the Jetson
through a secure channel. The script never prints the token.

Available options:

```text
--env-file PATH       Use a non-default A6000 Compose environment file
--health-timeout SEC  Change the model startup timeout from 600 seconds
--no-build            Start already-built images
--skip-power-setup    Do not set Jetson MAXN mode or lock clocks
--remove-images       Remove project-built images during uninstall
```

The script does not replace `config/local.yaml`, `config/remote-server.local.yaml`, or an
existing `.env`. It does not start the interactive orchestrator. Continue with the
runtime command printed after all resident services become healthy.

The script first attempts Docker access as the current user. When the Docker socket
requires root privileges, it automatically uses `sudo docker` for Docker and Compose
operations. Jetson power commands also use `sudo` when the script is not running as root.
Do not run the complete script with `sudo`; host-specific files should remain owned by
the deployment account.

### Quick uninstall

Remove project containers and the Compose network:

```bash
# Jetson
bash scripts/deploy.sh uninstall jetson

# A6000
bash scripts/deploy.sh uninstall a6000
```

Also remove images built by this project:

```bash
bash scripts/deploy.sh uninstall jetson --remove-images
bash scripts/deploy.sh uninstall a6000 --remove-images
```

Uninstall never removes model artifacts, logs, SQLite data, `.env`, local YAML overrides,
or upstream base images. `--remove-images` removes only images named for this project.

## macOS Development Installation

### Install dependencies

Run from the repository root:

```bash
uv sync --frozen
uv run python --version
uv run kotonoha --help
```

`uv sync --frozen` installs the default development dependency group from `uv.lock` and
fails instead of changing the lock file. Do not install the `device` extra on macOS. It
contains target-specific inference packages.

Install COMET and evaluation-only packages only on the development workstation:

```bash
uv sync --frozen --group eval
```

### Verify the workstation

```bash
uv run ruff check .
uv run pytest -q
uv run kotonoha doctor
```

`kotonoha doctor` can report missing model and audio dependencies on macOS. Those entries
do not represent Jetson compatibility. Unit tests must continue to run without models,
microphones, or network access.

`kotonoha doctor` must report `uvloop` as available. The orchestrator, both Textual
applications, CLI network probes, and all Python model services require uvloop. Uvicorn
commands set `--loop uvloop` explicitly, so a missing wheel fails during installation or
startup instead of changing the event-loop implementation silently.

## Model Artifact Preparation

### Download artifacts

Run on a host with model-repository access:

```bash
bash scripts/fetch_models.sh
du -sh models/*
```

The script creates the following layout:

```text
models/
├── Qwen3-ASR-1.7B/
├── Qwen3-TTS-0.6B/
├── faster-whisper-large-v3/
├── gguf/
│   ├── Qwen3-14B-Q4_K_M.gguf
│   └── Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
└── silero_vad.onnx
```

The TTS download is best-effort because MeloTTS remains the fallback. A failed TTS
download must be recorded before deployment; it is not equivalent to a successful Qwen3
TTS installation.

### Verify artifacts

```bash
test -s models/silero_vad.onnx
test -d models/Qwen3-ASR-1.7B
test -d models/faster-whisper-large-v3
test -s models/gguf/Qwen3-14B-Q4_K_M.gguf
test -s models/gguf/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
```

If Qwen3 TTS is selected after Spike 2, also run:

```bash
test -d models/Qwen3-TTS-0.6B
```

`fetch_models.sh` uses Hugging Face local directories rather than a cache-only layout.
Offline deployments must configure model services to use the mounted absolute paths
shown below. A repository ID alone can trigger a cache lookup that does not resolve the
local directory.

### Transfer pre-fetched artifacts

When the runtime host has no internet access, transfer the complete directory from the
artifact-preparation host. Replace the placeholders with operator-controlled addresses.

```bash
rsync -a --info=progress2 models/ <jetson-host>:/opt/kotonoha/models/
rsync -a --info=progress2 models/ <a6000-host>:/opt/kotonoha/models/
```

After transfer, bind the destination to the repository `models` path or set the A6000
`MODELS_DIR` Compose variable to the absolute host path.

## Jetson Installation

### Validate the host

Run before building containers:

```bash
uname -m
head -n 1 /etc/nv_tegra_release
docker version
docker compose version
docker info
ls -la /dev/snd
```

Expected results:

- `uname -m` returns `aarch64`.
- `/etc/nv_tegra_release` identifies L4T r36.4.x.
- Docker reports the NVIDIA runtime.
- `/dev/snd` contains the intended capture and playback devices.

Add the operator account to the audio group if required, then start a new login session:

```bash
sudo usermod -aG audio "$USER"
```

### Set power and clocks

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
sudo nvpmodel -q
jetson_clocks --show
```

Run `jtop` in a second terminal during model loading and measurements. Reject results
collected during thermal throttling or power-mode changes.

### Place source and models

The default Compose mounts expect this layout:

```text
KotonohaInterpreter/
├── config/
├── data/
├── docker/
├── models/
├── scripts/
└── src/
```

Create runtime directories before starting containers:

```bash
mkdir -p data/logs models/gguf
```

### Create the Jetson override

The quick deployment script copies `config/jetson.local.example.yaml` to
`config/local.yaml` when the override does not exist. For manual deployment, create the
file with the following complete host-path overlay. The file is ignored by Git and is
shared with every Jetson container through `/app`.

```yaml
# Jetson-specific paths for an offline deployment.
frontend:
  vad:
    model_path: /app/models/silero_vad.onnx

asr:
  vllm_model_id: /models/Qwen3-ASR-1.7B

asr_verify:
  model_id: /models/faster-whisper-large-v3

llm:
  models_dir: /models/gguf

tts:
  model_id: /models/Qwen3-TTS-0.6B
```

Protect host-specific configuration because it can later contain remote credentials:

```bash
chmod 600 config/local.yaml
```

Do not change `frontend.vad.preroll_ms`, `asr.n_best`, or the other hard constraints
during installation.

### Validate Compose configuration

```bash
docker compose -f docker/compose.yaml config --quiet
docker compose -f docker/compose.yaml config --images
```

Both Compose files set the project name to `kotonohainterpreter`. Locally built images
therefore use the `kotonohainterpreter-<service>:latest` naming pattern regardless of the
`docker/` directory name. The expected ASR image is
`kotonohainterpreter-asr:latest`.

Confirm that these image families remain unchanged in the rendered configuration:

- `ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04`
- `dustynv/faster-whisper:r36.4.0`
- `dustynv/llama_cpp:r36.4.0`
- `dustynv/pytorch:r36.4.0`

### Build Jetson images

```bash
docker compose -f docker/compose.yaml build asr asr-verify tts orchestrator
```

Review the build output for the following conditions:

- PyTorch reports a CUDA build.
- Transformers is 4.57 or later in the ASR image.
- CTranslate2 and faster-whisper import in the verification image.
- `onnxruntime` and DeepFilterNet installation status is explicit.
- At least one TTS backend installs.

The orchestrator and TTS Dockerfiles currently permit selected target dependencies to
fail during image construction. A successful image build does not prove those backends
loaded. The service health checks remain mandatory.

### Start model services

```bash
docker compose -f docker/compose.yaml up -d asr asr-verify llm tts
docker compose -f docker/compose.yaml ps
```

Inspect startup logs before starting the orchestrator:

```bash
docker compose -f docker/compose.yaml logs --tail=200 asr
docker compose -f docker/compose.yaml logs --tail=200 asr-verify
docker compose -f docker/compose.yaml logs --tail=200 llm
docker compose -f docker/compose.yaml logs --tail=200 tts
```

### Verify service health

```bash
curl -fsS http://127.0.0.1:8001/health | python3 -m json.tool
curl -fsS http://127.0.0.1:8002/health | python3 -m json.tool
curl -fsS http://127.0.0.1:8003/health | python3 -m json.tool
curl -fsS http://127.0.0.1:8004/health | python3 -m json.tool
```

HTTP 200 alone is insufficient for the Python services. Their JSON response must contain
`"ok": true`. A response with `"ok": false` means the web process started but the model
failed to load.

### Configure audio and the interface

List ALSA and PortAudio devices from the same container context used at runtime:

```bash
docker compose -f docker/compose.yaml run --rm orchestrator \
  python3 -m kotonoha._cli devices
```

Open the configuration TUI:

```bash
docker compose -f docker/compose.yaml run --rm orchestrator \
  python3 -m kotonoha._cli config
```

Set `audio.input_device` and `audio.output_device` to a stable device name when possible.
Numeric PortAudio indexes can change when USB devices are reconnected. The configuration
TUI writes `config/local.yaml` only after validating the complete configuration.

Import the baseline glossary:

```bash
docker compose -f docker/compose.yaml run --rm orchestrator \
  python3 -m kotonoha._cli glossary import config/glossary.seed.yaml
```

### Run diagnostics

```bash
docker compose -f docker/compose.yaml run --rm orchestrator \
  python3 -m kotonoha._cli doctor
```

Resolve every required-module failure, missing VAD artifact, down service, and CUDA
failure before an operator session.

### Start the interpreter

```bash
docker compose -f docker/compose.yaml run --rm orchestrator
```

The initial session uses push-to-talk. Press `space` to start and stop an utterance. The
orchestrator container is intentionally interactive and is not started as a detached
background service.

## A6000 Installation

### Validate GPU container access

Run on the A6000 host:

```bash
uname -m
nvidia-smi
docker version
docker compose version
docker info
```

After configuring the container runtime, validate GPU allocation with the NVIDIA sample
workload below. Record the resolved diagnostic image digest in controlled deployment
environments.

### Configure the NVIDIA Docker runtime

`nvidia-smi` on the host does not prove that Docker can allocate the GPU. If Docker
reports `could not select device driver "nvidia" with capabilities: [[gpu]]`, install and
configure NVIDIA Container Toolkit before running the deployment script.

For Ubuntu or Debian, use the NVIDIA production repository:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify runtime registration and GPU allocation:

```bash
sudo docker info --format '{{json .Runtimes}}'
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

The first command must contain `nvidia`. The second command must display the A6000. The
deployment script performs the runtime-registration check before changing Compose state.
The installation and sample commands follow the
[NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.18.0/install-guide.html)
and [sample workload procedure](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.18.0/sample-workload.html).

### Use the same source revision

```bash
git clone git@github.com:zeroday0619/KotonohaInterpreter.git
cd KotonohaInterpreter
git rev-parse HEAD
```

The hash must match the Jetson deployment.

### Place model artifacts

The default remote Compose file mounts the repository `models` directory at `/models`.
Use the same artifact layout documented in Model Artifact Preparation.

```bash
test -d models/Qwen3-ASR-1.7B
test -d models/faster-whisper-large-v3
test -s models/gguf/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
```

### Create the remote service override

The quick deployment script copies `config/remote-server.local.example.yaml` when the
override does not exist. For manual deployment, create `config/remote-server.local.yaml`:

```yaml
# A6000-specific paths for an offline deployment.
asr:
  vllm_model_id: /models/Qwen3-ASR-1.7B

asr_verify:
  model_id: /models/faster-whisper-large-v3

llm:
  models_dir: /models/gguf

tts:
  model_id: /models/Qwen3-TTS-0.6B
```

```bash
chmod 600 config/remote-server.local.yaml
```

### Define remote deployment variables

Create a repository-root `.env` file. Replace the token placeholder with the output of
`openssl rand -hex 32`. Do not commit the file.

```dotenv
KOTONOHA_SERVICE_TOKEN=<64-hex-character-random-token>
MODELS_DIR=../models
REMOTE_BASE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
REMOTE_ASR_BASE=vllm/vllm-openai:v0.19.1
REMOTE_TTS_BUILD_BASE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
LLM_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
LLM_PROFILE=moe
LLM_CTX=4096
TRANSFORMERS_OFFLINE=1
```

```bash
chmod 600 .env
```

The Python services receive `KOTONOHA_SERVICE_TOKEN`. The llama.cpp image does not use
the project authentication middleware. Restrict port 8003 to the Jetson at the host
firewall unless an authenticated reverse proxy protects it.

The TTS build uses the CUDA devel image to compile FlashAttention 2 against the same
PyTorch and CUDA ABI as the runtime image. The compiler toolchain is not copied into the
final TTS image. The build fails if FlashAttention, Qwen3-TTS, SoX, or another required
runtime dependency cannot be imported. The first TTS build can take several minutes.

The remote TTS service does not load MeloTTS. A remote Qwen3-TTS startup or request
failure is retried by the orchestrator against the resident Jetson TTS service, whose
default backend is MeloTTS. This separation avoids loading Qwen3-TTS and MeloTTS into one
Python environment with incompatible Transformers requirements.

### Validate and build the remote stack

Load the token into the Compose process, then validate the rendered configuration:

```bash
set -a
source .env
set +a
docker compose -f docker/compose.remote.yaml config --quiet
docker compose -f docker/compose.remote.yaml config --images
docker compose -f docker/compose.remote.yaml build asr asr-verify tts
docker compose -f docker/compose.remote.yaml pull llm
```

The build must produce three role-specific images:

| Service | Image | Dockerfile target |
|---|---|---|
| Primary ASR | `kotonohainterpreter-asr:latest` | `asr` |
| Verification ASR | `kotonohainterpreter-asr-verify:latest` | `asr-verify` |
| TTS | `kotonohainterpreter-tts:latest` | `tts` |

The targets share a cached application layer but install and verify role-specific runtime
dependencies. The common layer imports `pydantic_settings` during the build. A missing
core dependency therefore fails the image build instead of entering a restart loop.

The llama.cpp image stores `llama-server` under `/app`. Its Compose service mounts only
the launcher, configuration directory, and models. Do not restore the repository-wide
`/app` bind mount on this service because it hides the image binary and causes exit code
127. Flash Attention remains best-effort in the TTS target; its failure must appear in
the build log and TTS health must report the backend that loaded.

### Start and verify the remote stack

```bash
docker compose -f docker/compose.remote.yaml up -d asr asr-verify llm tts
docker compose -f docker/compose.remote.yaml ps
docker compose -f docker/compose.remote.yaml logs --tail=200
```

From the A6000 host:

```bash
curl -fsS http://127.0.0.1:8001/health | python3 -m json.tool
curl -fsS http://127.0.0.1:8002/health | python3 -m json.tool
curl -fsS http://127.0.0.1:8003/health | python3 -m json.tool
curl -fsS http://127.0.0.1:8004/health | python3 -m json.tool
```

Confirm concurrent residency. A service that loads successfully in isolation can still
fail when all models occupy the GPU. Check `nvidia-smi` after all four services report
healthy and retain the output in the deployment record.

## Connect the Jetson to the A6000

### Configure service addresses

Copy `config/performance.yaml` to a host-specific overlay or edit the corresponding local
fields through `kotonoha config`. Replace `a6000.lan` with a resolvable hostname or fixed
address. Do not commit the bearer token.

The complete performance overlay is `config/performance.yaml`. It selects remote mode,
defines all four service URLs, and applies A6000-oriented model policy. Local
`config/local.yaml` continues to override host-specific audio and model paths.

### Pass the bearer token

The Jetson Compose file does not automatically forward arbitrary host environment
variables into the orchestrator. Pass the token explicitly for each management,
diagnostic, and runtime command:

```bash
export KOTONOHA__REMOTE__TOKEN=<same-token-as-a6000>
```

### Measure the network path

```bash
docker compose -f docker/compose.yaml run --rm \
  -e KOTONOHA__REMOTE__TOKEN="$KOTONOHA__REMOTE__TOKEN" \
  orchestrator python3 -m kotonoha._cli \
  -c config/performance.yaml netcheck
```

`netcheck` measures health-check RTT and binary upload time. Use `hybrid` when remote
audio transfer is prohibited or when measured network overhead consumes more than 25%
of the post-silence latency budget.

### Configure the remote server from the Jetson

```bash
docker compose -f docker/compose.yaml run --rm \
  -e KOTONOHA__REMOTE__TOKEN="$KOTONOHA__REMOTE__TOKEN" \
  orchestrator python3 -m kotonoha._cli \
  -c config/performance.yaml config
```

Select `Remote A6000` in the target selector. The remote target exposes only settings
owned by resident ASR, verification ASR, LLM, and TTS processes. Saving writes
`config/remote-server.local.yaml` through the authenticated ASR management endpoint.
LLM startup values are also written to `config/remote-llm.env`.

Remote model settings do not reload in the management request. Restart affected services
after saving:

| Changed section | Required A6000 restart |
|---|---|
| Primary ASR | `docker compose -f docker/compose.remote.yaml restart asr` |
| Verification ASR | `docker compose -f docker/compose.remote.yaml restart asr-verify` |
| Translation LLM | `docker compose -f docker/compose.remote.yaml restart llm` |
| TTS | `docker compose -f docker/compose.remote.yaml restart tts` |

### Start high-performance mode

```bash
docker compose -f docker/compose.yaml run --rm \
  -e KOTONOHA__REMOTE__TOKEN="$KOTONOHA__REMOTE__TOKEN" \
  orchestrator python3 -m kotonoha._cli \
  -c config/performance.yaml run
```

The TUI status bar must show the expected placement. In `remote` mode it must also show
that utterance audio leaves the Jetson. Review `placement` and `failovers` in
`data/logs/turns.jsonl` after the first test turn.

## Phase 0 Execution

Do not approve the Jetson deployment for interpreting sessions before running Phase 0.
The commands, acceptance thresholds, and report generation procedure are defined in
`spikes/README.md`.

Minimum preparation:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
mkdir -p spikes/out
```

After running the three spikes:

```bash
python3 spikes/report.py --dir spikes/out \
  --md spikes/out/PHASE0.md --patch spikes/out/local.yaml
```

Review the generated report before copying any decision patch into `config/local.yaml`.
Phase 1 starts only after explicit approval of all three verdicts.

## Operational Procedures

### Status

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

### Stop services

Stop containers without deleting local configuration, model files, or logs:

```bash
docker compose -f docker/compose.yaml stop
docker compose -f docker/compose.remote.yaml stop
```

### Uninstall service containers

Use `scripts/deploy.sh uninstall` when the deployment must be removed from a host. The
command performs `docker compose down --remove-orphans` and remains usable when GPU or
model preflight checks fail. Add `--remove-images` only when locally built project images
must also be deleted.

### Start existing services

```bash
docker compose -f docker/compose.yaml start asr asr-verify llm tts
docker compose -f docker/compose.remote.yaml start asr asr-verify llm tts
```

### Restart after configuration changes

Local `config/local.yaml` changes apply when the orchestrator or affected model service
starts again. Restart only the affected resident model service where possible. Frontend,
audio, routing, storage, and UI changes apply on the next orchestrator invocation.

### Back up mutable state

Back up these files before source updates or configuration migrations:

| Host | Path | Content |
|---|---|---|
| Jetson | `config/local.yaml` | Device, audio, and placement overrides |
| Jetson | `data/kotonoha.db` | Glossary and turn history |
| Jetson | `data/logs/` | Application and turn metrics |
| A6000 | `config/remote-server.local.yaml` | Remote model-service overrides |
| A6000 | `config/remote-llm.env` | Generated llama.cpp startup values |
| Both | `.env` | Deployment variables and secret token, when present |

Use an access-controlled backup location. Do not commit these files.

### Update source

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

### Roll back

1. Stop affected containers.
2. Restore the previously recorded source commit on both hosts.
3. Restore host-specific configuration and SQLite data from backup when their formats
   changed.
4. Rebuild the affected images from the previous commit.
5. Start services and repeat health checks.
6. Record the failed commit, service logs, health responses, and rollback time.

The repository does not provide an automated database migration or rollback command.
SQLite backup is therefore mandatory before changes that affect storage models.

## Security Controls

### Required controls

- Restrict A6000 ports 8001-8004 to the Jetson and administrative network.
- Restrict Jetson ports 8001-8004 to local or explicitly approved traffic.
- Generate `KOTONOHA_SERVICE_TOKEN` from a cryptographically secure random source.
- Store `.env`, `config/local.yaml`, and `config/remote-server.local.yaml` with mode 600.
- Keep secrets out of Git, logs, screenshots, and support bundles.
- Block outbound network access after container images and model artifacts are staged if
  full offline operation is required.
- Use `hybrid` mode when utterance audio must remain on the Jetson.
- Add operator-managed TLS before routing service traffic across an untrusted network.

### Authentication boundaries

| Endpoint | Authentication behavior |
|---|---|
| Jetson Python services | Authentication disabled by the default Compose environment |
| Python service `/health` | Open for health monitoring |
| Python service inference endpoints | Bearer token when `KOTONOHA_SERVICE_TOKEN` is set |
| ASR `/admin/config` | Bearer token when `KOTONOHA_SERVICE_TOKEN` is set |
| llama.cpp port 8003 | Not protected by the project FastAPI middleware |

An `auth.disabled` startup warning means a Python service is accepting unauthenticated
requests. Treat this as a deployment failure on the A6000.

## Troubleshooting

| Symptom | Inspection | Corrective action |
|---|---|---|
| Service returns `ok: false` | Service log and `error` field | Correct model path, dependency, CUDA, or memory failure; restart the service |
| Python services restart with missing `pydantic_settings` | Inspect x86_64 markers in `uv.lock` and the common image build check | Regenerate the lock with Linux x86_64 support and rebuild all three Python images |
| LLM restarts with exit code 127 | Inspect LLM mounts and `/app/llama-server` | Remove any bind mount targeting `/app`, then recreate the LLM container |
| LLM cannot load `libllama-server-impl.so` | Inspect `LD_LIBRARY_PATH` and run `ldd /app/llama-server` in the image | Set `/app` as the first library path and recreate the LLM container |
| ASR cannot find the model offline | Inspect `asr.vllm_model_id` | Set `/models/Qwen3-ASR-1.7B` in the host override |
| Verification downloads `large-v3` | Inspect `asr_verify.model_id` | Set `/models/faster-whisper-large-v3` |
| LLM reports `GGUF missing` | Inspect `/models/gguf` and `remote-llm.env` | Correct `MODELS_DIR`, profile, or profile file name; restart `llm` |
| TTS image cannot build FlashAttention | Inspect the devel image tag, CUDA version, memory, and build log | Restore matching build and runtime images; use the SDPA fallback only when the target service loads and Spike 2 records the result |
| TTS reports `sox: not found` | Run `sox --version` in the TTS container | Rebuild the TTS image; the current image installs `sox` and `libsox-fmt-all` |
| Remote TTS reports Qwen failure | Inspect TTS health and the orchestrator `failovers` metric | Correct the remote Qwen service; the current turn retries against the Jetson MeloTTS service before the first audio chunk |
| CUDA is absent in a container | Inspect image build output and `torch.version.cuda` | Restore the pinned CUDA base image; do not install a CPU PyTorch wheel |
| Docker cannot select the `nvidia` device driver | Inspect `docker info --format '{{json .Runtimes}}'` | Install NVIDIA Container Toolkit, configure the Docker runtime with `nvidia-ctk`, and restart Docker |
| Shared-memory errors | Inspect `ipc: host` and `/dev/shm` | Restore host IPC for Jetson services; do not use this path across hosts |
| No capture device | Run `devices`, inspect `/dev/snd`, check group membership | Set the correct device and restart the orchestrator login session |
| Remote request returns 401 | Compare A6000 and Jetson token values | Correct the token without logging it; restart affected clients or services |
| Remote config returns 422 | Read the response detail | Edit only server-owned paths exposed by the remote TUI |
| Remote role repeatedly fails over | Inspect `netcheck`, service logs, and turn `failovers` | Correct service health or network path; use `hybrid` or `onboard` until stable |
| A6000 OOM during startup | Inspect `nvidia-smi` and all service logs | Record the failure and revise measured placement; do not assume isolated load success |
| Latency exceeds 2.9 s | Inspect five-point turn timestamps | Identify the stage overrun; do not attribute it to a model without measurement |
| Thermal throttling | Inspect `jtop` during the same interval | Correct cooling and rerun the complete measurement |

## Deployment Acceptance Checklist

### Source and configuration

- [ ] Jetson and A6000 use the same approved Git commit.
- [ ] Working trees contain no unexplained changes.
- [ ] Host-specific files are ignored by Git and protected with mode 600.
- [ ] `docker compose config --quiet` succeeds on each host.
- [ ] No unvalidated JetPack, CUDA, L4T, or base-image change is present.

### Models and services

- [ ] Required model artifacts exist at the configured absolute paths.
- [ ] All selected backends loaded; Python health responses contain `"ok": true`.
- [ ] The llama.cpp health endpoint responds and the expected GGUF is loaded.
- [ ] CUDA is active in every GPU inference container.
- [ ] Concurrent A6000 residency is confirmed with all services running.

### Jetson runtime

- [ ] MAXN and locked clocks are active.
- [ ] `jtop` shows no throttling during validation.
- [ ] Input and output audio devices work from the orchestrator container.
- [ ] VAD uses the Silero ONNX model on the target, not the workstation energy fallback.
- [ ] Half-duplex microphone gating is observed during playback.

### High-performance mode

- [ ] A6000 ports are restricted to approved source hosts.
- [ ] Python services reject inference requests without the bearer token.
- [ ] `netcheck` completes from the Jetson.
- [ ] TUI placement matches the intended `onboard`, `hybrid`, or `remote` mode.
- [ ] `placement` and `failovers` appear in turn records.
- [ ] Audio egress in `remote` mode is approved by the deployment owner.

### Measurement and handoff

- [ ] Phase 0 reports all three verdicts from target execution.
- [ ] No unmeasured inference or latency value is reported as verified.
- [ ] One WAV replay completes and writes a turn record.
- [ ] One microphone utterance completes through playback.
- [ ] Backup and rollback locations are recorded.
- [ ] Operators have the status, log, restart, and escalation procedures.
