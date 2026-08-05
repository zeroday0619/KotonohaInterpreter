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

Deployment acceptance requires the verification checklist in this document and
[Performance Measurement](../performance/measurement.md).

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
| JetPack | 7.2 |
| Jetson Linux | 39.2 |
| Distribution | Ubuntu 24.04 |
| Kernel | 6.8 |
| CUDA | 13.2.1 |
| Python | 3.12 |
| Container runtime | Docker with NVIDIA runtime |
| Power mode | MAXN during validation and operation |
| Clocks | Locked with `jetson_clocks` during measurement |

JetPack 7.2 allows Orin to run Arm64 SBSA containers. The Jetson Compose file pins
`ghcr.io/nvidia-ai-iot/vllm:r38.2.arm64-sbsa-cu130-24.04`. The registry manifest is Linux
arm64 with digest `sha256:b587dd56b4cb076209ad5156a626ac75f5a976d0e8e7d1e6a9fccd56d1bd65e8`.
The image contains Ubuntu 24.04, CUDA 13.0, Python 3.12, and vLLM 0.19.0.

The image tag targets r38.2, while the host contract is Jetson Linux 39.2. Its build
metadata advertises CUDA architecture 11.0, while AGX Orin uses sm_87. Treat this pairing
as a deployment exception until Phase 0 confirms CUDA kernel execution. Do not change the
JetPack, CUDA, Jetson Linux, or base-image family without a separate compatibility
validation.

Platform references:

- [NVIDIA JetPack SDK Downloads and Notes](https://developer.nvidia.com/embedded/jetpack/downloads)
- [NVIDIA JetPack 7.2 Orin validation thread](https://forums.developer.nvidia.com/t/jetpack-7-2-jetson-linux-r39-2-on-jetson-agx-orin-developer-kit-getting-started-and-feedback-thread/372156)

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
through a secure channel. The script never prints the token. It also creates
`config/remote-gpu.env` with the selected GPU UUID for every model service.

Available options:

```text
--env-file PATH       Use a non-default A6000 Compose environment file
--health-timeout SEC  Change the model startup timeout from 600 seconds
--no-build            Start already-built images
--skip-power-setup    Do not set Jetson MAXN mode or lock clocks
--reallocate-gpus     Stop A6000 services and recalculate GPU placement from free memory
--remove-images       Remove project-built images during uninstall
```

The script does not replace `config/local.yaml`, `config/remote-server.local.yaml`, or an
existing `.env`. Routine deployment reuses `config/remote-gpu.env` so GPU enumeration or
temporary memory usage cannot move a resident model unexpectedly. Use
`--reallocate-gpus` after adding, removing, or repurposing GPUs. The option stops resident
services before measuring free memory. The script does not start the interactive
orchestrator. Continue with the runtime command printed after all resident services become
healthy.

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

Uninstall never removes model artifacts, logs, SQLite data, `.env`,
`config/remote-gpu.env`, local YAML overrides, or upstream base images.
`--remove-images` removes only images named for this project.

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
├── llm/
│   ├── Qwen3-14B-AWQ/
│   └── Qwen3-30B-A3B-Instruct-2507-AWQ/
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
test -s models/llm/Qwen3-14B-AWQ/config.json
test -s models/llm/Qwen3-30B-A3B-Instruct-2507-AWQ/config.json
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

### Install JetPack 7.2

Install JetPack 7.2 through the NVIDIA unified ISO or the matching Jetson Linux 39.2
Yocto image. Do not treat this migration as an in-place package update from JetPack 6.
Complete the NVIDIA flashing workflow, reboot, and verify Jetson Linux 39.2 before
deploying Kotonoha.

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
- `/etc/nv_tegra_release` identifies Jetson Linux 39.2.
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
mkdir -p data/logs models/llm
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
  models_dir: /models/llm

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

Confirm that every Jetson role resolves to the pinned Arm64 image family:

- `ghcr.io/nvidia-ai-iot/vllm:r38.2.arm64-sbsa-cu130-24.04`

### Build Jetson images

```bash
docker compose -f docker/compose.yaml build asr asr-verify tts orchestrator
```

Review the build output for the following conditions:

- PyTorch reports a CUDA build.
- vLLM reports version 0.19.0 in the ASR and translation images.
- CTranslate2 and faster-whisper import in the verification image.
- `onnxruntime` and DeepFilterNet installation status is explicit.
- At least one TTS backend installs.

The orchestrator and TTS Dockerfiles currently permit selected target dependencies to
fail during image construction. The AArch64 CTranslate2 wheel is published, but its GPU
path documents CUDA 12 rather than CUDA 13. A successful image build therefore does not
prove faster-whisper GPU execution or TTS backend loading. Service health checks and
target measurements remain mandatory.

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
test -s models/Qwen3-ASR-1.7B/config.json
test -d models/faster-whisper-large-v3
test -s models/faster-whisper-large-v3/config.json
test -s models/Qwen3-TTS-0.6B/config.json
test -s models/llm/Qwen3-30B-A3B-Instruct-2507-AWQ/config.json
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
  models_dir: /models/llm

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
LLM_IMAGE=vllm/vllm-openai:v0.19.1
LLM_PROFILE=moe
LLM_MAX_MODEL_LEN=4096
LLM_GPU_MEMORY_UTILIZATION=0.55
GPU_ALLOCATION_MODE=auto
GPU_NAME_FILTER=A6000
GPU_MEMORY_RESERVE_MIB=1024
LLM_GPU_MEMORY_MIB=27648
ASR_GPU_MEMORY_MIB=10240
ASR_VERIFY_GPU_MEMORY_MIB=6144
TTS_GPU_MEMORY_MIB=3072
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1
```

```bash
chmod 600 .env
```

Existing deployments must replace any `LLM_IMAGE` value containing `llama.cpp`, rename
`LLM_CTX` to `LLM_MAX_MODEL_LEN`, and update host overrides from `/models/gguf` to
`/models/llm`. The deployment script rejects an obsolete llama.cpp image selection.
Existing GGUF files are not read by the vLLM service and can be archived after the AWQ
snapshots pass target validation.

### Configure GPU allocation

Automatic allocation uses current free memory on the first deployment or after
`--reallocate-gpus`. It places larger reservations first and minimizes projected memory
utilization across eligible GPUs. `GPU_NAME_FILTER=A6000` excludes unrelated NVIDIA GPUs.
Set an empty filter to permit every GPU reported by `nvidia-smi`.

The checked-in reservation defaults are deployment guards, not measured peak memory
values.

| Role | Environment variable | Default reservation |
|---|---|---:|
| Translation LLM | `LLM_GPU_MEMORY_MIB` | 27,648 MiB |
| Primary ASR | `ASR_GPU_MEMORY_MIB` | 10,240 MiB |
| Verification ASR | `ASR_VERIFY_GPU_MEMORY_MIB` | 6,144 MiB |
| TTS | `TTS_GPU_MEMORY_MIB` | 3,072 MiB |
| Per-GPU safety reserve | `GPU_MEMORY_RESERVE_MIB` | 1,024 MiB |

Measure peak memory with all services resident, then raise reservations when retained
evidence exceeds a default. The allocator rejects a placement when no eligible GPU has
enough remaining capacity.

For fixed placement, set `GPU_ALLOCATION_MODE=manual` and define every device by stable
GPU UUID:

```dotenv
GPU_ALLOCATION_MODE=manual
ASR_GPU_DEVICE=GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
ASR_VERIFY_GPU_DEVICE=GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
LLM_GPU_DEVICE=GPU-11111111-2222-3333-4444-555555555555
TTS_GPU_DEVICE=GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
```

Use UUIDs instead of indexes because NVIDIA does not guarantee enumeration order across
reboots. Manual assignments still undergo aggregate capacity validation.

The Python services receive `KOTONOHA_SERVICE_TOKEN`. The vLLM translation launcher
passes the same token through `--api-key`. Restrict ports 8001-8004 to the Jetson even
when bearer authentication is enabled.

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
source config/remote-gpu.env
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

The ASR image build checks that the vLLM package contains the Qwen3-ASR module without
importing vLLM. Docker BuildKit does not attach the NVIDIA runtime, so CUDA-aware imports
belong to deployment. `scripts/deploy.sh` starts temporary ASR and LLM containers with the
Compose GPU reservation and verifies PyTorch CUDA, the GPU identity, and vLLM before
starting resident services. Flash Attention remains best-effort in the TTS target; its
failure must appear in the build log and TTS health must report the backend that loaded.

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

Confirm that Compose rendered the generated UUID assignments:

```bash
cat config/remote-gpu.env
docker compose -f docker/compose.remote.yaml config | sed -n '/device_ids:/,+2p'
```

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

## Hardware Performance Acceptance

Complete [Performance Measurement](../performance/measurement.md) before approving either
deployment target. That procedure owns benchmark conditions, acceptance thresholds,
evidence requirements, and report generation.

| Target | Required evidence |
|---|---|
| Jetson AGX Orin | `PHASE0.md`, accepted local configuration patch, thermal evidence |
| RTX A6000 | `PERFORMANCE.md`, accepted remote configuration patch, link and residency evidence |

Do not copy a generated configuration patch until the corresponding report passes all
target-specific acceptance criteria.

## Operations

Use the [Service Runbook](../operations/service-runbook.md) for service status, lifecycle,
backup, rollback, security controls, and troubleshooting. Use
[Observability](../operations/observability.md) for turn metrics and latency analysis.

## Deployment Acceptance Checklist

### Source and configuration

- [ ] Jetson and A6000 use the same approved Git commit.
- [ ] Working trees contain no unexplained changes.
- [ ] Host-specific files are ignored by Git and protected with mode 600.
- [ ] `docker compose config --quiet` succeeds on each host.
- [ ] No unvalidated JetPack, CUDA, Jetson Linux, or base-image change is present.

### Models and services

- [ ] Required model artifacts exist at the configured absolute paths.
- [ ] All selected backends loaded; Python health responses contain `"ok": true`.
- [ ] The vLLM translation health endpoint responds and the selected AWQ model is loaded.
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
