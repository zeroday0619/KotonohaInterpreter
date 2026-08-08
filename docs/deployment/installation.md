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
Jetson. Phase 0 determines the ASR runtime and translation model profile, and validates
the fixed vLLM-Omni TTS runtime.
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
TTS. `custom` selects local or remote placement for each role through the `placement`
mapping. Remote ASR traffic uses multipart binary PCM; it does not use shared memory or
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

The Jetson Compose file pins `nvcr.io/nvidia/vllm:26.07-py3`. The manifest list digest is
`sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268`; its Linux
arm64 manifest digest is
`sha256:1de8e6bfdb4c81c1f31a806cc9b13b5c6352714a7cec87f4d24964bcc91159b2`. The image
contains Ubuntu 24.04, CUDA 13.3.1, Python 3.12, and vLLM `0.24.0+092c4842`.

The image architecture list does not include Orin compute capability 8.7, and its CUDA
13.3 toolkit is newer than the Jetson host's CUDA 13.2 driver stack. Treat the image as an
explicit Phase 0 candidate until the target confirms container startup, CUDA access,
model loading, and kernel execution. Manifest inspection does not establish runtime
compatibility.

Platform references:

- [NVIDIA JetPack SDK Downloads and Notes](https://developer.nvidia.com/embedded/jetpack/downloads)
- [NVIDIA JetPack 7.2 Orin validation thread](https://forums.developer.nvidia.com/t/jetpack-7-2-jetson-linux-r39-2-on-jetson-agx-orin-developer-kit-getting-started-and-feedback-thread/372156)

### A6000 host

| Component | Requirement |
|---|---|
| GPU | NVIDIA RTX A6000 48 GB, sm_86 |
| Host architecture | x86_64 |
| Container runtime | Docker Engine, Compose plugin, NVIDIA Container Toolkit |
| Driver | Must initialize CUDA in the configured vLLM and vLLM-Omni images |
| Network | Stable route from the Jetson to TCP 8001-8004 |

The A6000 ASR and translation services pin `nvcr.io/nvidia/vllm:26.07-py3`. Its manifest
list digest is `sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268`.
The selected Linux amd64 manifest contains Ubuntu 24.04, Python 3.12, CUDA 13.3.1, and
vLLM `0.24.0+092c4842`. Its build metadata includes compute capability 8.6.

The repository does not specify an exact host driver version. Installation is blocked
until `nvidia-smi` works on the host and both configured CUDA images initialize through
the NVIDIA Container Toolkit. Manifest inspection does not verify Qwen3-ASR loading,
vLLM beam search, AWQ loading, latency, or concurrent residency on the A6000.

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
executes the same CUDA device-count path used by vLLM workers, starts resident model
services, and waits for health checks.

On the Jetson, automatic detection permits the target to be omitted:

```bash
bash scripts/manage.sh deploy
```

The same command detects an RTX A6000 server:

```bash
bash scripts/manage.sh deploy
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
--prepare-only        Validate the host and create configuration without deployment
--reallocate-gpus     Stop A6000 services and recalculate GPU placement from free memory
--remove-images       Remove project-built images during uninstall
--keep-images         Preserve project-built images during uninstall
-y, --yes             Confirm every management prompt without reading standard input
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
operations. The sudo path forwards an explicit allowlist of Compose interpolation
variables. This preserves Jetson image overrides, A6000 model paths, service tokens, and
per-role GPU assignments without forwarding the complete caller environment. Jetson
power commands also use `sudo` when the script is not running as root. Do not run the
complete script with `sudo`; host-specific files should remain owned by the deployment
account.

### Quick uninstall

Remove project containers and the Compose network:

```bash
# Jetson
bash scripts/manage.sh uninstall jetson

# A6000
bash scripts/manage.sh uninstall a6000
```

Uninstall asks whether to remove images built by this project. Select the behavior
explicitly for automation:

```bash
bash scripts/manage.sh -y uninstall jetson --remove-images
bash scripts/manage.sh -y uninstall a6000 --keep-images
```

Uninstall never removes model artifacts, logs, SQLite data, `.env`,
`config/remote-gpu.env`, local YAML overrides, or upstream base images.
Image removal enumerates only image repositories with the `kotonohainterpreter-` prefix.
It does not remove NVIDIA NGC, vLLM, PyTorch, or other upstream images.

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
bash scripts/manage.sh models fetch
du -sh models/*
```

The script creates the following layout:

```text
models/
├── Qwen3-ASR-0.6B/
├── Qwen3-ASR-0.6B-hf/
├── Voxtral-Mini-4B-Realtime-2602/
├── Qwen3-TTS-0.6B/
├── faster-whisper-large-v3/
├── llm/
│   ├── translategemma-4b-it/
│   └── translategemma-12b-it/
└── silero_vad.onnx
```

The Qwen3-TTS snapshot is required by the resident vLLM-Omni service on both deployment
hosts. The service never downloads model artifacts at startup.

### Verify artifacts

```bash
test -s models/silero_vad.onnx
test -d models/Qwen3-ASR-0.6B
test -d models/Qwen3-ASR-0.6B-hf
test -d models/Voxtral-Mini-4B-Realtime-2602
test -d models/faster-whisper-large-v3
test -s models/Qwen3-TTS-0.6B/config.json
test -s models/llm/translategemma-4b-it/config.json
test -s models/llm/translategemma-12b-it/config.json
```

`fetch_models.sh` uses Hugging Face local directories rather than a cache-only layout.
Offline deployments must configure model services to use the mounted absolute paths
shown below. A repository ID alone can trigger a cache lookup that does not resolve the
local directory.

TranslateGemma is gated on Hugging Face. Accept its license with the account used for
model preparation before running `models fetch`; deployment remains offline afterward.

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
  model_id: /models/Qwen3-ASR-0.6B-hf
  vllm_model_id: /models/Qwen3-ASR-0.6B
  vllm_realtime_architecture: qwen3_asr

asr_verify:
  model_id: /models/faster-whisper-large-v3

llm:
  models_dir: /models/llm

```

The deployment script preserves an existing `config/local.yaml`. Installations created
before the Jetson ASR changed to 0.6B can therefore still contain 1.7B paths. Update only
the three `asr` fields above while preserving the other local values. The deployment
preflight reads the effective merged configuration inside the ASR container and rejects
stale paths before starting resident services.

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

Confirm that the Jetson ASR, verification, translation, and orchestrator roles resolve to
the pinned NGC image:

- `nvcr.io/nvidia/vllm:26.07-py3`

The TTS build must use `vllm/vllm-omni:v0.26.0` as its base image. Its multi-platform
manifest digest is
`sha256:5cba1538c6f8ee81e8bea6708c24e68d7b2640f466a9fbf2ef15e68f2168b48b` and includes
Linux arm64 and amd64 variants. The manifest does not establish Jetson compatibility.

### Build Jetson images

```bash
docker compose -f docker/compose.yaml build asr asr-verify llm tts orchestrator
```

Review the build output for the following conditions:

- PyTorch reports a CUDA build.
- vLLM reports version `0.24.0+092c4842` in the ASR and translation images.
- CTranslate2 and faster-whisper import in the verification image.
- `onnxruntime` and DeepFilterNet installation status is explicit.
- The TTS service build resolves the official vLLM-Omni base to the arm64 manifest.

The orchestrator Dockerfile permits selected target dependencies to fail during image
construction. Target execution showed that the installed AArch64 CTranslate2 artifact
was not compiled with CUDA support. The Jetson verification service therefore uses the
documented faster-whisper CPU INT8 path, while the A6000 overlay retains CUDA FP16. The
verification service synchronizes the `asr-verify` extra and the application into one
locked environment, then checks CPU INT8 capability during construction. Jetson model
loading, transcription latency, and memory use remain target measurements. An image pull
or successful build also does not prove vLLM-Omni TTS loading.

The Jetson images patch vLLM CUDA platform detection to skip NVML and use the non-NVML
platform. Jetson's `nvgpu` runtime can segfault inside `nvmlInit()` instead of returning a
Python exception. Deployment probes use the raw CUDA device-count API for the same reason.

Jetson TTS uses `docker/tts/qwen3_tts_jetson.yaml` through `TTS_DEPLOY_CONFIG`. The profile
limits both Qwen3-TTS stages to one sequence, sets each stage memory budget to `0.30`, and
reduces Stage 0 prefill batching to 512 tokens. The upstream vLLM-Omni 0.26.0 profile is
tuned for a single H100 with 64 sequences per stage, which is not appropriate for the
sequential Jetson interpreter. A6000 keeps the upstream profile.
The patch does not establish model or kernel compatibility; those remain target
measurements.

Jetson translation keeps the `translategemma-4b-it` checkpoint and selects the
`nvidia.jetson.agx-orin` accelerator profile from
`config/profiles/accelerators/nvidia/jetson/agx-orin.yaml`. The profile uses FP8 KV cache,
one active sequence, a `0.35` GPU memory utilization limit, disabled prefix caching, and
zero limits for unused image, audio, and video inputs. The A6000 deployment selects
`nvidia.rtx.a6000` from `config/profiles/accelerators/nvidia/rtx/a6000.yaml`. It uses
automatic KV cache dtype selection, automatic prefix caching, a `4096` chunked-prefill
token budget, and compilation mode `2` with CUDA graph capture enabled by setting
`enforce_eager` to `false`. Its GPU memory utilization limit is `0.90`. CUDA Graph capture
is limited to batch sizes `[1, 2, 4]`, and compiled artifacts are persisted under
`/models/vllm-compile-cache`.

### Start model services

Prefer `bash scripts/manage.sh deploy jetson`. It validates the effective 0.6B ASR paths,
the combined Jetson ASR, LLM, and TTS memory budget, and starts the resident services in
dependency order. Each service must report healthy before the next service starts. This
prevents a late ASR cache allocation from competing with already initialized model engines.
It does not change the A6000 deployment path. It also validates the CPU INT8 verification
backend, raw CUDA device access, CUDA imports, and
TranslateGemma model configuration before Compose starts the resident containers. A
segmentation fault in the device-count probe stops deployment before resident services
enter a restart loop. The commands below are the manual path after those checks have
passed. The manual Compose path does not perform the memory-budget check or sequential
startup; use the management script for Jetson memory coordination.

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

Set `audio.input_device` and `audio.output_device` through the configuration TUI. It stores
the exact `device name, host API` selector because numeric PortAudio indexes can change
after a reboot or USB reconnect. The audio test reads 750 ms from the selected microphone,
reports the measured level, and sends a low-volume tone to the selected speaker. The
runtime first tries the configured sample rate and mono channel layout, then uses the
device default rate or stereo when ALSA rejects the requested format. The configuration
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
test -d models/Voxtral-Mini-4B-Realtime-2602
test -s models/Voxtral-Mini-4B-Realtime-2602/config.json
test -d models/faster-whisper-large-v3
test -s models/faster-whisper-large-v3/config.json
test -s models/Qwen3-TTS-0.6B/config.json
test -s models/llm/translategemma-12b-it/config.json
```

### Create the remote service override

The quick deployment script copies `config/remote-server.local.example.yaml` when the
override does not exist. For manual deployment, create `config/remote-server.local.yaml`:

```yaml
# A6000-specific paths for an offline deployment.
asr:
  vllm_model_id: /models/Voxtral-Mini-4B-Realtime-2602
  vllm_realtime_architecture: voxtral

asr_verify:
  model_id: /models/faster-whisper-large-v3

llm:
  models_dir: /models/llm

```

```bash
chmod 600 config/remote-server.local.yaml
```

### Define remote deployment variables

Create a repository-root `.env` file. Replace the token placeholder with the output of
`openssl rand -hex 32`. Do not commit the file.

```dotenv
KOTONOHA_SERVICE_TOKEN=<64-hex-character-random-token>
PROMETHEUS_PORT=9091
MODELS_DIR=../models
REMOTE_BASE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
REMOTE_ASR_BASE=nvcr.io/nvidia/vllm:26.07-py3
TTS_IMAGE=vllm/vllm-omni:v0.26.0
LLM_IMAGE=nvcr.io/nvidia/vllm:26.07-py3
LLM_MAX_MODEL_LEN=2048
LLM_GPU_MEMORY_UTILIZATION=0.90
LLM_MAX_NUM_BATCHED_TOKENS=4096
LLM_ENABLE_PREFIX_CACHING=1
LLM_COMPILATION_MODE=2
GPU_ALLOCATION_MODE=auto
GPU_NAME_FILTER=A6000
GPU_MEMORY_RESERVE_MIB=1024
LLM_GPU_MEMORY_MIB=43008
ASR_GPU_MEMORY_MIB=14336
ASR_VERIFY_GPU_MEMORY_MIB=6144
TTS_GPU_MEMORY_MIB=3072
TTS_GPU_MEMORY_UTILIZATION=0.25
TTS_ENFORCE_EAGER=1
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1
```

```bash
chmod 600 .env
```

The deployment script loads the container profile matching the selected accelerator.
NVIDIA defaults are defined in
`docker/profiles/accelerators/nvidia/jetson/agx-orin.env` and
`docker/profiles/accelerators/nvidia/rtx/a6000.env`. The profiles select the Docker
runtime, Compose GPU driver, visible-device environment variable, base images, and the
vLLM NVML patch policy. Explicit environment variables override profile defaults.

To add another accelerator, create a matching application profile under
`config/profiles/accelerators/` and a Docker profile under
`docker/profiles/accelerators/`, then validate the base images, GPU runtime, device
reservation, and service startup probes on the target host.

Existing deployments must set both `REMOTE_ASR_BASE` and `LLM_IMAGE` to
`nvcr.io/nvidia/vllm:26.07-py3`, or remove those variables to use the Compose defaults.
They must also rename `LLM_CTX` to `LLM_MAX_MODEL_LEN` and update host overrides from
`/models/gguf` to `/models/llm`. The deployment script rejects any explicitly configured
A6000 ASR or LLM image that differs from the pinned NGC image. Existing GGUF files are not
read by the vLLM service and can be archived after the AWQ snapshots pass target
validation.

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
| Primary ASR | `ASR_GPU_MEMORY_MIB` | 14,336 MiB |
| Verification ASR | `ASR_VERIFY_GPU_MEMORY_MIB` | 6,144 MiB |
| TTS | `TTS_GPU_MEMORY_MIB` | 3,072 MiB |
| Per-GPU safety reserve | `GPU_MEMORY_RESERVE_MIB` | 1,024 MiB |

The A6000 Voxtral startup measurement required 3.27 GiB of KV cache for a 4,096-token
context, but a 0.20 engine allocation left only 0.41 GiB. The resident profile therefore
uses 0.28 and reserves 14,336 MiB for primary ASR. Measure peak memory with all services
resident and retain the resulting evidence. The allocator rejects a placement when no
eligible GPU has enough remaining capacity; the current guarded role budgets do not fit
on one 48 GiB A6000 with the safety reserve.

Before starting the services, the deployment script resolves the ASR configuration inside
the built container and rejects an effective value below 0.28. This catches an older
`config/remote-server.local.yaml` value or a
`KOTONOHA__ASR__VLLM_GPU_MEMORY_UTILIZATION` environment override that would otherwise
silently take precedence over `config/remote-server.yaml`.

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

The Python services receive `KOTONOHA_SERVICE_TOKEN`. The TTS FastAPI layer enforces the
same bearer token before dispatching requests to its in-process vLLM-Omni engine.
Restrict ports 8001-8004 to the Jetson even when bearer authentication is enabled.

TTS builds `kotonohainterpreter-tts` from the official multi-platform
`vllm/vllm-omni:v0.26.0` base and mounts the local Qwen3-TTS snapshot at
`/models/Qwen3-TTS-0.6B`. `Dockerfile.tts` creates one uv-managed environment with access
to the base image's system packages, then installs the locked Kotonoha dependencies into
that environment. It does not install another vLLM, vLLM-Omni, or Transformers runtime.
A remote transport failure before the first PCM chunk is retried against the resident
Jetson service. The base manifest contains both Linux arm64 and amd64 variants, but only
Spike 2 model loading and CUDA kernel execution can establish compatibility on either
target.

`kotonoha.services._tts_server` is the public FastAPI service. Its lifespan validates the
model snapshot and matching vLLM/vLLM-Omni versions, constructs `AsyncOmni`,
`OpenAIServingModels`, and `OmniOpenAIServingSpeech` in the uvicorn process, and returns
the runtime's raw PCM stream from `/v1/audio/speech`. It does not open a second HTTP
server or use an internal HTTP proxy. The pinned 0.26.0 base satisfies the
[upstream installation contract](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/)
without installing a second inference runtime. Spike 2 starts the same FastAPI service.

### Validate and build the remote stack

Load the token into the Compose process, then validate the rendered configuration:

```bash
set -a
source .env
source config/remote-gpu.env
set +a
docker compose -f docker/compose.remote.yaml config --quiet
docker compose -f docker/compose.remote.yaml config --images
docker compose -f docker/compose.remote.yaml build metrics asr asr-verify llm tts
```

The build produces five project-specific service images. TTS retains the upstream
vLLM-Omni runtime through its base image:

| Service | Image | Dockerfile target |
|---|---|---|
| Primary ASR | `kotonohainterpreter-asr:latest` | `asr` |
| Verification ASR | `kotonohainterpreter-asr-verify:latest` | `asr-verify` |
| Translation LLM | `kotonohainterpreter-llm:latest` | `llm` |
| TTS | `kotonohainterpreter-tts:latest` | `docker/Dockerfile.tts` |
| Metrics receiver | `kotonohainterpreter-metrics:latest` | `metrics` |

The targets share a cached application layer but install and verify role-specific runtime
dependencies. The common layer imports `pydantic_settings` during the build. A missing
core dependency therefore fails the image build instead of entering a restart loop.
The remote stack executes the application code installed in each image. It mounts only
`config/`, `data/`, and the selected model directory; it does not expose the host source
tree or deployment scripts inside model-service containers. The metrics receiver mounts
`config/` read-only.
The Jetson stack retains a read-only repository mount for the integrated operator tools
and overlays writable `config/`, `data/`, and model mounts. Model services cannot modify
the checked-out source or deployment scripts.
The lock selects NumPy 2.x on Linux Python 3.12 for the NGC SciPy and scikit-learn stack,
while the macOS workstation retains NumPy 1.x. The A6000 ASR
target additionally synchronizes `a6000-asr`; that extra supplies `mistral-common[audio]`
for the Voxtral tokenizer and audio preprocessing without installing a second vLLM
runtime. The image build checks the Transformers lazy imports required by vLLM.

The Jetson ASR image checks for Qwen3-ASR batch and realtime modules. The A6000 ASR image
checks for Voxtral Realtime, the vLLM realtime connection, and the Mistral audio
dependencies without importing vLLM.
The Jetson translation image keeps locked application packages in
`/opt/kotonoha-venv` with system-site access to the NGC vLLM, PyTorch, and CUDA packages.
The project environment does not modify the vendor system Python.
The pinned NVIDIA vLLM 0.24.0 runtime returns only the audio embeddings for a mixed
offline Voxtral prefill, omitting the trailing text-token position. The A6000 image
applies a version-scoped compatibility patch that aligns audio embeddings through
vLLM's multimodal position mask, preserving both N-best batch transcription and the
unmodified realtime path. A patch context mismatch intentionally fails the image build.
Docker BuildKit does not attach the NVIDIA runtime, so CUDA-aware imports belong to
deployment. `scripts/deploy.sh` starts temporary ASR and LLM containers with the
Compose GPU reservation before starting resident services. The LLM probe uses the
application virtual environment, imports `GenerationMixin`, `AsyncEngineArgs`, and the
in-process engine builder, and constructs the selected model configuration. The model
probe also verifies that TranslateGemma retains both nested RoPE mappings. The same check
imports vLLM-Omni and initializes CUDA in a temporary TTS container. Spike 2 remains
responsible for model loading, FlashAttention kernel execution, and Speech API PCM
measurements.

### Start and verify the remote stack

```bash
docker compose -f docker/compose.remote.yaml up -d metrics asr asr-verify llm tts
docker compose -f docker/compose.remote.yaml ps
docker compose -f docker/compose.remote.yaml logs --tail=200
```

From the A6000 host:

```bash
curl -fsS http://127.0.0.1:8001/health | python3 -m json.tool
curl -fsS http://127.0.0.1:8002/health | python3 -m json.tool
curl -fsS http://127.0.0.1:8003/health | python3 -m json.tool
curl -fsS http://127.0.0.1:8004/health | python3 -m json.tool
curl -fsS -H "Authorization: Bearer ${KOTONOHA_SERVICE_TOKEN}" \
  http://127.0.0.1:9091/metrics | head
```

Confirm concurrent residency. A service that loads successfully in isolation can still
fail when all models occupy the GPU. Check `nvidia-smi` after all four services report
healthy and retain the output in the deployment record.

Confirm that Compose rendered the generated UUID assignments:

```bash
cat config/remote-gpu.env
docker compose -f docker/compose.remote.yaml config | sed -n '/device_ids:/,+2p'
```

### Deploy the Web control center

The Web image contains the CPU orchestrator and browser assets. It does not inherit a
CUDA base image and does not request the NVIDIA container runtime. The selected base
Compose stack owns accelerator attachment for the four resident model services.

Start the complete A6000 browser stack:

```bash
bash scripts/manage.sh web a6000
curl -fsS http://127.0.0.1:8080/health | python3 -m json.tool
```

Start the equivalent Jetson browser stack on the Jetson host:

```bash
bash scripts/manage.sh web jetson
curl -fsS http://127.0.0.1:8080/health | python3 -m json.tool
```

The command passes `.env` to Compose when the file exists. A6000 deployment combines
`compose.remote.yaml`, `compose.web.yaml`, and `compose.web.a6000.yaml`. Jetson deployment
combines `compose.yaml` and `compose.web.yaml`.

The default listener is loopback. For remote browser clients, place an authenticated TLS
reverse proxy on the host and set the application listener explicitly:

```dotenv
KOTONOHA_WEB_HOST=0.0.0.0
KOTONOHA_WEB_PORT=8080
KOTONOHA_WEB_SESSIONS=4
```

Browsers require a secure context for microphone capture. Do not expose port 8080 without
TLS and user authentication.

The configuration page writes `config/local.yaml` on Jetson and
`config/remote-server.local.yaml` on A6000. A valid save reconnects active browser
sessions. Model and accelerator changes call the authenticated reload endpoint for each
affected resident service. Requests targeting a service during its backend reload can
receive HTTP 503 and must be retried after `/health` reports `"ok": true`.

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
The translation service reads the same validated YAML directly when it restarts.

Remote model settings do not reload in the management request. Restart affected services
after saving:

| Changed section | Required A6000 restart |
|---|---|
| Primary ASR | `docker compose -f docker/compose.remote.yaml restart asr` |
| Verification ASR | `docker compose -f docker/compose.remote.yaml restart asr-verify` |
| Translation LLM | `docker compose -f docker/compose.remote.yaml restart llm` |
| TTS | `docker compose -f docker/compose.remote.yaml restart tts` |
| Metrics receiver | `docker compose -f docker/compose.remote.yaml restart metrics` |

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
- [ ] Translation health reports the in-process vLLM backend and TranslateGemma is loaded.
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
- [ ] TUI placement matches the intended `onboard`, `hybrid`, `remote`, or `custom` mode.
- [ ] `placement` and `failovers` appear in turn records.
- [ ] Audio egress in `remote` mode is approved by the deployment owner.

### Measurement and handoff

- [ ] Phase 0 reports all three verdicts from target execution.
- [ ] No unmeasured inference or latency value is reported as verified.
- [ ] One WAV replay completes and writes a turn record.
- [ ] One microphone utterance completes through playback.
- [ ] Backup and rollback locations are recorded.
- [ ] Operators have the status, log, restart, and escalation procedures.
