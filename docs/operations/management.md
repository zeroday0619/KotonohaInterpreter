# Management Script

## Purpose

`scripts/manage.sh` provides one confirmed operator entry point for workstation setup,
model staging, target-host preparation, deployment, GPU allocation, hardware benchmarks,
network benchmarks, and repository validation. It detects the current equipment when a
target is omitted and delegates execution to the specialized scripts.

## Workflow

### Development workstation

```bash
bash scripts/manage.sh setup workstation
bash scripts/manage.sh check
```

Use `setup workstation --eval` when the workstation runs COMET evaluation:

```bash
bash scripts/manage.sh setup workstation --eval
```

### New inference host

Run the following sequence on each target host:

```bash
bash scripts/manage.sh models fetch
bash scripts/manage.sh models verify
bash scripts/manage.sh detect
bash scripts/manage.sh setup
```

Automatic detection selects `jetson` from `/etc/nv_tegra_release`, `a6000` from the GPU
name reported by `nvidia-smi`, and `workstation` on macOS. An explicit target overrides
the detected value. Target setup validates the host, Docker NVIDIA runtime, model
artifacts, and Compose configuration. It creates missing host-specific configuration and
GPU allocation files. It does not build images, change Jetson power state, or start
services.

After setup completes, run the hardware benchmark before deployment acceptance:

```bash
bash scripts/manage.sh benchmark jetson
bash scripts/manage.sh deploy jetson
```

## Command Reference

| Operation | Command |
|---|---|
| Detect current equipment | `bash scripts/manage.sh detect` |
| Install workstation dependencies | `bash scripts/manage.sh setup workstation` |
| Include evaluation dependencies | `bash scripts/manage.sh setup workstation --eval` |
| Download model artifacts | `bash scripts/manage.sh models fetch` |
| Validate model artifacts | `bash scripts/manage.sh models verify` |
| Maintain translation catalogs | `bash scripts/manage.sh i18n check` |
| Prepare Jetson configuration | `bash scripts/manage.sh setup jetson` |
| Prepare A6000 configuration | `bash scripts/manage.sh setup a6000` |
| Run all Jetson hardware spikes | `bash scripts/manage.sh benchmark jetson` |
| Run one A6000 spike | `bash scripts/manage.sh benchmark a6000 --only 3` |
| Measure the remote link | `bash scripts/manage.sh benchmark link --samples 20 --seconds 6` |
| Allocate A6000 GPUs | `bash scripts/manage.sh gpu allocate` |
| Deploy Jetson services | `bash scripts/manage.sh deploy jetson` |
| Deploy A6000 services | `bash scripts/manage.sh deploy a6000` |
| Start the integrated TUI | `bash scripts/manage.sh tui` |
| Remove Jetson containers | `bash scripts/manage.sh uninstall jetson` |
| Remove A6000 containers | `bash scripts/manage.sh uninstall a6000` |
| Run environment diagnostics | `bash scripts/manage.sh doctor` |
| Run repository quality gates | `bash scripts/manage.sh check` |

Arguments after the operation and target pass to the specialized implementation. For
example, the following command preserves the deployment script's image-build option:

```bash
bash scripts/manage.sh deploy a6000 --no-build
```

The `setup`, `benchmark`, `deploy`, and `uninstall` operations accept `auto` or an omitted
target. `KOTONOHA_EQUIPMENT=workstation|jetson|a6000` provides an explicit automation
override when host interfaces are unavailable inside a controlled execution environment.

Catalog maintenance accepts `extract`, `update`, `compile`, and `check`:

```bash
bash scripts/manage.sh i18n extract
bash scripts/manage.sh i18n update
bash scripts/manage.sh i18n compile
bash scripts/manage.sh i18n check
```

## Confirmation

Every operation requires a `y` or `n` response before execution. Non-interactive jobs
must pass `-y` or `--yes`:

```bash
bash scripts/manage.sh -y check
bash scripts/manage.sh -y deploy
```

Uninstall requests a separate image-removal decision. `--remove-images` and
`--keep-images` select the result without the second prompt. `--yes` selects image
removal unless `--keep-images` is present.

## Dry Run

Place `--dry-run` before the operation to print delegated commands without executing
them. Dry runs still require confirmation because they represent a complete management
task:

```bash
bash scripts/manage.sh -y --dry-run setup jetson
bash scripts/manage.sh -y --dry-run benchmark a6000 --only 3
bash scripts/manage.sh -y --dry-run deploy a6000 --no-build
```

Model verification prints its resolved artifact directory during a dry run. It does not
read or modify model files.

## Environment

The management script preserves environment variables consumed by delegated workflows.
Docker workflows that require sudo forward an explicit Compose-variable allowlist because
the standard sudo policy removes exported shell variables.

| Variable | Purpose |
|---|---|
| `MODELS_DIR` | Model download and validation root |
| `KOTONOHA_SERVICE_TOKEN` | Remote service authentication and link benchmark |
| `SPIKE_VLLM_IMAGE` | Hardware Spike vendor vLLM image override |
| `SPIKE_ASR_IMAGE` | Hardware Spike prepared ASR image override |
| `SPIKE_TTS_IMAGE` | Hardware Spike TTS image override |
| `TTS_IMAGE` | Resident TTS service vLLM-Omni base-image override |
| `SPIKE_GPU_DEVICE` | GPU exposed to Spike containers |
| `OUT` | Hardware Spike output directory |

Docker accelerator defaults are stored under
`docker/profiles/accelerators/<vendor>/<family>/<model>.env`. The deployment and spike
scripts load the profile matching the selected target before invoking Docker. Explicit
environment variables override profile defaults.

## Safety

- `setup jetson` and `setup a6000` use deployment `--prepare-only` mode.
- Setup does not build images, start services, or stop resident services.
- Setup never combines `--prepare-only` with GPU reallocation because reallocation can
  stop resident A6000 services.
- Uninstall preserves model artifacts, configuration, secrets, logs, and SQLite data.
- Image removal enumerates only repositories whose names start with
  `kotonohainterpreter-`; NVIDIA, vLLM, PyTorch, and other upstream images remain intact.
- `--remove-images` and `--keep-images` remain valid only for uninstall operations.
- Model verification reports every missing required artifact before returning failure.

## Privilege Escalation

The management entry point runs as the deployment account. Delegated Docker workflows
test direct daemon access first and use `sudo docker` only when the Docker socket requires
elevation. Jetson power commands use `sudo` only for operations that require root. This
keeps generated configuration, model artifacts, logs, and caches owned by the deployment
account.
