# Management Script

## Purpose

`scripts/manage.sh` provides one operator entry point for workstation setup, model
staging, target-host preparation, deployment, GPU allocation, hardware benchmarks,
network benchmarks, and repository validation. It delegates implementation to the
existing specialized scripts and preserves their options and validation behavior.

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
bash scripts/manage.sh setup jetson
```

Replace `jetson` with `a6000` on the external server. Target setup validates the host,
Docker NVIDIA runtime, model artifacts, and Compose configuration. It creates missing
host-specific configuration and GPU allocation files. It does not build images, change
Jetson power state, or start services.

After setup completes, run the hardware benchmark before deployment acceptance:

```bash
bash scripts/manage.sh benchmark jetson
bash scripts/manage.sh deploy jetson
```

## Command Reference

| Operation | Command |
|---|---|
| Install workstation dependencies | `bash scripts/manage.sh setup workstation` |
| Include evaluation dependencies | `bash scripts/manage.sh setup workstation --eval` |
| Download model artifacts | `bash scripts/manage.sh models fetch` |
| Validate model artifacts | `bash scripts/manage.sh models verify` |
| Prepare Jetson configuration | `bash scripts/manage.sh setup jetson` |
| Prepare A6000 configuration | `bash scripts/manage.sh setup a6000` |
| Run all Jetson hardware spikes | `bash scripts/manage.sh benchmark jetson` |
| Run one A6000 spike | `bash scripts/manage.sh benchmark a6000 --only 3` |
| Measure the remote link | `bash scripts/manage.sh benchmark link --samples 20 --seconds 6` |
| Allocate A6000 GPUs | `bash scripts/manage.sh gpu allocate` |
| Deploy Jetson services | `bash scripts/manage.sh deploy jetson` |
| Deploy A6000 services | `bash scripts/manage.sh deploy a6000` |
| Remove Jetson containers | `bash scripts/manage.sh uninstall jetson` |
| Remove A6000 containers | `bash scripts/manage.sh uninstall a6000` |
| Run environment diagnostics | `bash scripts/manage.sh doctor` |
| Run repository quality gates | `bash scripts/manage.sh check` |

Arguments after the operation and target pass to the specialized implementation. For
example, the following command preserves the deployment script's image-build option:

```bash
bash scripts/manage.sh deploy a6000 --no-build
```

## Dry Run

Place `--dry-run` before the operation to print delegated commands without executing
them:

```bash
bash scripts/manage.sh --dry-run setup jetson
bash scripts/manage.sh --dry-run benchmark a6000 --only 3
bash scripts/manage.sh --dry-run deploy a6000 --no-build
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
| `SPIKE_VLLM_IMAGE` | Hardware Spike ASR and LLM image override |
| `SPIKE_TTS_IMAGE` | Hardware Spike TTS image override |
| `SPIKE_GPU_DEVICE` | GPU exposed to Spike containers |
| `OUT` | Hardware Spike output directory |

## Safety

- `setup jetson` and `setup a6000` use deployment `--prepare-only` mode.
- Setup does not build images, start services, or stop resident services.
- Setup never combines `--prepare-only` with GPU reallocation because reallocation can
  stop resident A6000 services.
- Uninstall preserves model artifacts, configuration, secrets, logs, and SQLite data.
- `--remove-images` remains valid only for uninstall operations.
- Model verification reports every missing required artifact before returning failure.
