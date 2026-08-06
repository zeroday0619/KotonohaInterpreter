# Docker Accelerator Profiles

Docker profiles provide the container-side contract for an accelerator profile.

## Naming

Use the same `<vendor>.<family>.<model>` identifier as the application profile. Store the
environment file at:

```text
docker/profiles/accelerators/<vendor>/<family>/<model>.env
```

The profile defines only container concerns:

| Variable | Purpose |
|---|---|
| `ACCELERATOR_PROFILE` | Application and Docker profile identifier |
| `CONTAINER_RUNTIME` | Docker runtime name, for example `nvidia` |
| `GPU_DRIVER` | Compose device reservation driver |
| `ACCELERATOR_DEVICE_ENV` | Vendor-specific visible-device environment variable |
| `VLLM_NVML_PATCH` | Whether the NVIDIA vLLM NVML compatibility patch is required |
| `ACCELERATOR_*_IMAGE` | Vendor runtime images shared by the service roles |

Profiles use Docker-compatible `KEY=VALUE` entries. The deployment and spike scripts
apply a profile value only when the caller has not already set that variable, so explicit
environment variables retain priority. The scripts forward the values through privileged
Docker access.

The current profiles target NVIDIA CUDA. Additional accelerator families require a
validated base image, runtime name, device environment variable, Compose driver, and
service-specific startup probes before deployment support is added.
