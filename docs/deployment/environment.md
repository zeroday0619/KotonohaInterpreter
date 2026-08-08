# Environment Variables

Kotonoha stores persistent application settings in SQLite. Docker Compose reads the
repository-root `.env` file for image, device, credential, and deployment interpolation.

## Precedence

Application settings use the following precedence, from lowest to highest:

1. Base defaults synchronized from `config/default.yaml`
2. The selected hardware-profile defaults synchronized from `config/profiles/accelerators/`
3. Operator overrides stored in `data/kotonoha.db`
4. `KOTONOHA__SECTION__FIELD` environment variables

The CLI loads `.env` before constructing settings. Existing process variables take
precedence over `.env`. Set `KOTONOHA_ENV_FILE` to an empty value to disable
application-side loading. Containers disable this loader because Compose injects the
resolved environment.

Environment overrides are not persisted. The Web reset action removes SQLite overrides
but cannot remove values supplied by the process environment.

## File Management

Create the deployment environment file from the committed template:

```bash
cp .env.example .env
chmod 600 .env
```

Generate a service token before exposing any model-service port:

```bash
python3 -c 'import secrets; print(secrets.token_hex(32))'
```

`.env` and `data/kotonoha.db` contain deployment state and credentials. Both files are
gitignored and require owner-only permissions.

## Application Variables

| Variable | Purpose | Default |
|---|---|---|
| `KOTONOHA_ENV_FILE` | Selects an alternative environment file. An empty value disables loading. | `.env` |
| `KOTONOHA_SETTINGS_DATABASE` | Selects the SQLite settings, history, and glossary database. | `data/kotonoha.db` |
| `KOTONOHA_INITIAL_HARDWARE_PROFILE` | Seeds a profile when the database has no stored selection. | `nvidia.jetson.agx-orin` |
| `KOTONOHA_LANG` | Selects `auto`, `en`, `ko`, `ja`, or `zh-TW` for CLI and Web text. | `auto` |
| `KOTONOHA_SERVICE_TOKEN` | Protects model service and configuration administration endpoints. | Unset |
| `KOTONOHA_DISABLE_NVML` | Selects the Jetson NVML bypass when the image profile enables it. | Profile-specific |
| `KOTONOHA_MEMORY_ARCHITECTURE` | Overrides detected discrete or unified memory architecture for diagnostics. | Auto-detected |
| `WHISPER_CPP_URL` | Selects the verification ASR fallback endpoint. | `http://127.0.0.1:8082` |

Any typed setting can be overridden with a nested name. Replace dots with double
underscores and use the `KOTONOHA__` prefix.

```dotenv
KOTONOHA__ACCELERATOR__PROFILE=nvidia.rtx.a6000
KOTONOHA__SERVICES__ASR=http://127.0.0.1:8001
KOTONOHA__SERVICES__ASR_VERIFY=http://127.0.0.1:8002
KOTONOHA__SERVICES__LLM=http://127.0.0.1:8003
KOTONOHA__SERVICES__TTS=http://127.0.0.1:8004
KOTONOHA__TRANSPORT__TOKEN=replace-with-the-shared-service-token
```

Use `KOTONOHA__ACCELERATOR__PROFILE` only for an intentional immutable process override.
Use the Web configuration page for persistent profile selection.

## Web Deployment Variables

| Variable | Purpose | Default |
|---|---|---|
| `KOTONOHA_WEB_HOST` | Web listener address. Use `0.0.0.0` only behind an authenticated TLS proxy. | `127.0.0.1` |
| `KOTONOHA_WEB_PORT` | Web listener port. | `8080` |
| `KOTONOHA_WEB_SESSIONS` | Maximum concurrent browser sessions. | `4` |
| `KOTONOHA_EQUIPMENT` | Overrides host detection with `workstation`, `jetson`, or `a6000`. | Auto-detected |
| `MODELS_DIR` | Host model directory mounted at `/models`. | `./models` |

Browsers require HTTPS for microphone access outside localhost. Kotonoha does not
terminate TLS or authenticate Web users. Bind to loopback unless a reverse proxy provides
both controls.

## Accelerator and Container Variables

| Variable | Purpose |
|---|---|
| `ACCELERATOR_PROFILE` | Selects the Docker profile used during deployment bootstrap. |
| `ACCELERATOR_VLLM_IMAGE` | Overrides the profile vLLM image. |
| `ACCELERATOR_OMNI_IMAGE` | Overrides the profile vLLM-Omni image. |
| `ACCELERATOR_SERVICE_BASE` | Overrides the profile base image for server Python services. |
| `CONTAINER_RUNTIME` | Selects the Compose accelerator runtime. |
| `GPU_DRIVER` | Selects the Compose device reservation driver. |
| `ASR_BASE`, `VERIFY_BASE`, `SERVICE_BASE` | Override role-specific build images. |
| `LLM_IMAGE`, `TTS_IMAGE` | Override translation and speech build images. |
| `VLLM_NVML_PATCH` | Controls application of the pinned Jetson vLLM NVML patch. |
| `ASR_GPU_DEVICE`, `ASR_VERIFY_GPU_DEVICE`, `LLM_GPU_DEVICE`, `TTS_GPU_DEVICE` | Bind A6000 roles to GPU indices or stable UUIDs. |
| `ASR_GPU_MEMORY_MIB`, `ASR_VERIFY_GPU_MEMORY_MIB`, `LLM_GPU_MEMORY_MIB`, `TTS_GPU_MEMORY_MIB` | Set allocator reservations per role. |
| `GPU_MEMORY_RESERVE_MIB`, `GPU_NAME_FILTER`, `GPU_ALLOCATION_MODE` | Control automatic A6000 assignment. |

## Runtime Environment

Model paths, memory fractions, context limits, cache policy, and eager execution are
typed application settings. Configure them through the Web page so SQLite remains the
effective persistent source.

| Variable | Purpose |
|---|---|
| `TRANSFORMERS_OFFLINE`, `HF_HUB_OFFLINE` | Disable Hugging Face network access after staging |
| `HF_HOME` | Hugging Face cache root inside a container |

Image defaults and tuning values remain hardware-profile decisions. Do not copy a Jetson
memory fraction, KV cache type, or NVML setting into an A6000 deployment.

## Legacy Variables

`KOTONOHA_CONFIG` and `KOTONOHA_LOCAL_CONFIG` select YAML files for one-time migration.
They do not define active persistence layers. `KOTONOHA_SKIP_LOCAL_CONFIG` remains a test
isolation control.
