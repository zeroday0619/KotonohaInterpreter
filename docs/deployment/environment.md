# Environment Variables

Kotonoha reads application configuration from YAML and environment variables. Docker
Compose also reads the repository-root `.env` file for image, device, and deployment
interpolation.

## Precedence

Application settings use the following precedence, from lowest to highest:

1. `config/default.yaml`
2. The selected accelerator profile
3. `KOTONOHA_CONFIG` or `--config`
4. `KOTONOHA_LOCAL_CONFIG`
5. `KOTONOHA__SECTION__FIELD` environment variables

The CLI loads `.env` before importing Typer or constructing `Settings`. Existing process
environment variables take precedence over `.env`. Set `KOTONOHA_ENV_FILE` to an empty
value to disable application-side loading. Containers disable this loader because Compose
already injects the resolved environment.

Only names beginning with `KOTONOHA_` enter the Python process through the application
loader. Compose reads deployment variables such as `MODELS_DIR` and `LLM_GPU_DEVICE`
directly from `.env`.

## File Management

Create the local file from the committed template:

```bash
cp .env.example .env
chmod 600 .env
```

Generate the service token before an A6000 or network-accessible deployment:

```bash
python3 -c 'import secrets; print(secrets.token_hex(32))'
```

`.env` is gitignored. `.env.example` contains no deployable credential.

## Application Variables

| Variable | Purpose | Default |
|---|---|---|
| `KOTONOHA_ENV_FILE` | Selects an alternative environment file. An empty value disables loading. | `.env` |
| `KOTONOHA_CONFIG` | Selects the primary YAML overlay. | `config/default.yaml` |
| `KOTONOHA_LOCAL_CONFIG` | Selects the writable host override. | `config/local.yaml` |
| `KOTONOHA_LANG` | Selects `auto`, `en`, `ko`, `ja`, or `zh-TW` for CLI and TUI text. | `auto` |
| `KOTONOHA_SERVICE_TOKEN` | Protects model service and configuration administration endpoints. | Unset |
| `KOTONOHA_DISABLE_NVML` | Selects the Jetson NVML bypass when the image profile enables it. | Profile-specific |
| `KOTONOHA_MEMORY_ARCHITECTURE` | Overrides detected discrete or unified memory architecture for diagnostics. | Auto-detected |
| `KOTONOHA_SKIP_LOCAL_CONFIG` | Excludes the default local override. Reserved for tests. | Unset |
| `WHISPER_CPP_URL` | Selects the verification ASR fallback endpoint. | `http://127.0.0.1:8082` |

Any typed setting can be overridden with a nested name. Replace dots with double
underscores and use the `KOTONOHA__` prefix.

```dotenv
KOTONOHA__PERF_MODE=remote
KOTONOHA__REMOTE__ENABLED=true
KOTONOHA__REMOTE__TOKEN=replace-with-the-shared-service-token
KOTONOHA__REMOTE__SERVICES__ASR=http://127.0.0.1:8001
KOTONOHA__REMOTE__SERVICES__ASR_VERIFY=http://127.0.0.1:8002
KOTONOHA__REMOTE__SERVICES__LLM=http://127.0.0.1:8003
KOTONOHA__REMOTE__SERVICES__TTS=http://127.0.0.1:8004
```

The Web configuration editor displays every typed setting. Saving validates and writes
the configured local override atomically. Existing browser sessions reconnect with the
new settings. Changes under `asr`, `asr_verify`, `llm`, `tts`, or `accelerator` invoke the
authenticated `/admin/reload` endpoint on the affected resident service.

## Web Deployment Variables

| Variable | Purpose | Default |
|---|---|---|
| `KOTONOHA_WEB_HOST` | Web listener address. Use `0.0.0.0` only behind an authenticated TLS proxy. | `127.0.0.1` |
| `KOTONOHA_WEB_PORT` | Web listener port. | `8080` |
| `KOTONOHA_WEB_SESSIONS` | Maximum concurrent browser sessions. | `4` |
| `KOTONOHA_WEB_CONFIG` | Primary config path inside the Web container. | `/app/config/default.yaml` |
| `KOTONOHA_WEB_LOCAL_CONFIG` | Writable override path inside the Web container. | `/app/config/local.yaml` |
| `KOTONOHA_EQUIPMENT` | Overrides host detection with `workstation`, `jetson`, or `a6000`. | Auto-detected |
| `MODELS_DIR` | Host model directory mounted at `/models`. | `./models` |

Remote browsers require HTTPS for microphone access. Kotonoha does not terminate TLS or
authenticate Web UI users. Bind to loopback unless a reverse proxy provides both controls.

## Accelerator and Container Variables

| Variable | Purpose |
|---|---|
| `ACCELERATOR_PROFILE` | Selects `<vendor>.<family>.<model>` runtime tuning. |
| `ACCELERATOR_VLLM_IMAGE` | Overrides the profile vLLM image. |
| `ACCELERATOR_OMNI_IMAGE` | Overrides the profile vLLM-Omni image. |
| `ACCELERATOR_REMOTE_BASE` | Overrides the profile base image for remote Python services. |
| `CONTAINER_RUNTIME` | Selects the Compose accelerator runtime. |
| `GPU_DRIVER` | Selects the Compose device reservation driver. |
| `ASR_BASE`, `VERIFY_BASE`, `ORCH_BASE`, `REMOTE_BASE`, `REMOTE_ASR_BASE` | Override role-specific build images. |
| `LLM_IMAGE`, `TTS_IMAGE` | Override translation and speech build images. |
| `VLLM_NVML_PATCH` | Controls application of the pinned Jetson vLLM NVML patch. |
| `ASR_GPU_DEVICE`, `ASR_VERIFY_GPU_DEVICE`, `LLM_GPU_DEVICE`, `TTS_GPU_DEVICE` | Bind A6000 roles to GPU indices or stable UUIDs. |
| `ASR_GPU_MEMORY_MIB`, `ASR_VERIFY_GPU_MEMORY_MIB`, `LLM_GPU_MEMORY_MIB`, `TTS_GPU_MEMORY_MIB` | Set allocator reservations per role. |
| `GPU_MEMORY_RESERVE_MIB`, `GPU_NAME_FILTER`, `GPU_ALLOCATION_MODE` | Control automatic A6000 placement. |

## Runtime Tuning Variables

| Variable | Setting |
|---|---|
| `LLM_MAX_MODEL_LEN` | `llm.max_model_len` |
| `LLM_MAX_NUM_BATCHED_TOKENS` | `llm.max_num_batched_tokens` |
| `LLM_GPU_MEMORY_UTILIZATION` | `llm.gpu_memory_utilization` |
| `LLM_KV_CACHE_DTYPE` | `llm.kv_cache_dtype` |
| `LLM_ENABLE_PREFIX_CACHING` | `llm.enable_prefix_caching` |
| `LLM_COMPILATION_MODE` | `llm.compilation_mode` |
| `TTS_GPU_MEMORY_UTILIZATION` | vLLM-Omni memory fraction |
| `TTS_ENFORCE_EAGER` | Disables speech CUDA graph capture when set to `1` |
| `TTS_MODEL` | TTS snapshot path inside the container |
| `TTS_DEPLOY_CONFIG` | vLLM-Omni stage configuration path |
| `TTS_SERVED_MODEL_NAME` | Speech API model name |
| `TTS_STARTUP_TIMEOUT_SECONDS` | Speech engine startup timeout |
| `PROMETHEUS_PORT` | Optional headless metrics receiver port; the Web `/metrics` endpoint does not require it |
| `TRANSFORMERS_OFFLINE`, `HF_HUB_OFFLINE` | Disable Hugging Face network access after staging |
| `HF_HOME` | Hugging Face cache root inside a container |

Image defaults and tuning values remain accelerator-profile decisions. Do not copy a
Jetson memory fraction, KV cache type, or NVML setting into an A6000 deployment.
