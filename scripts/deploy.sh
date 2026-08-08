#!/usr/bin/env bash
# Deploy the resident model services on either supported inference host.
#
# The script validates host identity and required artifacts before Compose changes state.
# Existing host overrides and environment files are never replaced.
set -euo pipefail

repository_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repository_root"

deployment_target=""
operation="deploy"
uninstall_requested=false
environment_file="$repository_root/.env"
gpu_environment_file="$repository_root/config/remote-gpu.env"
health_timeout_seconds=600
build_images=true
prepare_jetson_power=true
remove_images=false
reallocate_gpus=false
prepare_only=false
a6000_vllm_image="nvcr.io/nvidia/vllm:26.07-py3"
vllm_omni_image="vllm/vllm-omni:v0.26.0"
docker_profile_file=""
docker_display_command="docker"
docker_requires_sudo=false
docker_environment_names=(
  ASR_BASE
  ASR_GPU_DEVICE
  ASR_VERIFY_GPU_DEVICE
  ACCELERATOR_OMNI_IMAGE
  ACCELERATOR_DEVICE_ENV
  ACCELERATOR_PROFILE
  ACCELERATOR_REMOTE_BASE
  ACCELERATOR_VLLM_IMAGE
  CONTAINER_RUNTIME
  GPU_DRIVER
  KOTONOHA_DISABLE_NVML
  VLLM_NVML_PATCH
  LLM_GPU_DEVICE
  LLM_GPU_MEMORY_UTILIZATION
  LLM_KV_CACHE_DTYPE
  LLM_ENABLE_PREFIX_CACHING
  LLM_MAX_NUM_BATCHED_TOKENS
  LLM_COMPILATION_MODE
  LLM_IMAGE
  LLM_MAX_MODEL_LEN
  MODELS_DIR
  ORCH_BASE
  PROMETHEUS_PORT
  REMOTE_ASR_BASE
  REMOTE_BASE
  TRANSFORMERS_OFFLINE
  TTS_ENFORCE_EAGER
  TTS_GPU_DEVICE
  TTS_GPU_MEMORY_UTILIZATION
  TTS_IMAGE
  VERIFY_BASE
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/deploy.sh jetson [options]
  bash scripts/deploy.sh a6000 [options]
  bash scripts/deploy.sh uninstall jetson [--remove-images]
  bash scripts/deploy.sh uninstall a6000 [--remove-images]

Options:
  --env-file PATH       Compose environment file for A6000 deployment (default: .env)
  --health-timeout SEC  Maximum model startup wait in seconds (default: 600)
  --no-build            Start existing images without building
  --skip-power-setup    Do not set Jetson MAXN mode or lock clocks
  --prepare-only        Validate the host and create configuration without deployment
  --reallocate-gpus     Recompute A6000 role assignments from current free GPU memory
  --remove-images       Also remove project-built images during uninstall
  -h, --help            Show this help

The script does not start the interactive orchestrator. It prints the runtime command
after the resident model services pass their health checks.

Uninstall removes project containers and networks. Models, configuration, secrets,
logs, and SQLite data are always preserved.
EOF
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

require_file() {
  [ -s "$1" ] || fail "required artifact not found or empty: $1"
}

require_directory() {
  [ -d "$1" ] || fail "required artifact directory not found: $1"
}

select_docker_profile() {
  case "$deployment_target" in
    jetson)
      docker_profile_file="$repository_root/docker/profiles/accelerators/nvidia/jetson/agx-orin.env"
      ;;
    a6000)
      docker_profile_file="$repository_root/docker/profiles/accelerators/nvidia/rtx/a6000.env"
      ;;
  esac
  [ -f "$docker_profile_file" ] \
    || fail "Docker accelerator profile not found: $docker_profile_file"
}

load_docker_profile() {
  select_docker_profile
  local variable_name
  local variable_value
  while IFS='=' read -r variable_name variable_value; do
    case "$variable_name" in
      ''|'#'*) continue ;;
    esac
    if [ "${!variable_name+x}" != x ]; then
      export "$variable_name=$variable_value"
    fi
  done < "$docker_profile_file"
  a6000_vllm_image="${ACCELERATOR_VLLM_IMAGE:-$a6000_vllm_image}"
  vllm_omni_image="${ACCELERATOR_OMNI_IMAGE:-$vllm_omni_image}"
  printf 'Docker accelerator profile: %s\n' "${ACCELERATOR_PROFILE:-unknown}"
  printf 'Docker runtime: %s\n' "${CONTAINER_RUNTIME:-unknown}"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    jetson|a6000)
      [ -z "$deployment_target" ] || fail "deployment target specified more than once"
      deployment_target=$1
      shift
      ;;
    uninstall)
      [ "$uninstall_requested" = false ] || fail "uninstall specified more than once"
      operation="uninstall"
      uninstall_requested=true
      shift
      ;;
    --env-file)
      [ "$#" -ge 2 ] || fail "--env-file requires a path"
      environment_file=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
      shift 2
      ;;
    --health-timeout)
      [ "$#" -ge 2 ] || fail "--health-timeout requires a number"
      health_timeout_seconds=$2
      shift 2
      ;;
    --no-build)
      build_images=false
      shift
      ;;
    --skip-power-setup)
      prepare_jetson_power=false
      shift
      ;;
    --prepare-only)
      prepare_only=true
      prepare_jetson_power=false
      shift
      ;;
    --reallocate-gpus)
      reallocate_gpus=true
      shift
      ;;
    --remove-images)
      remove_images=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

[ -n "$deployment_target" ] || {
  usage >&2
  exit 2
}

case "$health_timeout_seconds" in
  ''|*[!0-9]*) fail "--health-timeout must be a positive integer" ;;
esac
[ "$health_timeout_seconds" -gt 0 ] || fail "--health-timeout must be greater than zero"
[ "$remove_images" = false ] || [ "$operation" = "uninstall" ] \
  || fail "--remove-images is valid only with uninstall"
[ "$prepare_only" = false ] || [ "$operation" = "deploy" ] \
  || fail "--prepare-only is valid only with deployment"
[ "$prepare_only" = false ] || [ "$reallocate_gpus" = false ] \
  || fail "--reallocate-gpus cannot be combined with --prepare-only"
[ "$reallocate_gpus" = false ] || { [ "$operation" = "deploy" ] \
  && [ "$deployment_target" = "a6000" ]; } \
  || fail "--reallocate-gpus is valid only with A6000 deployment"

load_docker_profile

require_command docker

run_docker() {
  if [ "$docker_requires_sudo" = false ]; then
    docker "$@"
    return
  fi

  local docker_environment=()
  local variable_name
  for variable_name in "${docker_environment_names[@]}"; do
    if [ "${!variable_name+x}" = x ]; then
      docker_environment+=("$variable_name=${!variable_name}")
    fi
  done
  sudo env "${docker_environment[@]}" docker "$@"
}

configure_docker_access() {
  if docker info >/dev/null 2>&1; then
    docker_requires_sudo=false
    docker_display_command="docker"
  else
    require_command sudo
    sudo docker info >/dev/null 2>&1 \
      || fail "Docker daemon is not available through docker or sudo docker"
    docker_requires_sudo=true
    docker_display_command="sudo docker"
    printf 'Docker requires elevated access; using sudo docker.\n'
  fi
  run_docker compose version >/dev/null 2>&1 \
    || fail "Docker Compose plugin is not available"
}

configure_docker_access

if [ "$operation" = "deploy" ]; then
  require_command curl
  require_command python3
fi

ensure_override() {
  local example_path=$1
  local target_path=$2
  if [ -e "$target_path" ]; then
    printf 'Using existing configuration: %s\n' "$target_path"
    return
  fi
  cp "$example_path" "$target_path"
  chmod 600 "$target_path"
  printf 'Created configuration: %s\n' "$target_path"
}

ensure_remote_environment() {
  if [ -e "$environment_file" ]; then
    chmod 600 "$environment_file"
    local environment_token
    local configured_asr_image
    local configured_asr_memory_mib
    local configured_llm_image
    local configured_tts_image
    environment_token=$(sed -n 's/^KOTONOHA_SERVICE_TOKEN=//p' "$environment_file" | tail -n 1)
    [ -n "$environment_token" ] \
      || fail "environment file does not define KOTONOHA_SERVICE_TOKEN: $environment_file"
    [ "${#environment_token}" -ge 32 ] \
      || fail "KOTONOHA_SERVICE_TOKEN must contain at least 32 characters"
    case "$environment_token" in
      *'<'*|*'>'*) fail "replace the token placeholder in $environment_file" ;;
    esac
    configured_asr_image=$(sed -n 's/^REMOTE_ASR_BASE=//p' "$environment_file" | tail -n 1)
    configured_asr_memory_mib=$(sed -n 's/^ASR_GPU_MEMORY_MIB=//p' "$environment_file" | tail -n 1)
    configured_llm_image=$(sed -n 's/^LLM_IMAGE=//p' "$environment_file" | tail -n 1)
    configured_tts_image=$(sed -n 's/^TTS_IMAGE=//p' "$environment_file" | tail -n 1)
    [ -z "$configured_asr_image" ] || [ "$configured_asr_image" = "$a6000_vllm_image" ] \
      || fail "environment file must set REMOTE_ASR_BASE=$a6000_vllm_image"
    if [ -n "$configured_asr_memory_mib" ]; then
      case "$configured_asr_memory_mib" in
        *[!0-9]*) fail "environment file ASR_GPU_MEMORY_MIB must be an integer" ;;
      esac
      [ "$configured_asr_memory_mib" -ge 14336 ] \
        || fail "environment file must set ASR_GPU_MEMORY_MIB to at least 14336"
    fi
    [ -z "$configured_llm_image" ] || [ "$configured_llm_image" = "$a6000_vllm_image" ] \
      || fail "environment file must set LLM_IMAGE=$a6000_vllm_image"
    [ -z "$configured_tts_image" ] || [ "$configured_tts_image" = "$vllm_omni_image" ] \
      || fail "environment file must set TTS_IMAGE=$vllm_omni_image"
    if [ -n "${KOTONOHA_SERVICE_TOKEN:-}" ] \
      && [ "$KOTONOHA_SERVICE_TOKEN" != "$environment_token" ]; then
      fail "shell and environment-file service tokens differ"
    fi
    printf 'Using existing environment file: %s\n' "$environment_file"
    return
  fi

  require_command openssl
  local service_token=${KOTONOHA_SERVICE_TOKEN:-}
  if [ -z "$service_token" ]; then
    service_token=$(openssl rand -hex 32)
  fi
  [ "${#service_token}" -ge 32 ] \
    || fail "KOTONOHA_SERVICE_TOKEN must contain at least 32 characters"

  umask 077
  {
    printf 'KOTONOHA_SERVICE_TOKEN=%s\n' "$service_token"
    printf 'MODELS_DIR=../models\n'
    printf 'REMOTE_BASE=%s\n' "${ACCELERATOR_REMOTE_BASE:-pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime}"
    printf 'REMOTE_ASR_BASE=%s\n' "$a6000_vllm_image"
    printf 'TTS_IMAGE=%s\n' "$vllm_omni_image"
    printf 'LLM_IMAGE=%s\n' "$a6000_vllm_image"
    printf 'LLM_MAX_MODEL_LEN=2048\n'
    printf 'LLM_GPU_MEMORY_UTILIZATION=0.90\n'
    printf 'LLM_MAX_NUM_BATCHED_TOKENS=4096\n'
    printf 'LLM_ENABLE_PREFIX_CACHING=1\n'
    printf 'LLM_COMPILATION_MODE=2\n'
    printf 'GPU_ALLOCATION_MODE=auto\n'
    printf 'GPU_NAME_FILTER=A6000\n'
    printf 'GPU_MEMORY_RESERVE_MIB=1024\n'
    printf 'LLM_GPU_MEMORY_MIB=43008\n'
    printf 'ASR_GPU_MEMORY_MIB=14336\n'
    printf 'ASR_VERIFY_GPU_MEMORY_MIB=6144\n'
    printf 'TTS_GPU_MEMORY_MIB=3072\n'
    printf 'TTS_GPU_MEMORY_UTILIZATION=0.25\n'
    printf 'TTS_ENFORCE_EAGER=1\n'
    printf 'TRANSFORMERS_OFFLINE=1\n'
  } >"$environment_file"
  printf 'Created protected environment file: %s\n' "$environment_file"
  printf 'Copy its KOTONOHA_SERVICE_TOKEN value to the Jetson through a secure channel.\n'
}

check_common_speech_models() {
  local models_path=$1
  require_directory "$models_path/faster-whisper-large-v3"
  require_file "$models_path/faster-whisper-large-v3/config.json"
  require_directory "$models_path/Qwen3-TTS-0.6B"
  require_file "$models_path/Qwen3-TTS-0.6B/config.json"
}

check_jetson_speech_models() {
  local models_path=$1
  check_common_speech_models "$models_path"
  require_directory "$models_path/Qwen3-ASR-0.6B"
  require_file "$models_path/Qwen3-ASR-0.6B/config.json"
  require_directory "$models_path/Qwen3-ASR-0.6B-hf"
  require_file "$models_path/Qwen3-ASR-0.6B-hf/config.json"
}

check_a6000_speech_models() {
  local models_path=$1
  check_common_speech_models "$models_path"
  require_directory "$models_path/Voxtral-Mini-4B-Realtime-2602"
  require_file "$models_path/Voxtral-Mini-4B-Realtime-2602/config.json"
}

remote_models_path() {
  local configured_path=${MODELS_DIR:-}
  if [ -z "$configured_path" ]; then
    configured_path=$(sed -n 's/^MODELS_DIR=//p' "$environment_file" | tail -n 1)
  fi
  [ -n "$configured_path" ] || configured_path=../models
  case "$configured_path" in
    /*) printf '%s\n' "$configured_path" ;;
    *) printf '%s\n' "$repository_root/docker/$configured_path" ;;
  esac
}

run_privileged() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    require_command sudo
    sudo "$@"
  fi
}

check_nvidia_container_runtime() {
  run_docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"' \
    || fail "NVIDIA Container Toolkit is not configured for Docker; run nvidia-ctk runtime configure --runtime=docker and restart Docker"
}

check_jetson_host() {
  [ "$(uname -m)" = "aarch64" ] || fail "Jetson deployment requires aarch64"
  [ -r /etc/nv_tegra_release ] || fail "Jetson Linux release file is missing"
  grep -Eq 'R39.*REVISION: 2' /etc/nv_tegra_release \
    || fail "Jetson deployment requires Jetson Linux 39.2"
  [ -d /dev/snd ] || fail "audio device directory is missing: /dev/snd"
  require_file "$repository_root/models/silero_vad.onnx"
  require_file "$repository_root/models/llm/translategemma-4b-it/config.json"
  check_nvidia_container_runtime

  require_command nvpmodel
  require_command jetson_clocks
  if [ "$prepare_jetson_power" = true ]; then
    run_privileged nvpmodel -m 0
    run_privileged jetson_clocks
  fi
  nvpmodel -q || fail "unable to read the Jetson power mode"
  # jetson_clocks refuses to run as a non-root user even for --show, so the
  # readback needs the same elevation as the commands that set the state. Without
  # it the deployment stops here whenever the script itself is not run under sudo,
  # even though Docker already elevates on its own.
  run_privileged jetson_clocks --show || fail "unable to read the Jetson clock state"
}

check_a6000_host() {
  local models_path=$1
  case "$(uname -m)" in
    x86_64|amd64) ;;
    *) fail "A6000 deployment requires an x86_64 host" ;;
  esac
  require_command nvidia-smi
  nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi cannot access the A6000"
  check_nvidia_container_runtime
  require_file "$models_path/llm/translategemma-12b-it/config.json"
}

load_gpu_environment() {
  require_file "$gpu_environment_file"
  while IFS='=' read -r key value; do
    case "$key" in
      ASR_GPU_DEVICE|ASR_VERIFY_GPU_DEVICE|LLM_GPU_DEVICE|TTS_GPU_DEVICE)
        [ -n "$value" ] || fail "empty GPU assignment in $gpu_environment_file: $key"
        export "$key=$value"
        ;;
    esac
  done < "$gpu_environment_file"
  [ -n "${ASR_GPU_DEVICE:-}" ] || fail "GPU allocation did not assign ASR"
  [ -n "${ASR_VERIFY_GPU_DEVICE:-}" ] || fail "GPU allocation did not assign ASR verification"
  [ -n "${LLM_GPU_DEVICE:-}" ] || fail "GPU allocation did not assign LLM"
  [ -n "${TTS_GPU_DEVICE:-}" ] || fail "GPU allocation did not assign TTS"
}

allocate_a6000_gpus() {
  local arguments=(
    python3 "$repository_root/scripts/py/allocate_gpus.py"
    --environment-file "$environment_file"
    --output "$gpu_environment_file"
  )
  if [ "$reallocate_gpus" = true ]; then
    arguments+=(--force)
  fi
  "${arguments[@]}" || fail "GPU allocation failed"
  load_gpu_environment
}

python_service_ready() {
  curl --connect-timeout 2 --max-time 5 -fsS "$1/health" 2>/dev/null \
    | python3 -c 'import json, sys; raise SystemExit(json.load(sys.stdin).get("ok") is not True)'
}

http_service_ready() {
  curl --connect-timeout 2 --max-time 5 -fsS "$1/health" >/dev/null 2>&1
}

wait_for_service() {
  local service_name=$1
  local service_url=$2
  local health_kind=$3
  local started_at
  local current_time
  local elapsed_seconds
  started_at=$(date +%s)
  printf 'Waiting for %s at %s' "$service_name" "$service_url"

  while true; do
    if [ "$health_kind" = "python" ]; then
      if python_service_ready "$service_url"; then
        printf ' ready\n'
        return
      fi
    elif http_service_ready "$service_url"; then
      printf ' ready\n'
      return
    fi

    current_time=$(date +%s)
    elapsed_seconds=$((current_time - started_at))
    if [ "$elapsed_seconds" -ge "$health_timeout_seconds" ]; then
      printf ' timeout\n'
      return 1
    fi
    printf '.'
    sleep 5
  done
}

show_failed_health() {
  local compose_file=$1
  local compose_environment_file=$2
  shift 2
  printf '\nService startup failed. Recent logs:\n' >&2
  if [ -n "$compose_environment_file" ]; then
    run_docker compose --env-file "$compose_environment_file" -f "$compose_file" \
      logs --tail=120 "$@" >&2 || true
  else
    run_docker compose -f "$compose_file" logs --tail=120 "$@" >&2 || true
  fi
  exit 1
}

start_jetson_service() {
  local compose_file=$1
  local service_name=$2
  local service_url=$3
  local health_kind=$4
  if [ "$build_images" = false ]; then
    run_docker compose -f "$compose_file" up -d --no-build "$service_name" || return 1
  else
    run_docker compose -f "$compose_file" up -d "$service_name" || return 1
  fi
  wait_for_service "$service_name" "$service_url" "$health_kind"
}

# The vLLM probes end with os._exit(0).
#
# Loading vLLM registers torch.library custom operators. At interpreter shutdown
# their weakref finalizers run against already-released C++ state, and
# torch._C._jit_get_operation reads a corrupted operator name:
#
#   File "weakref.py", in _exitfunc -> torch/library.py, in _del_library
#   UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf7
#
# CPython reports that as an ignored exception and still exits 0, so the check
# passes while printing a traceback that is indistinguishable from a failure in a
# deployment log. These are one-shot probes whose only output is the banner above,
# so terminating before finalization removes the noise without hiding anything.
# The assertions run first; a real failure still raises and exits non-zero.
verify_vllm_cuda_runtime() {
  local compose_file=$1
  local compose_environment_file=$2
  local service_name=$3
  local compose_arguments=(compose)
  if [ -n "$compose_environment_file" ]; then
    compose_arguments+=(--env-file "$compose_environment_file")
  fi
  compose_arguments+=(-f "$compose_file")

  run_docker "${compose_arguments[@]}" run --rm --no-deps \
    --entrypoint python3 "$service_name" -c \
    'import torch, vllm; assert torch.cuda.is_available(), "CUDA is unavailable in the vLLM container"; device_count = torch._C._cuda_getDeviceCount(); assert device_count > 0, "CUDA reported no devices"; print("CUDA", torch.version.cuda, "| GPUs", device_count, "| GPU", torch.cuda.get_device_name(0), "| vLLM", vllm.__version__, "| NVML", "disabled" if __import__("os").environ.get("KOTONOHA_DISABLE_NVML") == "1" else "enabled"); import os as _os, sys as _sys; _sys.stdout.flush(); _sys.stderr.flush(); _os._exit(0)' \
    || fail "$service_name container cannot initialize the CUDA runtime and vLLM"
}

verify_vllm_translation_runtime() {
  local compose_file=$1
  local compose_environment_file=$2
  local compose_arguments=(compose)
  if [ -n "$compose_environment_file" ]; then
    compose_arguments+=(--env-file "$compose_environment_file")
  fi
  compose_arguments+=(-f "$compose_file")

  run_docker "${compose_arguments[@]}" run --rm --no-deps \
    --entrypoint /opt/kotonoha-venv/bin/python llm -c \
    'import sys; from pathlib import Path; import numpy, torch, transformers, vllm; from transformers import GenerationMixin; from vllm.engine.arg_utils import AsyncEngineArgs; from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args; from kotonoha._config import load_settings; from kotonoha.services._llm_server import _engine_arguments; assert torch.cuda.is_available(), "CUDA is unavailable in the translation container"; device_count = torch._C._cuda_getDeviceCount(); assert device_count > 0, "CUDA reported no devices"; assert Path(sys.executable).is_relative_to("/opt/kotonoha-venv"), sys.executable; engine_config = AsyncEngineArgs(**_engine_arguments(load_settings().llm)).create_engine_config(); model_config = engine_config.model_config; text_config = model_config.hf_text_config; rope_parameters = getattr(text_config, "rope_parameters", None); nested_keys = {"full_attention", "sliding_attention"}; assert not isinstance(rope_parameters, dict) or not nested_keys.intersection(rope_parameters) or nested_keys <= rope_parameters.keys() and all(isinstance(value, dict) and "rope_type" in value for value in rope_parameters.values()), rope_parameters; print("CUDA", torch.version.cuda, "| GPUs", device_count, "| GPU", torch.cuda.get_device_name(0), "| vLLM", vllm.__version__, "| Transformers", transformers.__version__, "| NumPy", numpy.__version__, "| model", model_config.model); import os as _os, sys as _sys; _sys.stdout.flush(); _sys.stderr.flush(); _os._exit(0)' \
    || fail "llm container cannot validate the in-process vLLM model configuration"
}

# The configuration probes report every mismatch at once. Each run costs a
# container start on top of an image build, so exiting on the first stale key made
# an operator repeat the cycle once per key in config/local.yaml.
verify_jetson_asr_configuration() {
  local compose_file=$1

  run_docker compose -f "$compose_file" run --rm --no-deps \
    --entrypoint /opt/kotonoha-venv/bin/python asr -c \
    'import sys; from pathlib import Path; from kotonoha._config import load_settings; config = load_settings().asr; checks = [("asr.vllm_model_id", "/models/Qwen3-ASR-0.6B", str(Path(config.vllm_model_id))), ("asr.model_id", "/models/Qwen3-ASR-0.6B-hf", str(Path(config.model_id))), ("asr.vllm_realtime_architecture", "qwen3_asr", config.vllm_realtime_architecture)]; stale = [f"  {key}: effective {actual}, expected {expected}" for key, expected, actual in checks if actual != expected]; missing = [str(Path(path) / "config.json") for path in (config.vllm_model_id, config.model_id) if not (Path(path) / "config.json").is_file()]; stale and print("Stale asr overrides in config/local.yaml:", *stale, sep="\n"); missing and print("Offline ASR model files are missing:", *[f"  {item}" for item in missing], sep="\n"); (stale or missing) and sys.exit(1); print("Effective Jetson ASR models:", config.vllm_model_id, "| fallback", config.model_id)' \
    || fail "Jetson ASR configuration does not select the mounted Qwen3-ASR 0.6B models"
}

verify_jetson_asr_verification_runtime() {
  local compose_file=$1

  run_docker compose -f "$compose_file" run --rm --no-deps \
    --entrypoint /opt/kotonoha-venv/bin/python asr-verify -c \
    'import sys; from pathlib import Path; import ctranslate2; from kotonoha._config import load_settings; config = load_settings().asr_verify; checks = [("asr_verify.device", "cpu", config.device), ("asr_verify.compute_type", "int8", config.compute_type)]; stale = [f"  {key}: effective {actual}, expected {expected}" for key, expected, actual in checks if actual != expected]; model_config = Path(config.model_id) / "config.json"; missing = [] if model_config.is_file() else [str(model_config)]; supported = sorted(ctranslate2.get_supported_compute_types("cpu")); unsupported = [] if config.compute_type in supported else [f"  asr_verify.compute_type {config.compute_type} is not in {supported}"]; stale and print("Stale asr_verify overrides in config/local.yaml:", *stale, sep="\n"); missing and print("Offline verification ASR model is missing:", *[f"  {item}" for item in missing], sep="\n"); unsupported and print("CTranslate2 cannot use the configured compute type:", *unsupported, sep="\n"); (stale or missing or unsupported) and sys.exit(1); print("Effective Jetson verification ASR:", config.device, config.compute_type, "| CTranslate2", ctranslate2.__version__)' \
    || fail "Jetson verification ASR cannot use the configured CTranslate2 runtime"
}

verify_jetson_memory_budget() {
  local compose_file=$1
  local maximum_utilization=0.85

  run_docker compose -f "$compose_file" run --rm --no-deps \
    --entrypoint /opt/kotonoha-venv/bin/python asr -c \
    'import sys; from pathlib import Path; import yaml; from kotonoha._config import load_settings; settings = load_settings(); tts_profile = yaml.safe_load(Path("/app/docker/tts/qwen3_tts_jetson.yaml").read_text()); tts_utilization = max(float(stage["gpu_memory_utilization"]) for stage in tts_profile["stages"]); values = {"asr": settings.asr.vllm_gpu_memory_utilization, "llm": settings.llm.gpu_memory_utilization, "tts": tts_utilization}; total = sum(values.values()); print("Jetson resident memory budget:", " + ".join(f"{role}={value:.2f}" for role, value in values.items()), f"= {total:.2f}"); total <= float(sys.argv[1]) or sys.exit(f"resident vLLM memory budget {total:.2f} exceeds limit {float(sys.argv[1]):.2f}; lower a Jetson profile budget before starting services")' \
    "$maximum_utilization" \
    || fail "Jetson resident services exceed the configured unified-memory budget"
}

verify_a6000_asr_configuration() {
  local compose_file=$1
  local compose_environment_file=$2
  local minimum_gpu_memory_utilization=0.28
  local compose_arguments=(compose)
  if [ -n "$compose_environment_file" ]; then
    compose_arguments+=(--env-file "$compose_environment_file")
  fi
  compose_arguments+=(-f "$compose_file")

  run_docker "${compose_arguments[@]}" run --rm --no-deps \
    --entrypoint python3 asr -c \
    'import sys; from kotonoha._config import load_settings; value = load_settings().asr.vllm_gpu_memory_utilization; minimum = float(sys.argv[1]); value >= minimum or sys.exit(f"effective asr.vllm_gpu_memory_utilization is {value}; expected at least {minimum}. Remove KOTONOHA__ASR__VLLM_GPU_MEMORY_UTILIZATION or update config/remote-server.local.yaml."); print(f"Effective A6000 ASR GPU memory utilization: {value:.2f}")' \
    "$minimum_gpu_memory_utilization" \
    || fail "A6000 ASR configuration cannot provide the measured Voxtral KV-cache budget"
}

verify_vllm_omni_cuda_runtime() {
  local compose_file=$1
  local compose_environment_file=$2
  local compose_arguments=(compose)
  if [ -n "$compose_environment_file" ]; then
    compose_arguments+=(--env-file "$compose_environment_file")
  fi
  compose_arguments+=(-f "$compose_file")

  run_docker "${compose_arguments[@]}" run --rm --no-deps \
    --entrypoint python3 tts -c \
    'from importlib.metadata import version; import torch, vllm, vllm_omni; assert torch.cuda.is_available(), "CUDA is unavailable in the vLLM-Omni container"; device_count = torch._C._cuda_getDeviceCount(); assert device_count > 0, "CUDA reported no devices"; print("CUDA", torch.version.cuda, "| GPUs", device_count, "| GPU", torch.cuda.get_device_name(0), "| vLLM", vllm.__version__, "| vLLM-Omni", version("vllm-omni"), "| NVML", "disabled" if __import__("os").environ.get("KOTONOHA_DISABLE_NVML") == "1" else "enabled"); import os as _os, sys as _sys; _sys.stdout.flush(); _sys.stderr.flush(); _os._exit(0)' \
    || fail "tts container cannot initialize the CUDA runtime and vLLM-Omni"
}

deploy_jetson() {
  local compose_file="$repository_root/docker/compose.yaml"
  printf 'Deploying Jetson model services from %s\n' "$repository_root"
  check_jetson_speech_models "$repository_root/models"
  check_jetson_host
  ensure_override \
    "$repository_root/config/jetson.local.example.yaml" \
    "$repository_root/config/local.yaml"

  run_docker compose -f "$compose_file" config --quiet
  if [ "$prepare_only" = true ]; then
    printf 'Jetson host configuration is prepared. No images or services were changed.\n'
    return
  fi
  if [ "$build_images" = true ]; then
    run_docker compose -f "$compose_file" build asr asr-verify llm tts orchestrator
  fi
  verify_jetson_asr_configuration "$compose_file"
  verify_jetson_asr_verification_runtime "$compose_file"
  verify_jetson_memory_budget "$compose_file"
  verify_vllm_cuda_runtime "$compose_file" "" asr
  verify_vllm_translation_runtime "$compose_file" ""
  verify_vllm_omni_cuda_runtime "$compose_file" ""
  if ! start_jetson_service "$compose_file" asr http://127.0.0.1:8001 python \
    || ! start_jetson_service "$compose_file" asr-verify http://127.0.0.1:8002 python \
    || ! start_jetson_service "$compose_file" llm http://127.0.0.1:8003 http \
    || ! start_jetson_service "$compose_file" tts http://127.0.0.1:8004 python; then
    show_failed_health "$compose_file" "" asr asr-verify llm tts
  fi

  printf '\nJetson model services are ready. Start the interactive interpreter with:\n'
  printf '%s compose -f docker/compose.yaml run --rm orchestrator\n' "$docker_display_command"
}

deploy_a6000() {
  local compose_file="$repository_root/docker/compose.remote.yaml"
  local prometheus_port="${PROMETHEUS_PORT:-9091}"
  local compose_command=(
    run_docker compose --env-file "$environment_file" -f "$compose_file"
  )
  local models_path
  printf 'Deploying A6000 model services from %s\n' "$repository_root"
  ensure_remote_environment
  models_path=$(remote_models_path)
  check_a6000_speech_models "$models_path"
  check_a6000_host "$models_path"
  if [ "$reallocate_gpus" = true ]; then
    printf 'Stopping resident services before measuring free GPU memory.\n'
    "${compose_command[@]}" stop asr asr-verify llm tts || true
  fi
  allocate_a6000_gpus
  ensure_override \
    "$repository_root/config/remote-server.local.example.yaml" \
    "$repository_root/config/remote-server.local.yaml"

  "${compose_command[@]}" config --quiet
  if [ "$prepare_only" = true ]; then
    printf 'A6000 host configuration is prepared. No images or services were changed.\n'
    return
  fi
  if [ "$build_images" = true ]; then
    "${compose_command[@]}" build metrics asr asr-verify llm tts
  fi
  verify_a6000_asr_configuration "$compose_file" "$environment_file"
  verify_vllm_cuda_runtime "$compose_file" "$environment_file" asr
  verify_vllm_translation_runtime "$compose_file" "$environment_file"
  verify_vllm_omni_cuda_runtime "$compose_file" "$environment_file"
  if [ "$build_images" = false ]; then
    "${compose_command[@]}" up -d --no-build metrics asr asr-verify llm tts
  else
    "${compose_command[@]}" up -d metrics asr asr-verify llm tts
  fi

  if ! wait_for_service metrics "http://127.0.0.1:$prometheus_port" python \
    || ! wait_for_service asr http://127.0.0.1:8001 python \
    || ! wait_for_service asr-verify http://127.0.0.1:8002 python \
    || ! wait_for_service llm http://127.0.0.1:8003 http \
    || ! wait_for_service tts http://127.0.0.1:8004 python; then
    show_failed_health "$compose_file" "$environment_file" metrics asr asr-verify llm tts
  fi

  printf '\nA6000 model and metrics services are ready. Configure the Jetson with the token in:\n'
  printf '%s\n' "$environment_file"
}

remove_project_images() {
  local image_name
  local project_images=()
  while IFS= read -r image_name; do
    case "$image_name" in
      kotonohainterpreter-*:*'<none>'*) ;;
      kotonohainterpreter-*:*) project_images+=("$image_name") ;;
    esac
  done < <(run_docker image ls --format '{{.Repository}}:{{.Tag}}')

  if [ "${#project_images[@]}" -eq 0 ]; then
    printf 'No Kotonoha project images are present.\n'
    return
  fi
  run_docker image rm "${project_images[@]}"
}

uninstall_jetson() {
  local compose_file="$repository_root/docker/compose.yaml"
  printf 'Removing Jetson project containers and network.\n'
  run_docker compose -f "$compose_file" down --remove-orphans

  if [ "$remove_images" = true ]; then
    remove_project_images
  fi

  printf 'Preserved: config/local.yaml, models/, data/, and upstream base images.\n'
}

uninstall_a6000() {
  local compose_file="$repository_root/docker/compose.remote.yaml"
  local environment_arguments=()
  local generated_environment_file=""
  if [ -e "$environment_file" ]; then
    environment_arguments=(--env-file "$environment_file")
  else
    # Compose requires token interpolation even though down does not start a service.
    generated_environment_file=$(mktemp "${TMPDIR:-/tmp}/kotonoha-uninstall.XXXXXX")
    chmod 600 "$generated_environment_file"
    printf 'KOTONOHA_SERVICE_TOKEN=uninstall-only\n' >"$generated_environment_file"
    environment_arguments=(--env-file "$generated_environment_file")
  fi
  local compose_command=(
    run_docker compose "${environment_arguments[@]}" -f "$compose_file"
  )

  printf 'Removing A6000 project containers and network.\n'
  local compose_status=0
  "${compose_command[@]}" down --remove-orphans || compose_status=$?
  if [ -n "$generated_environment_file" ]; then
    rm -f "$generated_environment_file"
  fi
  [ "$compose_status" -eq 0 ] || return "$compose_status"

  if [ "$remove_images" = true ]; then
    remove_project_images
  fi

  printf 'Preserved: config/remote-server.local.yaml, config/remote-gpu.env, .env, models/, and the NVIDIA NGC vLLM image.\n'
}

case "$operation:$deployment_target" in
  deploy:jetson) deploy_jetson ;;
  deploy:a6000) deploy_a6000 ;;
  uninstall:jetson) uninstall_jetson ;;
  uninstall:a6000) uninstall_a6000 ;;
esac
