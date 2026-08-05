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
docker_display_command="docker"
docker_requires_sudo=false
docker_environment_names=(
  ASR_BASE
  ASR_GPU_DEVICE
  ASR_VERIFY_GPU_DEVICE
  LLM_GPU_DEVICE
  LLM_GPU_MEMORY_UTILIZATION
  LLM_IMAGE
  LLM_MAX_MODEL_LEN
  LLM_PROFILE
  MODELS_DIR
  ORCH_BASE
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
    local configured_llm_image
    local configured_tts_image
    environment_token=$(sed -n 's/^KOTONOHA_SERVICE_TOKEN=//p' "$environment_file" | tail -n 1)
    [ -n "$environment_token" ] \
      || fail "environment file does not define KOTONOHA_SERVICE_TOKEN: $environment_file"
    case "$environment_token" in
      *'<'*|*'>'*) fail "replace the token placeholder in $environment_file" ;;
    esac
    configured_asr_image=$(sed -n 's/^REMOTE_ASR_BASE=//p' "$environment_file" | tail -n 1)
    configured_llm_image=$(sed -n 's/^LLM_IMAGE=//p' "$environment_file" | tail -n 1)
    configured_tts_image=$(sed -n 's/^TTS_IMAGE=//p' "$environment_file" | tail -n 1)
    [ -z "$configured_asr_image" ] || [ "$configured_asr_image" = "$a6000_vllm_image" ] \
      || fail "environment file must set REMOTE_ASR_BASE=$a6000_vllm_image"
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

  umask 077
  {
    printf 'KOTONOHA_SERVICE_TOKEN=%s\n' "$service_token"
    printf 'MODELS_DIR=../models\n'
    printf 'REMOTE_BASE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime\n'
    printf 'REMOTE_ASR_BASE=%s\n' "$a6000_vllm_image"
    printf 'TTS_IMAGE=%s\n' "$vllm_omni_image"
    printf 'LLM_IMAGE=%s\n' "$a6000_vllm_image"
    printf 'LLM_PROFILE=moe\n'
    printf 'LLM_MAX_MODEL_LEN=4096\n'
    printf 'LLM_GPU_MEMORY_UTILIZATION=0.55\n'
    printf 'GPU_ALLOCATION_MODE=auto\n'
    printf 'GPU_NAME_FILTER=A6000\n'
    printf 'GPU_MEMORY_RESERVE_MIB=1024\n'
    printf 'LLM_GPU_MEMORY_MIB=27648\n'
    printf 'ASR_GPU_MEMORY_MIB=10240\n'
    printf 'ASR_VERIFY_GPU_MEMORY_MIB=6144\n'
    printf 'TTS_GPU_MEMORY_MIB=3072\n'
    printf 'TTS_GPU_MEMORY_UTILIZATION=0.25\n'
    printf 'TTS_ENFORCE_EAGER=1\n'
    printf 'TRANSFORMERS_OFFLINE=1\n'
  } >"$environment_file"
  printf 'Created protected environment file: %s\n' "$environment_file"
  printf 'Copy its KOTONOHA_SERVICE_TOKEN value to the Jetson through a secure channel.\n'
}

check_speech_models() {
  local models_path=$1
  require_directory "$models_path/Qwen3-ASR-1.7B"
  require_file "$models_path/Qwen3-ASR-1.7B/config.json"
  require_directory "$models_path/faster-whisper-large-v3"
  require_file "$models_path/faster-whisper-large-v3/config.json"
  require_directory "$models_path/Qwen3-TTS-0.6B"
  require_file "$models_path/Qwen3-TTS-0.6B/config.json"
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
  require_file "$repository_root/models/llm/Qwen3-14B-AWQ/config.json"
  check_nvidia_container_runtime

  require_command nvpmodel
  require_command jetson_clocks
  if [ "$prepare_jetson_power" = true ]; then
    run_privileged nvpmodel -m 0
    run_privileged jetson_clocks
  fi
  nvpmodel -q || fail "unable to read the Jetson power mode"
  jetson_clocks --show || fail "unable to read the Jetson clock state"
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
  require_file "$models_path/llm/Qwen3-30B-A3B-Instruct-2507-AWQ/config.json"
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
    python3 "$repository_root/scripts/allocate_gpus.py"
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
    'import torch, vllm; assert torch.cuda.is_available(), "CUDA is unavailable in the vLLM container"; print("CUDA", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0), "| vLLM", vllm.__version__)' \
    || fail "$service_name container cannot initialize the CUDA runtime and vLLM"
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
    'from importlib.metadata import version; import torch, vllm, vllm_omni; assert torch.cuda.is_available(), "CUDA is unavailable in the vLLM-Omni container"; print("CUDA", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0), "| vLLM", vllm.__version__, "| vLLM-Omni", version("vllm-omni"))' \
    || fail "tts container cannot initialize the CUDA runtime and vLLM-Omni"
}

deploy_jetson() {
  local compose_file="$repository_root/docker/compose.yaml"
  printf 'Deploying Jetson model services from %s\n' "$repository_root"
  check_speech_models "$repository_root/models"
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
    run_docker compose -f "$compose_file" build asr asr-verify tts orchestrator
  fi
  verify_vllm_cuda_runtime "$compose_file" "" asr
  verify_vllm_cuda_runtime "$compose_file" "" llm
  verify_vllm_omni_cuda_runtime "$compose_file" ""
  if [ "$build_images" = false ]; then
    run_docker compose -f "$compose_file" \
      up -d --no-build asr asr-verify llm tts
  else
    run_docker compose -f "$compose_file" up -d asr asr-verify llm tts
  fi

  if ! wait_for_service asr http://127.0.0.1:8001 python \
    || ! wait_for_service asr-verify http://127.0.0.1:8002 python \
    || ! wait_for_service llm http://127.0.0.1:8003 http \
    || ! wait_for_service tts http://127.0.0.1:8004 python; then
    show_failed_health "$compose_file" "" asr asr-verify llm tts
  fi

  printf '\nJetson model services are ready. Start the interactive interpreter with:\n'
  printf '%s compose -f docker/compose.yaml run --rm orchestrator\n' "$docker_display_command"
}

deploy_a6000() {
  local compose_file="$repository_root/docker/compose.remote.yaml"
  local compose_command=(
    run_docker compose --env-file "$environment_file" -f "$compose_file"
  )
  local models_path
  printf 'Deploying A6000 model services from %s\n' "$repository_root"
  ensure_remote_environment
  models_path=$(remote_models_path)
  check_speech_models "$models_path"
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
    "${compose_command[@]}" build asr asr-verify tts
  fi
  verify_vllm_cuda_runtime "$compose_file" "$environment_file" asr
  verify_vllm_cuda_runtime "$compose_file" "$environment_file" llm
  verify_vllm_omni_cuda_runtime "$compose_file" "$environment_file"
  if [ "$build_images" = false ]; then
    "${compose_command[@]}" up -d --no-build asr asr-verify llm tts
  else
    "${compose_command[@]}" up -d asr asr-verify llm tts
  fi

  if ! wait_for_service asr http://127.0.0.1:8001 python \
    || ! wait_for_service asr-verify http://127.0.0.1:8002 python \
    || ! wait_for_service llm http://127.0.0.1:8003 http \
    || ! wait_for_service tts http://127.0.0.1:8004 python; then
    show_failed_health "$compose_file" "$environment_file" asr asr-verify llm tts
  fi

  printf '\nA6000 model services are ready. Configure the Jetson with the token in:\n'
  printf '%s\n' "$environment_file"
}

remove_project_image() {
  local image_name=$1
  if run_docker image inspect "$image_name" >/dev/null 2>&1; then
    run_docker image rm "$image_name"
  else
    printf 'Image not present: %s\n' "$image_name"
  fi
}

uninstall_jetson() {
  local compose_file="$repository_root/docker/compose.yaml"
  printf 'Removing Jetson project containers and network.\n'
  run_docker compose -f "$compose_file" down --remove-orphans

  if [ "$remove_images" = true ]; then
    remove_project_image kotonohainterpreter-asr
    remove_project_image kotonohainterpreter-asr-verify
    remove_project_image kotonohainterpreter-tts
    remove_project_image kotonohainterpreter-orchestrator
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
    remove_project_image kotonohainterpreter-asr
    remove_project_image kotonohainterpreter-asr-verify
    remove_project_image kotonohainterpreter-tts
  fi

  printf 'Preserved: config/remote-server.local.yaml, config/remote-llm.env, config/remote-gpu.env, .env, models/, and the NVIDIA NGC vLLM image.\n'
}

case "$operation:$deployment_target" in
  deploy:jetson) deploy_jetson ;;
  deploy:a6000) deploy_a6000 ;;
  uninstall:jetson) uninstall_jetson ;;
  uninstall:a6000) uninstall_a6000 ;;
esac
