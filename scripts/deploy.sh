#!/usr/bin/env bash
# Deploy the resident model services on either supported inference host.
#
# The script validates host identity and required artifacts before Compose changes state.
# Existing host overrides and environment files are never replaced.
set -euo pipefail

repository_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repository_root"

deployment_target=""
environment_file="$repository_root/.env"
health_timeout_seconds=600
build_images=true
prepare_jetson_power=true
docker_command=(docker)
docker_display_command="docker"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/deploy.sh jetson [options]
  bash scripts/deploy.sh a6000 [options]

Options:
  --env-file PATH       Compose environment file for A6000 deployment (default: .env)
  --health-timeout SEC  Maximum model startup wait in seconds (default: 600)
  --no-build            Start existing images without building
  --skip-power-setup    Do not set Jetson MAXN mode or lock clocks
  -h, --help            Show this help

The script does not start the interactive orchestrator. It prints the runtime command
after the resident model services pass their health checks.
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

require_command docker
require_command curl
require_command python3

configure_docker_access() {
  if docker info >/dev/null 2>&1; then
    docker_command=(docker)
    docker_display_command="docker"
  else
    require_command sudo
    sudo docker info >/dev/null 2>&1 \
      || fail "Docker daemon is not available through docker or sudo docker"
    docker_command=(sudo docker)
    docker_display_command="sudo docker"
    printf 'Docker requires elevated access; using sudo docker.\n'
  fi
  "${docker_command[@]}" compose version >/dev/null 2>&1 \
    || fail "Docker Compose plugin is not available"
}

configure_docker_access

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
    environment_token=$(sed -n 's/^KOTONOHA_SERVICE_TOKEN=//p' "$environment_file" | tail -n 1)
    [ -n "$environment_token" ] \
      || fail "environment file does not define KOTONOHA_SERVICE_TOKEN: $environment_file"
    case "$environment_token" in
      *'<'*|*'>'*) fail "replace the token placeholder in $environment_file" ;;
    esac
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
    printf 'LLM_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda\n'
    printf 'LLM_PROFILE=moe\n'
    printf 'LLM_CTX=4096\n'
    printf 'TRANSFORMERS_OFFLINE=1\n'
  } >"$environment_file"
  printf 'Created protected environment file: %s\n' "$environment_file"
  printf 'Copy its KOTONOHA_SERVICE_TOKEN value to the Jetson through a secure channel.\n'
}

check_speech_models() {
  local models_path=$1
  require_directory "$models_path/Qwen3-ASR-1.7B-hf"
  require_directory "$models_path/faster-whisper-large-v3"
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

check_jetson_host() {
  [ "$(uname -m)" = "aarch64" ] || fail "Jetson deployment requires aarch64"
  [ -r /etc/nv_tegra_release ] || fail "Jetson L4T release file is missing"
  grep -Eq 'R36.*REVISION: 4' /etc/nv_tegra_release \
    || fail "Jetson deployment requires L4T r36.4.x"
  [ -d /dev/snd ] || fail "audio device directory is missing: /dev/snd"
  require_file "$repository_root/models/silero_vad.onnx"
  require_file "$repository_root/models/gguf/Qwen3-14B-Q4_K_M.gguf"
  "${docker_command[@]}" info --format '{{json .Runtimes}}' | grep -q '"nvidia"' \
    || fail "Docker does not report the NVIDIA runtime"

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
  require_file "$models_path/gguf/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
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
    "${docker_command[@]}" compose --env-file "$compose_environment_file" -f "$compose_file" \
      logs --tail=120 "$@" >&2 || true
  else
    "${docker_command[@]}" compose -f "$compose_file" logs --tail=120 "$@" >&2 || true
  fi
  exit 1
}

deploy_jetson() {
  local compose_file="$repository_root/docker/compose.yaml"
  printf 'Deploying Jetson model services from %s\n' "$repository_root"
  check_speech_models "$repository_root/models"
  check_jetson_host
  ensure_override \
    "$repository_root/config/jetson.local.example.yaml" \
    "$repository_root/config/local.yaml"

  "${docker_command[@]}" compose -f "$compose_file" config --quiet
  if [ "$build_images" = true ]; then
    "${docker_command[@]}" compose -f "$compose_file" build asr asr-verify tts orchestrator
  fi
  if [ "$build_images" = false ]; then
    "${docker_command[@]}" compose -f "$compose_file" \
      up -d --no-build asr asr-verify llm tts
  else
    "${docker_command[@]}" compose -f "$compose_file" up -d asr asr-verify llm tts
  fi

  wait_for_service asr http://127.0.0.1:8001 python \
    && wait_for_service asr-verify http://127.0.0.1:8002 python \
    && wait_for_service llm http://127.0.0.1:8003 http \
    && wait_for_service tts http://127.0.0.1:8004 python \
    || show_failed_health "$compose_file" "" asr asr-verify llm tts

  printf '\nJetson model services are ready. Start the interactive interpreter with:\n'
  printf '%s compose -f docker/compose.yaml run --rm orchestrator\n' "$docker_display_command"
}

deploy_a6000() {
  local compose_file="$repository_root/docker/compose.remote.yaml"
  local compose_command=(
    "${docker_command[@]}" compose --env-file "$environment_file" -f "$compose_file"
  )
  local models_path
  printf 'Deploying A6000 model services from %s\n' "$repository_root"
  ensure_remote_environment
  models_path=$(remote_models_path)
  check_speech_models "$models_path"
  check_a6000_host "$models_path"
  ensure_override \
    "$repository_root/config/remote-server.local.example.yaml" \
    "$repository_root/config/remote-server.local.yaml"

  "${compose_command[@]}" config --quiet
  if [ "$build_images" = true ]; then
    "${compose_command[@]}" build asr
  fi
  if [ "$build_images" = false ]; then
    "${compose_command[@]}" up -d --no-build asr asr-verify llm tts
  else
    "${compose_command[@]}" up -d asr asr-verify llm tts
  fi

  wait_for_service asr http://127.0.0.1:8001 python \
    && wait_for_service asr-verify http://127.0.0.1:8002 python \
    && wait_for_service llm http://127.0.0.1:8003 http \
    && wait_for_service tts http://127.0.0.1:8004 python \
    || show_failed_health "$compose_file" "$environment_file" asr asr-verify llm tts

  printf '\nA6000 model services are ready. Configure the Jetson with the token in:\n'
  printf '%s\n' "$environment_file"
}

case "$deployment_target" in
  jetson) deploy_jetson ;;
  a6000) deploy_a6000 ;;
esac
