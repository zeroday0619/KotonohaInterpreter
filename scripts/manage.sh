#!/usr/bin/env bash
# Provide one confirmed operator entry point for setup, deployment, and benchmarks.
set -euo pipefail

repository_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repository_root"

dry_run=false
assume_yes=false
image_removal=ask
management_arguments=()
docker_requires_sudo=false
docker_environment_names=(
  ACCELERATOR_OMNI_IMAGE
  ACCELERATOR_PROFILE
  ACCELERATOR_VLLM_IMAGE
  ASR_BASE
  CONTAINER_RUNTIME
  KOTONOHA_DISABLE_NVML
  KOTONOHA_WEB_HOST
  KOTONOHA_WEB_PORT
  KOTONOHA_WEB_SESSIONS
  LLM_ENABLE_PREFIX_CACHING
  LLM_GPU_MEMORY_UTILIZATION
  LLM_IMAGE
  LLM_KV_CACHE_DTYPE
  LLM_MAX_MODEL_LEN
  LLM_MAX_NUM_BATCHED_TOKENS
  ORCH_BASE
  TTS_ENFORCE_EAGER
  TTS_GPU_MEMORY_UTILIZATION
  TTS_IMAGE
  VERIFY_BASE
  VLLM_NVML_PATCH
)

for argument in "$@"; do
  case "$argument" in
    --dry-run)
      dry_run=true
      ;;
    -y|--yes)
      assume_yes=true
      ;;
    --remove-images)
      image_removal=remove
      ;;
    --keep-images)
      image_removal=keep
      ;;
    *)
      management_arguments+=("$argument")
      ;;
  esac
done
set -- "${management_arguments[@]}"

show_usage() {
  cat <<'EOF'
Usage:
  bash scripts/manage.sh [--dry-run] [-y|--yes] detect
  bash scripts/manage.sh [--dry-run] [-y|--yes] setup [auto|workstation|jetson|a6000] [--eval]
  bash scripts/manage.sh [--dry-run] [-y|--yes] models fetch
  bash scripts/manage.sh [--dry-run] [-y|--yes] models verify
  bash scripts/manage.sh [--dry-run] [-y|--yes] i18n [extract|update|compile|check]
  bash scripts/manage.sh [--dry-run] [-y|--yes] benchmark [auto|jetson|a6000] [--only 1|2|3|all]
  bash scripts/manage.sh [--dry-run] [-y|--yes] benchmark link [netcheck options]
  bash scripts/manage.sh [--dry-run] [-y|--yes] deploy [auto|jetson|a6000] [deploy options]
  bash scripts/manage.sh [--dry-run] [-y|--yes] tui
  bash scripts/manage.sh [--dry-run] [-y|--yes] web
  bash scripts/manage.sh [--dry-run] [-y|--yes] uninstall [auto|jetson|a6000] [--remove-images|--keep-images]
  bash scripts/manage.sh [--dry-run] [-y|--yes] gpu allocate [allocator options]
  bash scripts/manage.sh [--dry-run] [-y|--yes] doctor [doctor options]
  bash scripts/manage.sh [--dry-run] [-y|--yes] check

Commands:
  detect      Report the automatically detected equipment class.
  setup       Prepare the detected host or an explicitly selected host.
  models      Download or validate every offline model artifact.
  i18n        Extract, update, compile, or check gettext catalogs.
  benchmark   Run Docker hardware spikes or the Jetson-to-A6000 link benchmark.
  deploy      Build and start resident model services on the detected inference host.
  tui         Start the interactive orchestrator in Docker.
  web         Serve the browser interface over HTTP. Audio runs in the
              browser. Loopback unless KOTONOHA_WEB_HOST is set, and it
              has no authentication.
  uninstall   Remove project containers and optionally Kotonoha Docker images.
  gpu         Run memory-aware A6000 GPU allocation.
  doctor      Report application environment and service health.
  check       Run lint, tests, and localization catalog validation.

Every command requests confirmation. Use -y or --yes to answer yes without an
interactive prompt. Uninstall asks whether to remove Kotonoha images unless
--remove-images or --keep-images is specified. With --yes, image removal defaults to yes.

Set KOTONOHA_EQUIPMENT to workstation, jetson, or a6000 only when automatic detection
must be overridden for automation. Target setup creates host-specific configuration but
does not build images or start services.
EOF
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

# Step progress for the subcommands that run several long tools in sequence.
#
# The bar is printed as its own line rather than redrawn in place: every step
# here runs a tool that writes to the terminal, so an in-place bar would be
# scrolled away by the first line the tool prints. Progress goes to standard
# error so that redirected command output stays machine-readable.
progress_bar_filled='########################'
progress_bar_empty='........................'
progress_width=24
progress_total=0
progress_index=0

progress_begin() {
  progress_total=$1
  progress_index=0
  printf '\n== %s (%d steps)\n' "$2" "$progress_total" >&2
}

progress_step() {
  [ "$progress_total" -gt 0 ] || return 0
  progress_index=$((progress_index + 1))
  local filled=$((progress_index * progress_width / progress_total))
  printf '[%s%s] %3d%% (%d/%d) %s\n' \
    "${progress_bar_filled:0:filled}" \
    "${progress_bar_empty:0:$((progress_width - filled))}" \
    "$((progress_index * 100 / progress_total))" \
    "$progress_index" "$progress_total" "$1" >&2
}

progress_end() {
  printf '[%s] 100%% %s\n\n' "$progress_bar_filled" "${1:-done}" >&2
  progress_total=0
  progress_index=0
}

run_command() {
  if [ "$dry_run" = true ]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  "$@"
}

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
  command -v docker >/dev/null 2>&1 || fail "required command not found: docker"
  if docker info >/dev/null 2>&1; then
    docker_requires_sudo=false
    return
  fi
  command -v sudo >/dev/null 2>&1 || fail "required command not found: sudo"
  sudo docker info >/dev/null 2>&1 \
    || fail "Docker daemon is not available through docker or sudo docker"
  docker_requires_sudo=true
  printf 'Docker requires elevated access; using sudo docker.\n'
}

run_tui() {
  local compose_file="$repository_root/docker/compose.yaml"
  if [ "$dry_run" = true ]; then
    run_command docker compose -f "$compose_file" run --rm orchestrator
    return
  fi
  configure_docker_access
  run_docker compose -f "$compose_file" run --rm orchestrator
}

# The web overlay only adds the front end, so both files are always passed: the
# model services it depends on are declared in the base compose file.
run_web() {
  local compose_file="$repository_root/docker/compose.yaml"
  local web_compose_file="$repository_root/docker/compose.web.yaml"
  local bind_address=${KOTONOHA_WEB_HOST:-127.0.0.1}
  if [ "$dry_run" = true ]; then
    run_command docker compose -f "$compose_file" -f "$web_compose_file" up -d web
    return
  fi
  configure_docker_access
  run_docker compose -f "$compose_file" -f "$web_compose_file" up -d web
  printf 'Web interface listening on http://%s:%s\n' \
    "$bind_address" "${KOTONOHA_WEB_PORT:-8080}"
  if [ "$bind_address" = "127.0.0.1" ]; then
    printf 'Reachable from this host only. Set KOTONOHA_WEB_HOST=0.0.0.0 to expose it.\n'
  fi
}

require_no_arguments() {
  [ "$#" -eq 0 ] || fail "unexpected argument: $1"
}

confirm() {
  local prompt=$1
  local response
  if [ "$assume_yes" = true ]; then
    printf 'Confirmed by --yes: %s\n' "$prompt"
    return
  fi
  [ -t 0 ] || fail "confirmation requires an interactive terminal; rerun with -y or --yes"
  printf '%s [y/N] ' "$prompt"
  read -r response
  case "$response" in
    y|Y|yes|YES|Yes) ;;
    *)
      printf 'Cancelled.\n'
      exit 0
      ;;
  esac
}

confirm_optional() {
  local prompt=$1
  local response
  if [ "$assume_yes" = true ]; then
    return 0
  fi
  [ -t 0 ] || fail "confirmation requires an interactive terminal; rerun with -y or --yes"
  printf '%s [y/N] ' "$prompt"
  read -r response
  case "$response" in
    y|Y|yes|YES|Yes) return 0 ;;
    *) return 1 ;;
  esac
}

detect_equipment() {
  local configured_equipment=${KOTONOHA_EQUIPMENT:-}
  if [ -n "$configured_equipment" ]; then
    case "$configured_equipment" in
      workstation|jetson|a6000)
        printf '%s\n' "$configured_equipment"
        return
        ;;
      *) fail "KOTONOHA_EQUIPMENT must be workstation, jetson, or a6000" ;;
    esac
  fi

  if [ -r /etc/nv_tegra_release ] && [ "$(uname -m)" = "aarch64" ]; then
    printf 'jetson\n'
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null \
      | grep -Fq 'A6000'; then
    printf 'a6000\n'
    return
  fi
  if [ "$(uname -s)" = "Darwin" ]; then
    printf 'workstation\n'
    return
  fi
  fail "equipment could not be detected; specify a target or set KOTONOHA_EQUIPMENT"
}

resolve_equipment() {
  local requested_equipment=${1:-auto}
  case "$requested_equipment" in
    auto|'') detect_equipment ;;
    workstation|jetson|a6000) printf '%s\n' "$requested_equipment" ;;
    *) fail "unknown equipment target: $requested_equipment" ;;
  esac
}

resolve_inference_equipment() {
  local requested_equipment=${1:-auto}
  local detected_equipment
  detected_equipment=$(resolve_equipment "$requested_equipment")
  case "$detected_equipment" in
    jetson|a6000) printf '%s\n' "$detected_equipment" ;;
    *) fail "this operation requires a Jetson or A6000 host, detected: $detected_equipment" ;;
  esac
}

resolve_models_directory() {
  local configured_directory=${MODELS_DIR:-./models}
  case "$configured_directory" in
    /*) printf '%s\n' "$configured_directory" ;;
    *) printf '%s/%s\n' "$repository_root" "${configured_directory#./}" ;;
  esac
}

verify_model_artifacts() {
  local models_directory
  local missing_count=0
  local artifact
  models_directory=$(resolve_models_directory)
  local required_artifacts=(
    silero_vad.onnx
    Qwen3-ASR-0.6B/config.json
    Qwen3-ASR-0.6B-hf/config.json
    Voxtral-Mini-4B-Realtime-2602/config.json
    Qwen3-TTS-0.6B/config.json
    faster-whisper-large-v3/config.json
    llm/translategemma-4b-it/config.json
    llm/translategemma-12b-it/config.json
  )

  for artifact in "${required_artifacts[@]}"; do
    if [ -s "$models_directory/$artifact" ]; then
      printf 'OK: %s\n' "$models_directory/$artifact"
    else
      printf 'MISSING: %s\n' "$models_directory/$artifact" >&2
      missing_count=$((missing_count + 1))
    fi
  done
  [ "$missing_count" -eq 0 ] \
    || fail "$missing_count required model artifact(s) are missing"
}

command_name=${1:-}
if [ -z "$command_name" ]; then
  show_usage
  exit 2
fi
shift
case "$command_name" in
  -h|--help|help|uninstall) ;;
  *)
    [ "$image_removal" = ask ] \
      || fail "image removal options are valid only with uninstall"
    ;;
esac

case "$command_name" in
  -h|--help|help)
    show_usage
    ;;
  detect)
    require_no_arguments "$@"
    confirm "Detect equipment"
    printf '%s\n' "$(detect_equipment)"
    ;;
  setup)
    setup_target=${1:-auto}
    case "$setup_target" in
      auto|workstation|jetson|a6000) [ "$#" -eq 0 ] || shift ;;
      --*) setup_target=auto ;;
      *) fail "setup target must be auto, workstation, jetson, or a6000" ;;
    esac
    setup_target=$(resolve_equipment "$setup_target")
    confirm "Set up $setup_target"
    case "$setup_target" in
      workstation)
        synchronize_arguments=(sync)
        if [ "${1:-}" = "--eval" ]; then
          synchronize_arguments+=(--group eval)
          shift
        fi
        require_no_arguments "$@"
        progress_begin 2 "Workstation setup"
        progress_step "dependencies"
        run_command uv "${synchronize_arguments[@]}"
        progress_step "translation catalogs"
        run_command uv run python scripts/py/i18n.py compile
        progress_end "workstation ready"
        ;;
      jetson|a6000)
        run_command bash scripts/deploy.sh "$setup_target" --prepare-only "$@"
        ;;
    esac
    ;;
  models)
    models_operation=${1:-}
    [ -n "$models_operation" ] || fail "models requires fetch or verify"
    shift
    case "$models_operation" in
      fetch)
        require_no_arguments "$@"
        confirm "Fetch offline model artifacts"
        run_command bash scripts/fetch_models.sh
        ;;
      verify)
        require_no_arguments "$@"
        confirm "Verify offline model artifacts"
        if [ "$dry_run" = true ]; then
          printf 'DRY RUN: verify model artifacts in %s\n' "$(resolve_models_directory)"
        else
          verify_model_artifacts
        fi
        ;;
      *) fail "unknown models operation: $models_operation" ;;
    esac
    ;;
  i18n)
    i18n_operation=${1:-}
    case "$i18n_operation" in
      extract|update|compile|check) ;;
      '') fail "i18n requires extract, update, compile, or check" ;;
      *) fail "unknown i18n operation: $i18n_operation" ;;
    esac
    shift
    require_no_arguments "$@"
    confirm "Run i18n $i18n_operation"
    run_command uv run python scripts/py/i18n.py "$i18n_operation"
    ;;
  benchmark)
    benchmark_target=${1:-auto}
    case "$benchmark_target" in
      auto|jetson|a6000|link) [ "$#" -eq 0 ] || shift ;;
      --*) benchmark_target=auto ;;
      *) fail "benchmark target must be auto, jetson, a6000, or link" ;;
    esac
    if [ "$benchmark_target" = "link" ]; then
      confirm "Benchmark the external network link"
      run_command uv run kotonoha -c config/performance.yaml netcheck "$@"
    else
      benchmark_target=$(resolve_inference_equipment "$benchmark_target")
      confirm "Run $benchmark_target hardware benchmarks"
      run_command bash spikes/run_all.sh "$benchmark_target" "$@"
    fi
    ;;
  deploy)
    deployment_target=${1:-auto}
    case "$deployment_target" in
      auto|jetson|a6000) [ "$#" -eq 0 ] || shift ;;
      --*) deployment_target=auto ;;
      *) fail "deploy target must be auto, jetson, or a6000" ;;
    esac
    deployment_target=$(resolve_inference_equipment "$deployment_target")
    confirm "Deploy services on $deployment_target"
    run_command bash scripts/deploy.sh "$deployment_target" "$@"
    ;;
  tui)
    require_no_arguments "$@"
    confirm "Start the integrated TUI"
    run_tui
    ;;
  web)
    require_no_arguments "$@"
    confirm "Serve the browser interface over HTTP"
    run_web
    ;;
  uninstall)
    deployment_target=${1:-auto}
    case "$deployment_target" in
      auto|jetson|a6000) [ "$#" -eq 0 ] || shift ;;
      --*) deployment_target=auto ;;
      *) fail "uninstall target must be auto, jetson, or a6000" ;;
    esac
    deployment_target=$(resolve_inference_equipment "$deployment_target")
    if [ "$image_removal" = ask ]; then
      if confirm_optional "Remove Kotonoha Docker images after uninstall"; then
        image_removal=remove
      else
        image_removal=keep
      fi
    fi
    if [ "$image_removal" = remove ]; then
      uninstall_arguments=(--remove-images "$@")
      uninstall_description="Uninstall $deployment_target services and remove Kotonoha images"
    else
      uninstall_arguments=("$@")
      uninstall_description="Uninstall $deployment_target services and preserve images"
    fi
    confirm "$uninstall_description"
    run_command bash scripts/deploy.sh uninstall "$deployment_target" \
      "${uninstall_arguments[@]}"
    ;;
  gpu)
    gpu_operation=${1:-}
    [ "$gpu_operation" = "allocate" ] || fail "gpu requires allocate"
    shift
    confirm "Allocate A6000 GPUs"
    run_command python3 scripts/py/allocate_gpus.py "$@"
    ;;
  doctor)
    confirm "Run environment diagnostics"
    run_command uv run kotonoha doctor "$@"
    ;;
  check)
    require_no_arguments "$@"
    confirm "Run repository quality checks"
    progress_begin 3 "Quality gates"
    progress_step "ruff"
    run_command uv run ruff check .
    progress_step "pytest"
    run_command uv run pytest -q
    progress_step "translation catalogs"
    run_command uv run python scripts/py/i18n.py check
    progress_end "quality gates passed"
    ;;
  *)
    fail "unknown command: $command_name"
    ;;
esac
