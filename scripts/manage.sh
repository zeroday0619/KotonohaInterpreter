#!/usr/bin/env bash
# Provide one operator entry point for setup, deployment, and benchmarking workflows.
set -euo pipefail

repository_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repository_root"

dry_run=false
if [ "${1:-}" = "--dry-run" ]; then
  dry_run=true
  shift
fi

show_usage() {
  cat <<'EOF'
Usage:
  bash scripts/manage.sh [--dry-run] setup workstation [--eval]
  bash scripts/manage.sh [--dry-run] setup jetson [deploy options]
  bash scripts/manage.sh [--dry-run] setup a6000 [deploy options]
  bash scripts/manage.sh [--dry-run] models fetch
  bash scripts/manage.sh [--dry-run] models verify
  bash scripts/manage.sh [--dry-run] benchmark jetson [--only 1|2|3|all]
  bash scripts/manage.sh [--dry-run] benchmark a6000 [--only 1|2|3|all]
  bash scripts/manage.sh [--dry-run] benchmark link [netcheck options]
  bash scripts/manage.sh [--dry-run] deploy jetson [deploy options]
  bash scripts/manage.sh [--dry-run] deploy a6000 [deploy options]
  bash scripts/manage.sh [--dry-run] uninstall jetson [--remove-images]
  bash scripts/manage.sh [--dry-run] uninstall a6000 [--remove-images]
  bash scripts/manage.sh [--dry-run] gpu allocate [allocator options]
  bash scripts/manage.sh [--dry-run] doctor [doctor options]
  bash scripts/manage.sh [--dry-run] check

Commands:
  setup       Prepare a workstation or validate and initialize a target host.
  models      Download or validate every offline model artifact.
  benchmark   Run Docker hardware spikes or the Jetson-to-A6000 link benchmark.
  deploy      Build and start resident model services.
  uninstall   Remove project containers while preserving models and configuration.
  gpu         Run memory-aware A6000 GPU allocation.
  doctor      Report application environment and service health.
  check       Run lint, tests, and localization catalog validation.

Run models fetch before setup on a new Jetson or A6000 host. Target setup creates
host-specific configuration but does not build images or start services.
EOF
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
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

require_no_arguments() {
  [ "$#" -eq 0 ] || fail "unexpected argument: $1"
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
  -h|--help|help)
    show_usage
    ;;
  setup)
    setup_target=${1:-}
    [ -n "$setup_target" ] || fail "setup requires workstation, jetson, or a6000"
    shift
    case "$setup_target" in
      workstation)
        synchronize_arguments=(sync)
        if [ "${1:-}" = "--eval" ]; then
          synchronize_arguments+=(--group eval)
          shift
        fi
        require_no_arguments "$@"
        run_command uv "${synchronize_arguments[@]}"
        run_command uv run python scripts/i18n.py compile
        ;;
      jetson|a6000)
        run_command bash scripts/deploy.sh "$setup_target" --prepare-only "$@"
        ;;
      *) fail "unknown setup target: $setup_target" ;;
    esac
    ;;
  models)
    models_operation=${1:-}
    [ -n "$models_operation" ] || fail "models requires fetch or verify"
    shift
    case "$models_operation" in
      fetch)
        require_no_arguments "$@"
        run_command bash scripts/fetch_models.sh
        ;;
      verify)
        require_no_arguments "$@"
        if [ "$dry_run" = true ]; then
          printf 'DRY RUN: verify model artifacts in %s\n' "$(resolve_models_directory)"
        else
          verify_model_artifacts
        fi
        ;;
      *) fail "unknown models operation: $models_operation" ;;
    esac
    ;;
  benchmark)
    benchmark_target=${1:-}
    [ -n "$benchmark_target" ] || fail "benchmark requires jetson, a6000, or link"
    shift
    case "$benchmark_target" in
      jetson|a6000)
        run_command bash spikes/run_all.sh "$benchmark_target" "$@"
        ;;
      link)
        run_command uv run kotonoha -c config/performance.yaml netcheck "$@"
        ;;
      *) fail "unknown benchmark target: $benchmark_target" ;;
    esac
    ;;
  deploy)
    deployment_target=${1:-}
    case "$deployment_target" in
      jetson|a6000) ;;
      *) fail "deploy requires jetson or a6000" ;;
    esac
    shift
    run_command bash scripts/deploy.sh "$deployment_target" "$@"
    ;;
  uninstall)
    deployment_target=${1:-}
    case "$deployment_target" in
      jetson|a6000) ;;
      *) fail "uninstall requires jetson or a6000" ;;
    esac
    shift
    run_command bash scripts/deploy.sh uninstall "$deployment_target" "$@"
    ;;
  gpu)
    gpu_operation=${1:-}
    [ "$gpu_operation" = "allocate" ] || fail "gpu requires allocate"
    shift
    run_command python3 scripts/allocate_gpus.py "$@"
    ;;
  doctor)
    run_command uv run kotonoha doctor "$@"
    ;;
  check)
    require_no_arguments "$@"
    run_command uv run ruff check .
    run_command uv run pytest -q
    run_command uv run python scripts/i18n.py check
    ;;
  *)
    fail "unknown command: $command_name"
    ;;
esac
