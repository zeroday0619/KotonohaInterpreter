#!/usr/bin/env bash
# Run every hardware spike in an isolated Docker container on the selected host.
set -uo pipefail

repository_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repository_root"

show_usage() {
  cat <<'EOF'
Usage: bash spikes/run_all.sh TARGET [--only 1|2|3|all]

TARGET is jetson or a6000. Every selected probe and the report generator run in Docker.
EOF
}

deployment_target=${TARGET:-jetson}
selected_spike=all
if [ "$#" -gt 0 ]; then
  case "$1" in
    -h|--help)
      show_usage
      exit 0
      ;;
  esac
  deployment_target=$1
  shift
fi
case "$deployment_target" in
  jetson|a6000) ;;
  *)
    printf 'Target must be jetson or a6000.\n' >&2
    exit 2
    ;;
esac

while [ "$#" -gt 0 ]; do
  case "$1" in
    --only)
      [ "$#" -ge 2 ] || {
        printf '%s\n' '--only requires 1, 2, 3, or all.' >&2
        exit 2
      }
      selected_spike=$2
      shift 2
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done
case "$selected_spike" in
  1|2|3|all) ;;
  *)
    printf '%s\n' '--only requires 1, 2, 3, or all.' >&2
    exit 2
    ;;
esac

docker_command=(docker)
docker_display_command="docker"
docker_requires_sudo=false
asr_image_was_configured=false
tts_image_was_configured=false
if [ -n "${SPIKE_ASR_IMAGE:-}" ]; then
  asr_image_was_configured=true
fi
if [ -n "${SPIKE_TTS_IMAGE:-}" ]; then
  tts_image_was_configured=true
fi

configure_docker_access() {
  if docker info >/dev/null 2>&1; then
    return
  fi
  command -v sudo >/dev/null 2>&1 || {
    printf 'Docker requires elevated access, but sudo is unavailable.\n' >&2
    exit 1
  }
  sudo docker info >/dev/null 2>&1 || {
    printf 'Docker is unavailable through both docker and sudo docker.\n' >&2
    exit 1
  }
  docker_command=(sudo docker)
  docker_display_command="sudo docker"
  docker_requires_sudo=true
  printf 'Docker requires elevated access; using sudo docker.\n'
}

configure_target() {
  if [ "$deployment_target" = "jetson" ]; then
    : "${SPIKE_VLLM_IMAGE:=ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04}"
    : "${SPIKE_ASR_IMAGE:=kotonohainterpreter-spike-asr:jetson}"
    : "${SPIKE_TTS_IMAGE:=vllm/vllm-omni:v0.26.0}"
    : "${SPIKE_GPU_DEVICE:=all}"
    : "${SPIKE_PYTHON:=/opt/venv/bin/python}"
    : "${SPIKE_TTS_PYTHON:=python3}"
    : "${OUT:=spikes/out}"
    default_context=2048
    report_name=PHASE0.md
    patch_name=local.yaml
  else
    : "${SPIKE_VLLM_IMAGE:=nvcr.io/nvidia/vllm:26.07-py3}"
    : "${SPIKE_ASR_IMAGE:=kotonohainterpreter-spike-asr:a6000}"
    : "${SPIKE_TTS_IMAGE:=vllm/vllm-omni:v0.26.0}"
    : "${SPIKE_GPU_DEVICE:=0}"
    : "${SPIKE_PYTHON:=python3}"
    : "${SPIKE_TTS_PYTHON:=python3}"
    : "${OUT:=spikes/out/a6000}"
    default_context=4096
    report_name=PERFORMANCE.md
    patch_name=remote-server.local.yaml
  fi
  models_directory=${MODELS_DIR:-./models}
  case "$models_directory" in
    /*) MODELS_DIR=$models_directory ;;
    *) MODELS_DIR=$repository_root/${models_directory#./} ;;
  esac
  case "$OUT" in
    /*|*../*|../*)
      printf 'OUT must be a repository-relative path: %s\n' "$OUT" >&2
      exit 2
      ;;
  esac
  : "${SPIKE_USER_ID:=$(id -u)}"
  : "${SPIKE_GROUP_ID:=$(id -g)}"
  export MODELS_DIR OUT SPIKE_ASR_IMAGE SPIKE_GPU_DEVICE SPIKE_GROUP_ID
  export SPIKE_PYTHON SPIKE_TTS_IMAGE SPIKE_TTS_PYTHON SPIKE_USER_ID SPIKE_VLLM_IMAGE
}

build_asr_image() {
  if [ "$asr_image_was_configured" = true ]; then
    "${docker_command[@]}" image inspect "$SPIKE_ASR_IMAGE" >/dev/null 2>&1 || {
      printf 'Configured ASR spike image is missing: %s\n' "$SPIKE_ASR_IMAGE" >&2
      exit 1
    }
    return
  fi
  if [ "${SPIKE_SKIP_BUILD:-0}" = "1" ]; then
    "${docker_command[@]}" image inspect "$SPIKE_ASR_IMAGE" >/dev/null 2>&1 || {
      printf 'Required ASR spike image is missing: %s\n' "$SPIKE_ASR_IMAGE" >&2
      exit 1
    }
    return
  fi

  printf 'Building ASR spike image: %s\n' "$SPIKE_ASR_IMAGE"
  if [ "$deployment_target" = "jetson" ]; then
    "${docker_command[@]}" build \
      --build-arg "BASE_IMAGE=$SPIKE_VLLM_IMAGE" \
      --file docker/Dockerfile.asr \
      --tag "$SPIKE_ASR_IMAGE" \
      .
  else
    "${docker_command[@]}" build \
      --build-arg "ASR_BASE_IMAGE=$SPIKE_VLLM_IMAGE" \
      --file docker/Dockerfile.remote \
      --target asr \
      --tag "$SPIKE_ASR_IMAGE" \
      .
  fi
}

prepare_tts_image() {
  if [ "$tts_image_was_configured" = true ]; then
    "${docker_command[@]}" image inspect "$SPIKE_TTS_IMAGE" >/dev/null 2>&1 || {
      printf 'Configured TTS spike image is missing: %s\n' "$SPIKE_TTS_IMAGE" >&2
      exit 1
    }
    return
  fi
  if [ "${SPIKE_SKIP_BUILD:-0}" = "1" ]; then
    "${docker_command[@]}" image inspect "$SPIKE_TTS_IMAGE" >/dev/null 2>&1 || {
      printf 'Required TTS spike image is missing: %s\n' "$SPIKE_TTS_IMAGE" >&2
      exit 1
    }
    return
  fi

  "${docker_command[@]}" image inspect "$SPIKE_TTS_IMAGE" >/dev/null 2>&1 || {
    printf 'Pulling vLLM-Omni TTS image: %s\n' "$SPIKE_TTS_IMAGE"
    "${docker_command[@]}" pull "$SPIKE_TTS_IMAGE"
  }
}

require_model_snapshot() {
  local model_directory=$1
  if [ ! -s "$MODELS_DIR/$model_directory/config.json" ]; then
    printf 'Required model snapshot is missing: %s/config.json\n' \
      "$MODELS_DIR/$model_directory" >&2
    exit 1
  fi
}

validate_models() {
  if [ "$selected_spike" = "1" ] || [ "$selected_spike" = "all" ]; then
    case "${ASR_ONLY:-}" in
      transformers)
        require_model_snapshot Qwen3-ASR-1.7B-hf
        ;;
      vllm)
        require_model_snapshot Qwen3-ASR-1.7B
        ;;
      *)
        require_model_snapshot Qwen3-ASR-1.7B
        if [ "$deployment_target" = "jetson" ]; then
          require_model_snapshot Qwen3-ASR-1.7B-hf
        fi
        ;;
    esac
  fi
  if [ "$selected_spike" = "2" ] || [ "$selected_spike" = "all" ]; then
    require_model_snapshot Qwen3-TTS-0.6B
  fi
  if [ "$selected_spike" = "3" ] || [ "$selected_spike" = "all" ]; then
    require_model_snapshot llm/Qwen3-14B-AWQ
    require_model_snapshot llm/Qwen3-30B-A3B-Instruct-2507-AWQ
  fi
}

configure_target
validate_models
configure_docker_access
"${docker_command[@]}" compose version >/dev/null 2>&1 || {
  printf 'Docker Compose plugin is unavailable.\n' >&2
  exit 1
}
if [ "$selected_spike" = "1" ] || [ "$selected_spike" = "all" ]; then
  build_asr_image || {
    printf 'ASR spike image build failed: %s\n' "$SPIKE_ASR_IMAGE" >&2
    exit 1
  }
fi
if [ "$selected_spike" = "2" ] || [ "$selected_spike" = "all" ]; then
  prepare_tts_image || {
    printf 'TTS spike image preparation failed: %s\n' "$SPIKE_TTS_IMAGE" >&2
    exit 1
  }
fi
mkdir -p "$OUT"

compose_environment=(
  "MODELS_DIR=$MODELS_DIR"
  "OUT=$OUT"
  "SPIKE_ASR_IMAGE=$SPIKE_ASR_IMAGE"
  "SPIKE_GPU_DEVICE=$SPIKE_GPU_DEVICE"
  "SPIKE_GROUP_ID=$SPIKE_GROUP_ID"
  "SPIKE_PYTHON=$SPIKE_PYTHON"
  "SPIKE_TTS_IMAGE=$SPIKE_TTS_IMAGE"
  "SPIKE_TTS_PYTHON=$SPIKE_TTS_PYTHON"
  "SPIKE_USER_ID=$SPIKE_USER_ID"
  "SPIKE_VLLM_IMAGE=$SPIKE_VLLM_IMAGE"
)
if [ "$docker_requires_sudo" = true ]; then
  compose_command=(sudo env "${compose_environment[@]}" docker compose)
else
  compose_command=(env "${compose_environment[@]}" docker compose)
fi
compose_command+=(--file docker/compose.spikes.yaml)

printf 'Hardware spike target: %s\n' "$deployment_target"
printf 'Selected spike: %s\n' "$selected_spike"
printf 'Docker command: %s\n' "$docker_display_command"
printf 'vLLM image: %s\n' "$SPIKE_VLLM_IMAGE"
printf 'ASR image: %s\n' "$SPIKE_ASR_IMAGE"
printf 'TTS image: %s\n' "$SPIKE_TTS_IMAGE"
printf 'GPU selection: %s\n' "$SPIKE_GPU_DEVICE"
printf 'Output directory: %s\n' "$OUT"

wav_arguments=()
if [ "$selected_spike" = "1" ] || [ "$selected_spike" = "all" ]; then
  wav_path=${WAV:-samples/ko_6s.wav}
  if [ -f "$wav_path" ]; then
    case "$wav_path" in
      /*|*../*|../*)
        printf 'WAV must be a repository-relative path: %s\n' "$wav_path" >&2
        exit 2
        ;;
    esac
    wav_arguments=(--wav "/workspace/$wav_path")
  else
    printf 'No real recording found at %s; Spike 1 will use timing-only synthetic audio.\n' \
      "$wav_path"
  fi
fi

vllm_arguments=(
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
  --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}"
)
if [ "${VLLM_ENFORCE_EAGER:-1}" = "0" ]; then
  vllm_arguments+=(--no-enforce-eager)
else
  vllm_arguments+=(--enforce-eager)
fi

asr_arguments=()
if [ -n "${ASR_ONLY:-}" ]; then
  case "$ASR_ONLY" in
    vllm|transformers) asr_arguments=(--only "$ASR_ONLY") ;;
    *)
      printf 'ASR_ONLY must be vllm or transformers: %s\n' "$ASR_ONLY" >&2
      exit 2
      ;;
  esac
elif [ "$deployment_target" = "a6000" ]; then
  asr_arguments=(--only vllm)
fi

failure_count=0
if [ "$selected_spike" = "1" ] || [ "$selected_spike" = "all" ]; then
  printf '== Spike 1: ASR ==\n'
  "${compose_command[@]}" run --rm asr \
    --target "$deployment_target" \
    --vllm-model /models/Qwen3-ASR-1.7B \
    --transformers-model /models/Qwen3-ASR-1.7B-hf \
    "${wav_arguments[@]}" \
    "${vllm_arguments[@]}" \
    "${asr_arguments[@]}" \
    --out "/workspace/$OUT/spike1.json" || failure_count=$((failure_count + 1))
fi

if [ "$selected_spike" = "2" ] || [ "$selected_spike" = "all" ]; then
  printf '== Spike 2: FlashAttention and vLLM-Omni TTS ==\n'
  "${compose_command[@]}" run --rm tts \
    --target "$deployment_target" \
    --model /models/Qwen3-TTS-0.6B \
    --runs "${BENCHMARK_RUNS:-3}" \
    --gpu-memory-utilization "${TTS_GPU_MEMORY_UTILIZATION:-0.25}" \
    --log "/workspace/$OUT/spike2-vllm-omni.log" \
    --out "/workspace/$OUT/spike2.json" || failure_count=$((failure_count + 1))
fi

if [ "$selected_spike" = "3" ] || [ "$selected_spike" = "all" ]; then
  printf '== Spike 3: translation throughput ==\n'
  "${compose_command[@]}" run --rm llm \
    --target "$deployment_target" \
    --vllm-command vllm \
    --models-dir /models/llm \
    --context "${LLM_CONTEXT:-$default_context}" \
    --output-tokens "${LLM_OUTPUT_TOKENS:-60}" \
    --gpu-memory-utilization "${LLM_GPU_MEMORY_UTILIZATION:-0.80}" \
    --runs "${BENCHMARK_RUNS:-3}" \
    --out "/workspace/$OUT/spike3.json" || failure_count=$((failure_count + 1))
fi

printf '== Report ==\n'
"${compose_command[@]}" run --rm report \
  --target "$deployment_target" \
  --dir "/workspace/$OUT" \
  --md "/workspace/$OUT/$report_name" \
  --patch "/workspace/$OUT/$patch_name" || failure_count=$((failure_count + 1))

if [ "$failure_count" -gt 0 ]; then
  printf '%s spike container command(s) failed.\n' "$failure_count" >&2
  exit 1
fi
