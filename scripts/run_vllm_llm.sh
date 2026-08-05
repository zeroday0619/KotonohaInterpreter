#!/usr/bin/env bash
# Start the resident vLLM translation server with an offline model snapshot.
set -euo pipefail

MANAGED_ENVIRONMENT=${KOTONOHA_LLM_CONFIG_ENV:-}
if [ -n "$MANAGED_ENVIRONMENT" ] && [ -f "$MANAGED_ENVIRONMENT" ]; then
  # The administration service writes validated and shell-quoted values.
  # shellcheck disable=SC1090
  source "$MANAGED_ENVIRONMENT"
fi

PROFILE=${LLM_PROFILE:-dense}
MODEL_DIRECTORY=${LLM_MODELS_DIR:-/models/llm}
PORT=${LLM_PORT:-8003}
MAX_MODEL_LENGTH=${LLM_MAX_MODEL_LEN:-2048}
GPU_MEMORY_UTILIZATION=${LLM_GPU_MEMORY_UTILIZATION:-0.55}
MAX_NUM_SEQUENCES=${LLM_MAX_NUM_SEQS:-1}
ENFORCE_EAGER=${LLM_ENFORCE_EAGER:-1}
SERVED_MODEL_NAME=${LLM_SERVED_MODEL_NAME:-kotonoha-translation}

if [ -n "${LLM_MODEL:-}" ]; then
  MODEL=$LLM_MODEL
else
  case "$PROFILE" in
    moe) MODEL="$MODEL_DIRECTORY/Qwen3-30B-A3B-Instruct-2507-AWQ" ;;
    dense) MODEL="$MODEL_DIRECTORY/Qwen3-14B-AWQ" ;;
    *) echo "LLM_PROFILE must be moe or dense"; exit 1 ;;
  esac
fi

[ -s "$MODEL/config.json" ] || { echo "vLLM model snapshot is incomplete: $MODEL"; exit 1; }
command -v vllm >/dev/null 2>&1 || { echo "vllm executable is not available"; exit 1; }

arguments=(
  serve "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --max-model-len "$MAX_MODEL_LENGTH"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQUENCES"
  --dtype "${LLM_DTYPE:-half}"
  --quantization "${LLM_QUANTIZATION:-awq}"
  --default-chat-template-kwargs '{"enable_thinking":false}'
)

if [ "$ENFORCE_EAGER" = "1" ]; then
  arguments+=(--enforce-eager)
else
  arguments+=(--no-enforce-eager)
fi
if [ -n "${KOTONOHA_SERVICE_TOKEN:-}" ]; then
  arguments+=(--api-key "$KOTONOHA_SERVICE_TOKEN")
fi

echo "vLLM translation server  profile=$PROFILE  model=$MODEL  port=$PORT"
exec vllm "${arguments[@]}"
