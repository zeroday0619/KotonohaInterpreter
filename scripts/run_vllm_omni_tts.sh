#!/usr/bin/env bash
# Start the resident vLLM-Omni Qwen3-TTS Speech API from an offline snapshot.
set -euo pipefail

MODEL=${TTS_MODEL:-/models/Qwen3-TTS-0.6B}
PORT=${TTS_PORT:-8004}
SERVED_MODEL_NAME=${TTS_SERVED_MODEL_NAME:-kotonoha-tts}
GPU_MEMORY_UTILIZATION=${TTS_GPU_MEMORY_UTILIZATION:-0.25}
ENFORCE_EAGER=${TTS_ENFORCE_EAGER:-1}

[ -s "$MODEL/config.json" ] || {
  echo "vLLM-Omni TTS model snapshot is incomplete: $MODEL"
  exit 1
}
command -v vllm >/dev/null 2>&1 || {
  echo "vLLM executable is not available"
  exit 1
}

arguments=(
  serve "$MODEL"
  --omni
  --host 0.0.0.0
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --stage-overrides '{"1":{"max_num_seqs":1}}'
)

if [ "$ENFORCE_EAGER" = "1" ]; then
  arguments+=(--enforce-eager)
else
  arguments+=(--no-enforce-eager)
fi
if [ -n "${KOTONOHA_SERVICE_TOKEN:-}" ]; then
  arguments+=(--api-key "$KOTONOHA_SERVICE_TOKEN")
else
  echo "auth.disabled service=tts"
fi

echo "vLLM-Omni TTS server  model=$MODEL  served_name=$SERVED_MODEL_NAME  port=$PORT"
exec vllm "${arguments[@]}"
