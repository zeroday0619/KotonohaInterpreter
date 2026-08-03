#!/usr/bin/env bash
# Resident translation LLM server (llama.cpp, OpenAI-compatible).
#
# The profile follows llm.profile in the config; before Spike 3 the default is
# dense (14B). Loading the model per request would destroy the latency budget
# (§12), so this process stays up.
set -euo pipefail
cd "$(dirname "$0")/.."

# The authenticated management API writes this file. Sourcing it after Compose has
# populated the environment allows managed settings to override deployment defaults.
MANAGED_ENV=${KOTONOHA_LLM_CONFIG_ENV:-./config/remote-llm.env}
if [ -f "$MANAGED_ENV" ]; then
  # The file is generated with shell-quoted scalar assignments only.
  # shellcheck disable=SC1090
  source "$MANAGED_ENV"
fi

PROFILE=${LLM_PROFILE:-dense}
BINARY_DIRECTORY=${LLAMA_BIN:-/opt/llama.cpp/build/bin}
MODEL_DIRECTORY=${MODELS_DIR:-./models/gguf}
PORT=${LLM_PORT:-8003}
CONTEXT_SIZE=${LLM_CTX:-2048}
BATCH_SIZE=${LLM_BATCH:-512}
GPU_LAYERS=${LLM_NGL:-999}
SERVER_BINARY="$BINARY_DIRECTORY/llama-server"

if [ -n "${LLM_MODEL:-}" ]; then
  MODEL=$LLM_MODEL
else
  case "$PROFILE" in
    moe)   MODEL="$MODEL_DIRECTORY/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf" ;;
    dense) MODEL="$MODEL_DIRECTORY/Qwen3-14B-Q4_K_M.gguf" ;;
    *) echo "LLM_PROFILE must be moe or dense"; exit 1 ;;
  esac
fi

[ -f "$MODEL" ] || { echo "GGUF missing: $MODEL — run scripts/fetch_models.sh first"; exit 1; }
[ -x "$SERVER_BINARY" ] || { echo "llama-server is not executable: $SERVER_BINARY"; exit 1; }

# Recent llama.cpp images split server code into shared libraries beside the
# executable. ELF loaders do not search that directory unless the binary has a
# matching RUNPATH, so preserve existing paths while making the image directory
# explicit.
export LD_LIBRARY_PATH="$BINARY_DIRECTORY${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if command -v ldd >/dev/null 2>&1; then
  MISSING_LIBRARIES=$(ldd "$SERVER_BINARY" 2>/dev/null | awk '/not found/ { print $1 }')
  if [ -n "$MISSING_LIBRARIES" ]; then
    echo "llama-server shared libraries are missing from LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "$MISSING_LIBRARIES"
    exit 1
  fi
fi

echo "llama-server  profile=$PROFILE  ctx=$CONTEXT_SIZE  batch=$BATCH_SIZE  port=$PORT"
exec "$SERVER_BINARY" \
  -m "$MODEL" \
  -c "$CONTEXT_SIZE" \
  -b "$BATCH_SIZE" \
  -ngl "$GPU_LAYERS" \
  -np 1 \
  --host 0.0.0.0 --port "$PORT" \
  --no-webui \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn auto
