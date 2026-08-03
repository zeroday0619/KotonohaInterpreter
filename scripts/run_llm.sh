#!/usr/bin/env bash
# Resident translation LLM server (llama.cpp, OpenAI-compatible).
#
# The profile follows llm.profile in the config; before Spike 3 the default is
# dense (14B). Loading the model per request would destroy the latency budget
# (§12), so this process stays up.
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE=${LLM_PROFILE:-dense}
BIN=${LLAMA_BIN:-/opt/llama.cpp/build/bin}
GGUF_DIR=${MODELS_DIR:-./models/gguf}
PORT=${LLM_PORT:-8003}
NCTX=${LLM_CTX:-2048}
NGL=${LLM_NGL:-999}

case "$PROFILE" in
  moe)   MODEL="$GGUF_DIR/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf" ;;
  dense) MODEL="$GGUF_DIR/Qwen3-14B-Q4_K_M.gguf" ;;
  *) echo "LLM_PROFILE must be moe or dense"; exit 1 ;;
esac

[ -f "$MODEL" ] || { echo "GGUF missing: $MODEL — run scripts/fetch_models.sh first"; exit 1; }

echo "llama-server  profile=$PROFILE  ctx=$NCTX  port=$PORT"
exec "$BIN/llama-server" \
  -m "$MODEL" \
  -c "$NCTX" \
  -ngl "$NGL" \
  -np 1 \
  --host 0.0.0.0 --port "$PORT" \
  --no-webui \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn auto
