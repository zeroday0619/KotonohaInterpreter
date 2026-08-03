#!/usr/bin/env bash
# 번역 LLM 상주 서버 (llama.cpp, OpenAI 호환).
#
# 프로필은 config 의 llm.profile 을 따른다. Spike 3 전에는 dense(14B)가 기본이다.
# 매 요청마다 모델을 로드하면 지연 예산이 무너지므로(§12) 이 프로세스는 계속 떠 있는다.
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
  *) echo "LLM_PROFILE 은 moe 또는 dense"; exit 1 ;;
esac

[ -f "$MODEL" ] || { echo "GGUF 없음: $MODEL — scripts/fetch_models.sh 먼저"; exit 1; }

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
