#!/usr/bin/env bash
# 모델 내려받기. 완전 오프라인 동작이 목표이므로 실사용 전에 전부 로컬에 둔다.
#
# 저장소 ID 는 2026-08 시점에 실제 조회로 확인한 것이다.
#   Qwen/Qwen3-ASR-1.7B-hf                     transformers 포맷
#   Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice       프리셋 음색 9종
#   unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF   Q4_K_M ≈ 18.6GB
#   unsloth/Qwen3-14B-GGUF                     Q4_K_M ≈ 9GB
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS_DIR:-./models}
GGUF="$MODELS/gguf"
mkdir -p "$GGUF"

need() { command -v "$1" >/dev/null || { echo "필요: $1"; exit 1; }; }
need curl

# --- Silero VAD (ONNX, 약 2MB) ---
VAD="$MODELS/silero_vad.onnx"
if [ ! -f "$VAD" ]; then
  echo "== silero_vad.onnx =="
  curl -fL -o "$VAD" \
    https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx
fi

# --- HF 저장소 ---
# uv 가 있으면 uvx 로 일회성 실행한다. 프로젝트 의존성에 huggingface_hub 를
# 넣을 이유가 없다 — 런타임에는 쓰지 않고 모델 내려받을 때만 필요하다.
if command -v uvx >/dev/null; then HFCLI="uvx --from huggingface_hub hf download"
elif command -v hf >/dev/null; then HFCLI="hf download"
elif command -v huggingface-cli >/dev/null; then HFCLI="huggingface-cli download"
else echo "uv 또는 huggingface_hub CLI 필요"; exit 1; fi

echo "== Qwen3-ASR 1.7B =="
$HFCLI Qwen/Qwen3-ASR-1.7B-hf --local-dir "$MODELS/Qwen3-ASR-1.7B-hf"

echo "== Qwen3-TTS 0.6B (Spike 2 결과에 따라 사용) =="
$HFCLI Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir "$MODELS/Qwen3-TTS-0.6B" || \
  echo "  건너뜀 — MeloTTS 로 시작하는 경우 필요 없다"

echo "== faster-whisper large-v3 =="
$HFCLI Systran/faster-whisper-large-v3 --local-dir "$MODELS/faster-whisper-large-v3"

echo "== GGUF (Spike 3 에서 둘 다 필요) =="
$HFCLI unsloth/Qwen3-14B-GGUF Qwen3-14B-Q4_K_M.gguf --local-dir "$GGUF"
$HFCLI unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
       Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --local-dir "$GGUF"

echo
echo "완료. 용량:"
du -sh "$MODELS"/* 2>/dev/null || true
