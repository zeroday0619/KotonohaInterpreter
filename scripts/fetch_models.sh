#!/usr/bin/env bash
# Fetch the models. The device runs fully offline, so everything has to be
# local before it is used in anger.
#
# Repository IDs were confirmed by lookup in 2026-08:
#   Qwen/Qwen3-ASR-1.7B                        vLLM format
#   Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice       nine preset timbres
#   unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF   Q4_K_M ~ 18.6 GB
#   unsloth/Qwen3-14B-GGUF                     Q4_K_M ~ 9 GB
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS_DIR:-./models}
GGUF="$MODELS/gguf"
mkdir -p "$GGUF"

need() { command -v "$1" >/dev/null || { echo "required: $1"; exit 1; }; }
need curl

# --- Silero VAD (ONNX, roughly 2 MB) ---
VAD="$MODELS/silero_vad.onnx"
if [ ! -f "$VAD" ]; then
  echo "== silero_vad.onnx =="
  curl -fL -o "$VAD" \
    https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx
fi

# --- Hugging Face repositories ---
# Prefer a one-shot uvx run when uv is available. There is no reason to make
# huggingface_hub a project dependency: it is never used at runtime, only here.
if command -v uvx >/dev/null; then HFCLI="uvx --from huggingface_hub hf download"
elif command -v hf >/dev/null; then HFCLI="hf download"
elif command -v huggingface-cli >/dev/null; then HFCLI="huggingface-cli download"
else echo "needs uv, or the huggingface_hub CLI"; exit 1; fi

echo "== Qwen3-ASR 1.7B =="
$HFCLI Qwen/Qwen3-ASR-1.7B --local-dir "$MODELS/Qwen3-ASR-1.7B"

echo "== Qwen3-ASR 1.7B Transformers fallback =="
$HFCLI Qwen/Qwen3-ASR-1.7B-hf --local-dir "$MODELS/Qwen3-ASR-1.7B-hf"

echo "== Qwen3-TTS 0.6B (used depending on the Spike 2 result) =="
$HFCLI Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir "$MODELS/Qwen3-TTS-0.6B" || \
  echo "  skipped — not needed if we start on MeloTTS"

echo "== faster-whisper large-v3 =="
$HFCLI Systran/faster-whisper-large-v3 --local-dir "$MODELS/faster-whisper-large-v3"

echo "== GGUF (Spike 3 needs both) =="
$HFCLI unsloth/Qwen3-14B-GGUF Qwen3-14B-Q4_K_M.gguf --local-dir "$GGUF"
$HFCLI unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
       Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --local-dir "$GGUF"

echo
echo "done. sizes:"
du -sh "$MODELS"/* 2>/dev/null || true
