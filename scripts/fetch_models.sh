#!/usr/bin/env bash
# Fetch the models. The device runs fully offline, so everything has to be
# local before it is used in anger.
#
# Repository IDs were confirmed by lookup in 2026-08:
#   Qwen/Qwen3-ASR-0.6B                        Jetson vLLM ASR
#   mistralai/Voxtral-Mini-4B-Realtime-2602    A6000 vLLM realtime ASR
#   Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice       nine preset timbres
#   google/translategemma-4b-it                 Jetson translation model
#   google/translategemma-12b-it                A6000 translation model
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS_DIR:-./models}
LLM_MODELS="$MODELS/llm"
SILERO_VAD_REVISION=bfdc0193023f121ea5b3cc7b176dbed570a68a59
QWEN_ASR_VLLM_REVISION=5eb144179a02acc5e5ba31e748d22b0cf3e303b0
QWEN_ASR_TRANSFORMERS_REVISION=7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c
VOXTRAL_REVISION=2769294da9567371363522aac9bbcfdd19447add
QWEN_TTS_REVISION=85e237c12c027371202489a0ec509ded67b5e4b5
FASTER_WHISPER_REVISION=edaa852ec7e145841d8ffdb056a99866b5f0a478
TRANSLATEGEMMA_4B_REVISION=10042cb0e6e7fdce748996a71dc3dc432a4e0c89
TRANSLATEGEMMA_12B_REVISION=d1b225e1caa17f1ddc7e62065d8637d0923f34e2
mkdir -p "$LLM_MODELS"

need() { command -v "$1" >/dev/null || { echo "required: $1"; exit 1; }; }
need curl

# --- Silero VAD (ONNX, roughly 2 MB) ---
VAD="$MODELS/silero_vad.onnx"
if [ ! -f "$VAD" ]; then
  echo "== silero_vad.onnx =="
  curl -fL -o "$VAD" \
    "https://raw.githubusercontent.com/snakers4/silero-vad/$SILERO_VAD_REVISION/src/silero_vad/data/silero_vad.onnx"
fi

# --- Hugging Face repositories ---
# Prefer a one-shot uvx run when uv is available. There is no reason to make
# huggingface_hub a project dependency: it is never used at runtime, only here.
if command -v uvx >/dev/null; then HFCLI="uvx --from huggingface_hub hf download"
elif command -v hf >/dev/null; then HFCLI="hf download"
elif command -v huggingface-cli >/dev/null; then HFCLI="huggingface-cli download"
else echo "needs uv, or the huggingface_hub CLI"; exit 1; fi

echo "== Qwen3-ASR 0.6B =="
$HFCLI Qwen/Qwen3-ASR-0.6B \
  --revision "$QWEN_ASR_VLLM_REVISION" \
  --local-dir "$MODELS/Qwen3-ASR-0.6B"

echo "== Qwen3-ASR 0.6B Transformers fallback =="
$HFCLI Qwen/Qwen3-ASR-0.6B-hf \
  --revision "$QWEN_ASR_TRANSFORMERS_REVISION" \
  --local-dir "$MODELS/Qwen3-ASR-0.6B-hf"

echo "== Voxtral Mini 4B Realtime =="
$HFCLI mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --revision "$VOXTRAL_REVISION" \
  --local-dir "$MODELS/Voxtral-Mini-4B-Realtime-2602"

echo "== Qwen3-TTS 0.6B for vLLM-Omni =="
$HFCLI Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --revision "$QWEN_TTS_REVISION" \
  --local-dir "$MODELS/Qwen3-TTS-0.6B"

echo "== faster-whisper large-v3 =="
$HFCLI Systran/faster-whisper-large-v3 \
  --revision "$FASTER_WHISPER_REVISION" \
  --local-dir "$MODELS/faster-whisper-large-v3"

echo "== TranslateGemma 4B IT (Jetson) =="
echo "The Hugging Face account must accept the TranslateGemma license before download."
$HFCLI google/translategemma-4b-it \
  --revision "$TRANSLATEGEMMA_4B_REVISION" \
  --local-dir "$LLM_MODELS/translategemma-4b-it"

echo "== TranslateGemma 12B IT (A6000) =="
$HFCLI google/translategemma-12b-it \
  --revision "$TRANSLATEGEMMA_12B_REVISION" \
  --local-dir "$LLM_MODELS/translategemma-12b-it"

echo
echo "done. sizes:"
du -sh "$MODELS"/* 2>/dev/null || true
