#!/usr/bin/env bash
# Run hardware spikes on the Jetson AGX Orin or the external RTX A6000.
#
# The spikes live in different containers, so this script runs whatever the
# current container can. Results remain separated by target.
set -uo pipefail

cd "$(dirname "$0")/.."
TARGET="${1:-${TARGET:-jetson}}"
case "$TARGET" in
  jetson|a6000) ;;
  *) echo "target must be jetson or a6000" >&2; exit 2 ;;
esac

if [ -z "${OUT:-}" ]; then
  if [ "$TARGET" = "jetson" ]; then
    OUT=spikes/out
  else
    OUT=spikes/out/a6000
  fi
fi
mkdir -p "$OUT"

echo "== environment =="
uname -m
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,
      'cap',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)" 2>/dev/null \
  || echo "no torch"

if command -v tegrastats >/dev/null; then
  echo "== tegrastats, 5s =="
  timeout 5 tegrastats || true
fi

WAV="${WAV:-samples/ko_6s.wav}"
[ -f "$WAV" ] || { echo "!! no real recording at $WAV — timing only, with synthetic audio"; WAV=""; }
WAV_ARGUMENTS=()
[ -n "$WAV" ] && WAV_ARGUMENTS=(--wav "$WAV")

VLLM_ARGUMENTS=(
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
  --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}"
)
if [ "${VLLM_ENFORCE_EAGER:-1}" = "0" ]; then
  VLLM_ARGUMENTS+=(--no-enforce-eager)
else
  VLLM_ARGUMENTS+=(--enforce-eager)
fi
ASR_ARGUMENTS=()
if [ -n "${ASR_ONLY:-}" ]; then
  ASR_ARGUMENTS=(--only "$ASR_ONLY")
elif [ "$TARGET" = "a6000" ]; then
  ASR_ARGUMENTS=(--only vllm)
fi

echo "== Spike 1 =="
python3 spikes/spike1_asr_load.py --target "$TARGET" "${WAV_ARGUMENTS[@]}" \
  "${VLLM_ARGUMENTS[@]}" "${ASR_ARGUMENTS[@]}" \
  --out "$OUT/spike1.json" || echo "spike1 failed"

echo "== Spike 2 =="
TTS_ARGUMENTS=()
[ "$TARGET" = "a6000" ] && TTS_ARGUMENTS=(--skip-melo)
python3 spikes/spike2_flash_attn.py --target "$TARGET" "${TTS_ARGUMENTS[@]}" \
  --out "$OUT/spike2.json" || echo "spike2 failed"

echo "== Spike 3 =="
LLAMA_BIN="${LLAMA_BIN:-/opt/llama.cpp/build/bin}"
if [ -d "$LLAMA_BIN" ]; then
  if [ "$TARGET" = "a6000" ]; then
    DEFAULT_CONTEXT=4096
  else
    DEFAULT_CONTEXT=2048
  fi
  python3 spikes/spike3_llm_tokrate.py --target "$TARGET" --bin "$LLAMA_BIN" \
      --models-dir "${MODELS_DIR:-./models/gguf}" \
      --context "${LLM_CONTEXT:-$DEFAULT_CONTEXT}" \
      --output-tokens "${LLM_OUTPUT_TOKENS:-60}" \
      --runs "${BENCHMARK_RUNS:-3}" --out "$OUT/spike3.json" || echo "spike3 failed"
else
  echo "llama.cpp bin not found: $LLAMA_BIN (set LLAMA_BIN)"
fi

echo "== report =="
if [ "$TARGET" = "a6000" ]; then
  REPORT_NAME=PERFORMANCE.md
  PATCH_NAME=remote-server.local.yaml
else
  REPORT_NAME=PHASE0.md
  PATCH_NAME=local.yaml
fi
python3 spikes/report.py --target "$TARGET" --dir "$OUT" \
  --md "$OUT/$REPORT_NAME" --patch "$OUT/$PATCH_NAME"
