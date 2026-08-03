#!/usr/bin/env bash
# Run the Phase 0 spikes. Only meaningful on a Jetson AGX Orin.
#
# The spikes live in different containers, so this script runs whatever the
# current container can. Run each one in its proper image, collect the results
# in spikes/out, then call report.py.
set -uo pipefail

cd "$(dirname "$0")/.."
OUT=spikes/out
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
WAVARG=""; [ -n "$WAV" ] && WAVARG="--wav $WAV"

echo "== Spike 1 =="
python3 spikes/spike1_asr_load.py $WAVARG --out "$OUT/spike1.json" || echo "spike1 failed"

echo "== Spike 2 =="
python3 spikes/spike2_flash_attn.py --out "$OUT/spike2.json" || echo "spike2 failed"

echo "== Spike 3 =="
LLAMA_BIN="${LLAMA_BIN:-/opt/llama.cpp/build/bin}"
if [ -d "$LLAMA_BIN" ]; then
  python3 spikes/spike3_llm_tokrate.py --bin "$LLAMA_BIN" \
      --models-dir "${MODELS_DIR:-./models/gguf}" --out "$OUT/spike3.json" || echo "spike3 failed"
else
  echo "llama.cpp bin not found: $LLAMA_BIN (set LLAMA_BIN)"
fi

echo "== report =="
python3 spikes/report.py --dir "$OUT" --md "$OUT/PHASE0.md" --patch "$OUT/local.yaml"
