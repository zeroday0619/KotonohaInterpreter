#!/usr/bin/env bash
# Phase 0 스파이크 일괄 실행. Jetson AGX Orin 에서만 의미가 있다.
#
# 컨테이너가 서로 다르므로 이 스크립트는 "현재 컨테이너에서 가능한 것"만 돌린다.
# 세 개를 각각 맞는 이미지에서 돌린 뒤 spikes/out 에 결과를 모으고 report.py 를 부른다.
set -uo pipefail

cd "$(dirname "$0")/.."
OUT=spikes/out
mkdir -p "$OUT"

echo "== 환경 =="
uname -m
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,
      'cap',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)" 2>/dev/null \
  || echo "torch 없음"

if command -v tegrastats >/dev/null; then
  echo "== tegrastats 5초 =="
  timeout 5 tegrastats || true
fi

WAV="${WAV:-samples/ko_6s.wav}"
[ -f "$WAV" ] || { echo "!! 실 녹음 $WAV 없음 — 합성 오디오로 타이밍만 잰다"; WAV=""; }
WAVARG=""; [ -n "$WAV" ] && WAVARG="--wav $WAV"

echo "== Spike 1 =="
python3 spikes/spike1_asr_load.py $WAVARG --out "$OUT/spike1.json" || echo "spike1 실패"

echo "== Spike 2 =="
python3 spikes/spike2_flash_attn.py --out "$OUT/spike2.json" || echo "spike2 실패"

echo "== Spike 3 =="
LLAMA_BIN="${LLAMA_BIN:-/opt/llama.cpp/build/bin}"
if [ -d "$LLAMA_BIN" ]; then
  python3 spikes/spike3_llm_tokrate.py --bin "$LLAMA_BIN" \
      --models-dir "${MODELS_DIR:-./models/gguf}" --out "$OUT/spike3.json" || echo "spike3 실패"
else
  echo "llama.cpp bin 없음: $LLAMA_BIN (LLAMA_BIN 환경변수로 지정)"
fi

echo "== 보고서 =="
python3 spikes/report.py --dir "$OUT" --md "$OUT/PHASE0.md" --patch "$OUT/local.yaml"
