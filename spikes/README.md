# Phase 0 — 검증 스파이크

**이 세 가지를 실기에서 돌리고 결과를 보고한 뒤 멈춘다.** 결과에 따라 이후
아키텍처가 갈라진다.

macOS 개발 PC에서는 실행할 수 없다. aarch64 + CUDA + Orin 대역폭이 있어야
의미 있는 수치가 나온다. 여기 있는 것은 실기에 그대로 올려 돌릴 스크립트다.

## 준비

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks       # MAXN + 클럭 고정
jtop                                            # 스로틀링 확인 (별도 터미널)
```

측정 중 `jtop` 에서 열 스로틀링이 걸리면 그 측정치는 버린다.

## Spike 1 — vLLM 이 Qwen3-ASR 을 로드하는가

```bash
jetson-containers run ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04
# 컨테이너 안에서
python3 spikes/spike1_asr_load.py --wav samples/ko_6s.wav --only vllm --out spikes/out/spike1_vllm.json
```

transformers 경로는 별도 컨테이너에서:

```bash
python3 spikes/spike1_asr_load.py --wav samples/ko_6s.wav --only transformers --out spikes/out/spike1.json
```

두 결과를 하나로 합칠 때는 `spike1.json` 에 `vllm` 키를 채워 넣는다.

확인 항목: 로드 성공 / 6초 전사 시간 / **N-best 5 출력 가능 여부** / 로그확률 획득 여부.
N-best 가 안 나오면 §5.2 를 만족하지 못하므로 그 경로는 채택할 수 없다.

## Spike 2 — flash-attn 이 sm_87 에서 빌드되는가

```bash
jetson-containers run $(autotag flash-attention)
python3 spikes/spike2_flash_attn.py --out spikes/out/spike2.json
```

import 성공만으로 판정하지 않는다. 실제 커널을 한 번 돌려서 확인한다.
Qwen3-TTS 는 `flash_attention_2` → `sdpa` → `eager` 순으로 시도해, flash-attn
없이도 뜨는지를 함께 본다. 전부 실패하면 MeloTTS 로 Phase 1~3 을 시작한다.

## Spike 3 — MoE vs 밀집 14B 실측 tok/s

GGUF 를 먼저 받아둔다 (`scripts/fetch_models.sh`). 합쳐서 약 28GB.

```bash
python3 spikes/spike3_llm_tokrate.py \
    --bin /opt/llama.cpp/build/bin \
    --models-dir ./models/gguf \
    --out spikes/out/spike3.json
```

조건은 컨텍스트 2048, 배치 1, 출력 60토큰. `llama-bench` 의 순수 생성 속도와,
실제 번역 프롬프트로 `llama-server` 를 때린 값을 **둘 다** 잰다. 후자가 우리가
실제로 겪을 값이다 — 프롬프트 처리 시간이 §6 의 '첫 절 0.7초'에 들어간다.

**판정: 5 tok/s 미만이면 밀집 14B로 회귀.**

## 보고서 생성

```bash
python3 spikes/report.py --dir spikes/out --md spikes/out/PHASE0.md --patch spikes/out/local.yaml
cp spikes/out/local.yaml config/local.yaml     # 결론을 설정에 반영
```

`config/local.yaml` 은 `config/default.yaml` 위에 자동으로 덮어써진다.
Phase 0 의 결론이 코드 수정이 아니라 설정 변경으로 끝나도록 만들어 두었다.
