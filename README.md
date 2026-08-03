# Kotonoha Interpreter

순차식(consecutive) 4언어 오프라인 음성 통역기. NVIDIA Jetson AGX Orin 64GB 전용.

한 사람이 한 발화를 마치면 기기가 그것을 다른 언어로 말한다. 동시통역이 아니다.
클라우드 호출은 없다. 정확도가 지연보다 우선하고, 발화 종료 후 첫 음성까지 약 3초를
목표로 한다.

| | |
|---|---|
| 언어 | 한국어 · English · 繁體中文(臺灣) · 日本語 — 12방향 |
| 소스 언어 | 자동 판별(LID) + 짧은 발화 승계 폴백 |
| 타깃 언어 | 세션 설정 (2언어 페어 / 고정 타깃 / 브로드캐스트) |
| 실행 환경 | JetPack 6.2 (L4T r36.4.x), CUDA 12.6, aarch64 |
| UI | TUI (Textual) |

---

## 지금 상태

**Phase 0 (검증 스파이크) 미실행.** 스파이크는 Jetson 실기에서만 의미가 있고,
그 결과에 따라 세 가지가 갈린다.

| 갈림길 | 결정할 설정 | 스파이크 |
|---|---|---|
| ASR 을 vLLM 으로 돌리는가 | `asr.backend` | Spike 1 |
| Qwen3-TTS 를 띄울 수 있는가 | `tts.backend` | Spike 2 |
| MoE 를 쓰는가 밀집 14B 로 가는가 | `llm.profile` | Spike 3 |

세 갈림길을 **코드가 아니라 설정**으로 처리했다. 스파이크 결과를
`spikes/report.py` 에 넣으면 `config/local.yaml` 이 나오고, 그걸 복사하면 끝난다.
결과가 나오기 전 기본값은 보수적인 쪽(`transformers` / `melo` / `dense`)이다.

추측으로 채우지 않은 곳이 하나 있다. `services/asr_server.py` 의 `VllmBackend` 는
명시적으로 `NotImplementedError` 를 던진다. vLLM 이 Qwen3-ASR 을 어떤 형태로 로드하고
N-best 를 내는지가 Spike 1 의 질문 그 자체이므로, 답이 나오기 전에 구현하면 지연 예산
계산이 통째로 틀어진다.

개발 PC(macOS)에서 확인한 것: 단위 테스트 32개 통과, ruff 클린, 목 서비스를 붙인
전 구간 통합 스모크(다섯 지점 계측 · 상태 전이 · 절 스트리밍 · 번역 저장)까지 동작.

---

## 구조

```
[마이크 48k]
   ↓ sounddevice
오디오 프런트엔드 (CPU)        audio/capture.py  denoise.py  vad.py
  · DeepFilterNet3 (48k)
  · Silero VAD + 프리롤 300ms  ← 타협 불가
  · EOU = 침묵 800ms
   ↓ 공유메모리 링버퍼          shmring.py
오케스트레이터 (asyncio)        core/orchestrator.py
  · 상태기계 · 언어 라우팅 · 품질 게이트
   ↓
:8001 asr          Qwen3-ASR 1.7B, N-best 5 + LID
:8002 asr-verify   faster-whisper large-v3  (조건부)
:8003 llm          llama.cpp server, 정정+번역 단일 패스
:8004 tts          Qwen3-TTS 0.6B / MeloTTS
   ↓
[스피커 + TUI]
```

오디오는 HTTP body 에 태우지 않는다. 공유 메모리 링버퍼(`shmring.py`)에 쓰고
`{name, slot, seq, frames}` 짜리 참조만 JSON 으로 넘긴다. 6초 PCM 을 base64 로
왕복시키면 턴마다 100~200ms 가 그냥 사라진다.

### 타협하지 않은 다섯 가지

1. **VAD 프리롤 300ms** — `audio/vad.py`. 없으면 한국어 초성 파열음과 일본어 촉음의
   첫 음절이 잘리고, 그러면 ASR 품질 문제로 오진하게 된다. 회귀 테스트로 고정해 뒀다
   (`tests/test_vad_segmenter.py`).
2. **ASR N-best 5** — 순차식이므로 그리디를 쓸 이유가 없다.
3. **정정과 번역을 단일 LLM 패스로** — `prompts/translate.py`. N-best + 히스토리 +
   용어집을 함께 주고 "문맥으로 정정한 뒤 번역"까지 한 번에 시킨다. 단계를 나누면
   정정 단계가 만든 오류를 번역 단계가 확대한다.
4. **절 단위 스트리밍 핸드오프** — `core/clauses.py`. LLM 출력이 끝나기를 기다리지
   않고 절이 완성되는 즉시 TTS 로 넘긴다. 성립 조건은 5 tok/s 이상.
5. **조건부 교차 검증** — `core/quality.py`. 평균 로그확률 미달 · N-best 불일치 ·
   1초 미만 발화일 때만 Whisper 를 부른다. 상시 호출하면 매 턴 0.8초가 붙는다.

### 반이중 게이팅

`SPEAKING` 진입 시 마이크를 닫는다. TTS 출력이 마이크로 되돌아가 새 발화로 인식되면
무한 루프가 된다. 게이팅은 `Orchestrator._on_state_change` **한 군데에서만** 하고,
`MicCapture.close_gate()` 는 큐에 남은 잔향까지 버린 뒤 리샘플러·VAD 상태를 리셋한다.

---

## 설치

패키지 관리는 [uv](https://docs.astral.sh/uv/)로 한다. `uv.lock` 을 커밋해 두었으므로
개발 PC와 실기가 같은 버전을 쓴다.

### 개발 PC (macOS / Linux)

모델 없이 프런트엔드·상태기계·프롬프트·계측을 돌려볼 수 있다.

```bash
uv sync                     # .python-version(3.12) 기준으로 venv 생성 + dev 그룹까지
uv run pytest -q
uv run ruff check .
uv run kotonoha doctor
```

`uv sync` 는 `dev` 그룹까지만 넣는다. 평가 도구(COMET 등)는 무거우니 필요할 때만:

```bash
uv sync --group eval
```

`device` extra 는 macOS 에서 설치하지 말 것 (aarch64/CUDA 전용). 잠금 파일의 대상
환경도 `darwin-arm64` 와 `linux-aarch64` 둘로 제한해 두었다 — x86 휠이 섞여
들어오지 않는다.

의존성을 바꿨다면:

```bash
uv add <pkg>                # 런타임
uv add --group dev <pkg>    # 개발 도구
uv lock --upgrade-package <pkg>
```

### Jetson AGX Orin

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks     # MAXN + 클럭 고정
bash scripts/fetch_models.sh                  # 모델 전량 로컬로 (약 40GB)
docker compose -f docker/compose.yaml up -d asr asr-verify llm tts
docker compose -f docker/compose.yaml run --rm orchestrator
```

베이스 이미지 태그는 `r36.4.0` 계열로 고정한다. 검증된 조합이므로 임의로 올리지
않는다. 실제 존재하는 태그는 `jetson-containers` 의 `autotag` 로 확인해 `.env` 에 적는다.

컨테이너 안에서는 `uv sync` 로 별도 venv 를 만들지 않는다. 베이스 이미지의 시스템
파이썬에 `uv pip install` 로 얹는다(`UV_SYSTEM_PYTHON=1`). venv 를 만들면 이미지에
들어 있는 CUDA 빌드 torch 가 가려지기 때문이다. 런타임 의존성은
`uv export --frozen` 으로 잠금 파일에서 뽑아 고정 설치하고, aarch64 해석이 개발 PC 와
다른 패키지(`onnxruntime`, `deepfilternet`, `qwen-tts`, `melotts`)만 잠금 밖에서
best-effort 로 설치한다. 각 이미지는 빌드 마지막에 `torch.version.cuda` 를 찍어,
그 설치가 CUDA 빌드를 PyPI CPU 빌드로 덮어썼는지 그 자리에서 드러나게 해 두었다.

---

## 사용

```bash
uv run kotonoha run                     # TUI
uv run kotonoha doctor                  # 환경·서비스 점검
uv run kotonoha devices                 # 오디오 장치 목록
uv run kotonoha replay foo.wav          # 마이크 없이 WAV 로 전 구간 재생 (EOU 회귀 확인)
uv run kotonoha glossary import config/glossary.seed.yaml
uv run kotonoha serve asr               # 개별 서비스 기동 (도커 없이)
```

컨테이너 안에서는 시스템 파이썬에 설치돼 있으므로 `uv run` 없이 `kotonoha ...` 로 쓴다.

TUI 키: `space` 말하기(토글) · `a` PTT/자동 · `r` 라우팅 · `c` 지우기 · `q` 종료.

터미널은 키를 뗀 이벤트를 주지 않으므로 push-to-talk 은 토글이다. PTT 라도 프리롤은
살아 있다 — 사람은 키보다 먼저 말하기 시작한다.

---

## 설정

`config/default.yaml` 이 기본이고, 같은 디렉터리에 `config/local.yaml` 이 있으면 그
위에 덮어쓴다(기기별 장치 인덱스, Phase 0 결과 등). 환경변수가 가장 세다.

```bash
KOTONOHA__LLM__PROFILE=moe KOTONOHA__ASR__N_BEST=3 kotonoha run
```

---

## 계측

매 턴 다섯 지점을 찍어 `data/logs/turns.jsonl` 에 한 줄로 남긴다.

```
EOU 감지 → ASR 완료 → 첫 절 → 첫 오디오 패킷 → 큐 소진
```

함께: 판정 언어와 그 출처(lid/inherited), LID 신뢰도, ASR 평균 로그확률,
교차 검증 발동 여부, 입력 오디오 길이, 출력 토큰 수, tok/s, 예산 초과 단계.

예산을 넘긴 단계는 `over_budget_ms` 에 어느 단계가 얼마나 넘겼는지로 나온다.
TUI 하단에도 실측/예산이 같이 뜬다.

애플리케이션 로그는 `data/logs/kotonoha.jsonl` 로 분리돼 있다 — 섞으면 턴 로그를
그대로 파싱할 수 없다.

| 단계 | 목표 |
|---|---|
| 침묵 대기 | 0.8초 |
| 프런트엔드 | 0.1초 |
| ASR (N-best 5) | 0.9초 |
| 교차 검증 (조건부 평균) | 0.1초 |
| 정정 + 번역 첫 절 | 0.7초 |
| TTS 첫 패킷 | 0.3초 |
| **발화 종료 → 첫 음성** | **약 2.9초** |

---

## 평가

**Phase 1 과 병행해 만든다.** 없으면 이후 튜닝이 전부 체감에 의존하고 반드시 퇴행한다.

```bash
uv run eval/record_set.py --lang ko --prompts eval/prompts/ko.txt --out eval/data/ko
uv run eval/run_asr.py    --manifest eval/data/ko/manifest.jsonl --out eval/out/ko.hyp.jsonl
uv run eval/score_cer.py  --manifest eval/data/ko/manifest.jsonl --hyp eval/out/ko.hyp.jsonl
uv run --group eval eval/score_comet.py --hyp eval/out/ko2en.jsonl   # 개발 PC 에서만
```

- 4개 언어 각 100발화. **실제 사용할 마이크로, 실제 사용할 공간에서** 녹음한다.
- ASR 은 CER(`jiwer`), 번역은 COMET(`unbabel-comet`). BLEU 는 쓰지 않는다 —
  한국어·일본어 품질과 상관이 낮다.
- COMET 은 Orin 에 올리지 않는다. `score_comet.py` 는 aarch64 에서 기본적으로 거부한다.

---

## 로드맵

- [ ] **Phase 0** 검증 스파이크 (`spikes/README.md`) ← **여기서 멈춰 있음**
- [ ] Phase 1 영↔한 최소 경로 + 평가셋 구축
- [ ] Phase 2 절 단위 스트리밍 체인, 첫 음성 3초 달성
- [ ] Phase 3 게이팅·상태기계·실패 처리 전체
- [ ] Phase 4 4언어 확장, 번체 후처리, 라우팅 3종
- [ ] Phase 5 정확도 튜닝 (N-best 정정, 조건부 교차검증, 6턴 컨텍스트, 역번역 검증)

## 하지 않는 것

클라우드 API · 동시통역 정책(AlignAtt, LocalAgreement) · 벡터DB/임베딩 ·
영어 피벗 번역 · 브라우저 마이크 캡처 · 요청마다 모델 로드 ·
검증되지 않은 JetPack/CUDA/베이스 이미지 업그레이드.

정확도 개선은 프런트엔드 → 프롬프트·컨텍스트 → N-best·정정 → 모델 크기 순으로 한다.
