"""설정 로딩 — YAML + 환경변수(pydantic-settings).

우선순위: 환경변수(KOTONOHA__*) > --config 로 준 YAML > config/default.yaml
중첩 키는 이중 언더스코어로 덮어쓴다.  예) KOTONOHA__LLM__PROFILE=moe
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"

Lang = Literal["ko", "en", "zh-TW", "ja"]


class SessionCfg(BaseModel):
    mode: Literal["push_to_talk", "auto"] = "push_to_talk"
    routing: Literal["pair", "fixed", "broadcast"] = "pair"
    pair: list[Lang] = ["ko", "en"]
    fixed_target: Lang = "en"
    broadcast_targets: list[Lang] = ["ko", "en", "zh-TW", "ja"]
    languages: list[Lang] = ["ko", "en", "zh-TW", "ja"]

    @model_validator(mode="after")
    def _check_pair(self) -> SessionCfg:
        if self.routing == "pair" and len(self.pair) != 2:
            raise ValueError("session.pair must contain exactly 2 languages")
        return self


class AudioCfg(BaseModel):
    input_device: int | str | None = None
    output_device: int | str | None = None
    capture_sample_rate: int = 48000
    capture_block_ms: int = 20
    channels: int = 1
    work_sample_rate: int = 16000
    playback_sample_rate: int = 24000

    @property
    def capture_block_frames(self) -> int:
        return int(self.capture_sample_rate * self.capture_block_ms / 1000)


class DenoiseCfg(BaseModel):
    enabled: bool = True
    backend: Literal["deepfilternet3", "none"] = "deepfilternet3"
    post_filter_beta: float = 0.02


class VadCfg(BaseModel):
    # energy 는 개발 PC 전용 폴백. 실기에서는 silero_onnx 만 쓴다.
    backend: Literal["silero_onnx", "energy"] = "silero_onnx"
    model_path: Path = Path("./models/silero_vad.onnx")
    threshold: float = 0.5
    neg_threshold: float = 0.35
    preroll_ms: int = Field(300, ge=200, le=500, description="§5.1 타협 불가")
    min_speech_ms: int = 120
    silence_ms: int = 800
    max_utterance_ms: int = 30000
    frame_ms: int = 32


class FrontendCfg(BaseModel):
    denoise: DenoiseCfg = DenoiseCfg()
    vad: VadCfg = VadCfg()


class ShmCfg(BaseModel):
    name: str = "kotonoha_audio"
    slots: int = 8
    slot_seconds: int = 30
    sample_rate: int = 16000


class ServicesCfg(BaseModel):
    asr: str = "http://127.0.0.1:8001"
    asr_verify: str = "http://127.0.0.1:8002"
    llm: str = "http://127.0.0.1:8003"
    tts: str = "http://127.0.0.1:8004"


class LidCfg(BaseModel):
    min_confidence: float = 0.60
    min_duration_s: float = 1.0


class AsrCfg(BaseModel):
    backend: Literal["transformers", "vllm"] = "transformers"
    model_id: str = "Qwen/Qwen3-ASR-1.7B-hf"
    vllm_model_id: str = "Qwen/Qwen3-ASR-1.7B"
    dtype: str = "float16"
    n_best: int = 5
    num_beams: int = 5
    max_new_tokens: int = 256
    timeout_s: float = 4.0
    avg_logprob_threshold: float = -0.55
    lid: LidCfg = LidCfg()


class AsrVerifyCfg(BaseModel):
    enabled: bool = True
    backend: Literal["faster_whisper", "whisper_cpp"] = "faster_whisper"
    model_id: str = "large-v3"
    compute_type: str = "int8_float16"
    device: str = "cuda"
    beam_size: int = 5
    timeout_s: float = 3.0
    divergence_cer: float = 0.25


class LlmProfile(BaseModel):
    repo: str
    file: str
    n_gpu_layers: int = -1


class LlmCfg(BaseModel):
    profile: Literal["moe", "dense"] = "dense"
    profiles: dict[str, LlmProfile]
    models_dir: Path = Path("./models/gguf")
    n_ctx: int = 2048
    n_batch: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    repeat_penalty: float = 1.05
    max_tokens: int = 512
    stream: bool = True
    timeout_s: float = 3.0
    min_tok_per_s: float = 5.0

    @property
    def active(self) -> LlmProfile:
        return self.profiles[self.profile]

    @property
    def gguf_path(self) -> Path:
        return self.models_dir / self.active.file


class TtsCfg(BaseModel):
    backend: Literal["qwen3", "melo"] = "melo"
    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    sample_rate: int = 24000
    chunk_ms: int = 200
    timeout_s: float = 5.0
    fallback: Literal["melo", "none"] = "melo"
    voices: dict[str, str] = {}
    melo_speakers: dict[str, str] = {}


class ZhCfg(BaseModel):
    opencc_config: str = "s2twp"
    apply_to: list[Literal["asr", "translation"]] = ["asr", "translation"]


class ContextCfg(BaseModel):
    history_turns: int = 6
    glossary_max_terms: int = 64


class StoreCfg(BaseModel):
    path: Path = Path("./data/kotonoha.db")


class LoggingCfg(BaseModel):
    level: str = "INFO"
    # 애플리케이션 로그와 턴 메트릭은 파일을 나눈다. 섞이면 §11 의 턴 로그를
    # 그대로 파싱할 수 없게 되고, 매번 필터링 코드를 짜게 된다.
    log_path: Path = Path("./data/logs/kotonoha.jsonl")
    turn_log_path: Path = Path("./data/logs/turns.jsonl")
    console: bool = False


class BudgetCfg(BaseModel):
    """§6 지연 예산 (ms)."""

    silence: int = 800
    frontend: int = 100
    asr: int = 900
    verify: int = 100
    llm_first_clause: int = 700
    tts_first_packet: int = 300
    total: int = 2900


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KOTONOHA__",
        env_nested_delimiter="__",
        extra="forbid",
    )

    session: SessionCfg = SessionCfg()
    audio: AudioCfg = AudioCfg()
    frontend: FrontendCfg = FrontendCfg()
    shm: ShmCfg = ShmCfg()
    services: ServicesCfg = ServicesCfg()
    asr: AsrCfg = AsrCfg()
    asr_verify: AsrVerifyCfg = AsrVerifyCfg()
    llm: LlmCfg
    tts: TtsCfg = TtsCfg()
    zh: ZhCfg = ZhCfg()
    context: ContextCfg = ContextCfg()
    store: StoreCfg = StoreCfg()
    logging: LoggingCfg = LoggingCfg()
    budget_ms: BudgetCfg = BudgetCfg()

    # 설정 파일이 놓인 위치 기준으로 상대경로를 풀기 위해 보관
    root: Path = REPO_ROOT

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        """환경변수가 YAML 을 이긴다.

        pydantic-settings 의 기본 순서는 init(=YAML 로드 결과)이 env 보다 우선이라
        `KOTONOHA__LLM__PROFILE=moe` 같은 임시 오버라이드가 조용히 무시된다.
        기기별 튜닝과 스파이크 실행에서 이 오버라이드에 계속 의존하므로 뒤집는다.
        """
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else (self.root / p).resolve()


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(path: str | Path | None = None) -> Settings:
    """YAML을 읽어 Settings를 만든다. path 미지정 시 KOTONOHA_CONFIG 또는 기본 파일."""
    cfg_path = Path(path or os.environ.get("KOTONOHA_CONFIG") or DEFAULT_CONFIG)
    data: dict[str, Any] = {}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    elif path is not None:
        raise FileNotFoundError(f"config not found: {cfg_path}")

    # 같은 디렉터리의 local.yaml 이 있으면 덮어쓴다 (기기별 장치 인덱스 등)
    local = cfg_path.parent / "local.yaml"
    if local.exists():
        data = _deep_merge(data, yaml.safe_load(local.read_text(encoding="utf-8")) or {})

    data.setdefault("root", str(REPO_ROOT))
    return Settings(**data)


__all__ = ["Settings", "load_settings", "Lang", "REPO_ROOT", "DEFAULT_CONFIG"]
