"""Configuration loading — YAML plus environment variables (pydantic-settings).

Precedence: environment (KOTONOHA__*) > local YAML > the selected overlay >
the selected accelerator profile > config/default.yaml. Nested keys use a double
underscore for overrides,
e.g. KOTONOHA__LLM__PROFILE=translategemma
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, ClassVar, Final, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from kotonoha._typing import override

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"
ACCELERATOR_PROFILES_ROOT = REPO_ROOT / "config" / "profiles" / "accelerators"
ACCELERATOR_PROFILE_COMPONENT = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")

SupportedLanguage = Literal["ko", "en", "zh-TW", "ja"]
ChineseVoice = Literal["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"]
EnglishVoice = Literal["Ryan", "Aiden"]
JapaneseVoice = Literal["Ono_Anna"]
KoreanVoice = Literal["Sohee"]
QwenVoice = Literal[
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]
QWEN_VOICE_NAMES: Final[dict[str, QwenVoice]] = {
    "vivian": "Vivian",
    "serena": "Serena",
    "uncle_fu": "Uncle_Fu",
    "dylan": "Dylan",
    "eric": "Eric",
    "ryan": "Ryan",
    "aiden": "Aiden",
    "ono_anna": "Ono_Anna",
    "sohee": "Sohee",
}
QWEN_LANGUAGE_VOICES: Final[dict[str, frozenset[QwenVoice]]] = {
    "Chinese": frozenset({"Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"}),
    "English": frozenset({"Ryan", "Aiden"}),
    "Japanese": frozenset({"Ono_Anna"}),
    "Korean": frozenset({"Sohee"}),
}


class SessionConfig(BaseModel):
    # text closes the microphone and takes utterances from the keyboard.
    __slots__: ClassVar[tuple[str, ...]] = ()
    mode: Literal["push_to_talk", "auto", "text"] = "push_to_talk"
    # Source language for typed input. auto reads it from the script.
    text_source_language: Literal["auto", "ko", "en", "zh-TW", "ja"] = "auto"
    routing: Literal["pair", "fixed", "broadcast"] = "pair"
    pair: list[SupportedLanguage] = Field(
        default_factory=lambda: ["ko", "en"],
        min_length=2,
        max_length=2,
    )
    fixed_target: SupportedLanguage = "en"
    broadcast_targets: list[SupportedLanguage] = Field(
        default_factory=lambda: ["ko", "en", "zh-TW", "ja"],
        min_length=1,
        max_length=4,
    )
    languages: list[SupportedLanguage] = Field(
        default_factory=lambda: ["ko", "en", "zh-TW", "ja"],
        min_length=1,
        max_length=4,
    )

    @model_validator(mode="after")
    def _check_pair(
        self,
        /,
    ) -> SessionConfig:
        if len(self.pair) != 2:
            raise ValueError("session.pair must contain exactly 2 languages")
        for field_name in ("pair", "broadcast_targets", "languages"):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"session.{field_name} must not contain duplicate languages")
        return self


class AudioConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    input_device: int | str | None = None
    output_device: int | str | None = None
    capture_sample_rate: int = Field(48000, ge=8000, le=192000)
    capture_block_ms: int = Field(20, ge=5, le=200)
    channels: int = Field(1, ge=1, le=8)
    work_sample_rate: int = Field(16000, ge=16000, le=16000)
    playback_sample_rate: int = Field(24000, ge=8000, le=192000)

    @property
    def capture_block_frames(
        self,
        /,
    ) -> int:
        return int(self.capture_sample_rate * self.capture_block_ms / 1000)


class DenoiseConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    enabled: bool = True
    backend: Literal["deepfilternet3", "none"] = "deepfilternet3"
    post_filter_beta: float = Field(0.02, ge=0.0, le=1.0)


class VadConfig(BaseModel):
    # "energy" is a development-machine fallback only; the device uses silero_onnx.
    __slots__: ClassVar[tuple[str, ...]] = ()
    backend: Literal["silero_onnx", "energy"] = "silero_onnx"
    model_path: Path = Path("./models/silero_vad.onnx")
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    neg_threshold: float = Field(0.35, ge=0.0, le=1.0)
    preroll_ms: int = Field(300, ge=200, le=300, description="non-negotiable, see §5.1")
    min_speech_ms: int = Field(120, ge=0, le=5000)
    silence_ms: int = Field(800, ge=100, le=5000)
    max_utterance_ms: int = Field(30000, ge=1000, le=120000)
    frame_ms: int = Field(32, ge=32, le=32)

    @model_validator(mode="after")
    def _check_thresholds(
        self,
        /,
    ) -> VadConfig:
        if self.neg_threshold > self.threshold:
            raise ValueError("frontend.vad.neg_threshold must not exceed threshold")
        if self.min_speech_ms > self.max_utterance_ms:
            raise ValueError("frontend.vad.min_speech_ms must not exceed max_utterance_ms")
        return self


class FrontendConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    denoise: DenoiseConfig = Field(default_factory=DenoiseConfig)
    vad: VadConfig = Field(default_factory=VadConfig)


class SharedMemoryConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    name: str = Field(
        "kotonoha_audio",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    slots: int = Field(8, ge=1, le=64)
    slot_seconds: int = Field(31, ge=1, le=121)
    sample_rate: int = Field(16000, ge=16000, le=16000)


class ServiceEndpointsConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    asr: str = Field("http://127.0.0.1:8001", max_length=2048)
    asr_verify: str = Field("http://127.0.0.1:8002", max_length=2048)
    llm: str = Field("http://127.0.0.1:8003", max_length=2048)
    tts: str = Field("http://127.0.0.1:8004", max_length=2048)

    @field_validator("asr", "asr_verify", "llm", "tts")
    @classmethod
    def _validate_endpoint(
        cls,
        value: str,
        /,
    ) -> str:
        del cls
        parsed = urlsplit(value)
        try:
            endpoint_port = parsed.port
        except ValueError as error:
            raise ValueError(f"invalid service endpoint port: {value}") from error
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError(f"service endpoint must be an HTTP(S) URL: {value}")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("service endpoint credentials must use bearer-token configuration")
        if endpoint_port == 0:
            raise ValueError("service endpoint port must be greater than zero")
        if parsed.query or parsed.fragment:
            raise ValueError("service endpoint must not contain a query or fragment")
        return value


class AcceleratorConfig(BaseModel):
    """Identify the accelerator and runtime selected by the active profile."""

    __slots__: ClassVar[tuple[str, ...]] = ()
    profile: str = "nvidia.jetson.agx-orin"
    vendor: str = "nvidia"
    family: str = "jetson"
    model: str = "agx-orin"
    runtime: str = "cuda"
    architecture: str = "sm_87"


ROLES = ("asr", "asr_verify", "llm", "tts")
Placement = Literal["local", "remote"]

# Which side runs each role, per performance mode.
#
#   onboard  everything on the Orin. The original design.
#   hybrid   only the LLM goes to the A6000. It is text-only, so audio remains
#            on the device.
#   remote   ASR, verification and TTS also move to the external server.
#            Utterance audio crosses the network.
#   custom   each role follows the matching entry in `placement`.
PERF_PLACEMENT: dict[str, dict[str, Placement]] = {
    "onboard": {"asr": "local", "asr_verify": "local", "llm": "local", "tts": "local"},
    "hybrid": {"asr": "local", "asr_verify": "local", "llm": "remote", "tts": "local"},
    "remote": {"asr": "remote", "asr_verify": "remote", "llm": "remote", "tts": "remote"},
    "custom": {"asr": "local", "asr_verify": "local", "llm": "local", "tts": "local"},
}


class RemoteConfig(BaseModel):
    """The external RTX A6000 box.

    This server is private infrastructure rather than a cloud API, so §12 still
    applies. In `remote` mode, utterance audio leaves the device. The `hybrid`
    mode keeps audio processing on the device.
    """
    __slots__: ClassVar[tuple[str, ...]] = ()

    enabled: bool = False
    services: ServiceEndpointsConfig = Field(
        default_factory=lambda: ServiceEndpointsConfig(
            asr="http://a6000.lan:8001",
            asr_verify="http://a6000.lan:8002",
            llm="http://a6000.lan:8003",
            tts="http://a6000.lan:8004",
        )
    )
    token: str | None = Field(None, min_length=32, max_length=4096)
    verify_tls: bool = True
    ca_bundle: Path | None = None
    connect_timeout_s: float = Field(1.5, gt=0.0, le=60.0)

    # Failover (§10 applied to the link): drop to the on-board service after
    # this many consecutive transport failures, and only return once the remote
    # has been healthy again for recover_after_s.
    failover_after: int = Field(2, ge=1, le=100)
    recover_after_s: float = Field(30.0, ge=0.0, le=3600.0)
    health_interval_s: float = Field(10.0, gt=0.0, le=3600.0)

    # Audio upload encoding. s16le halves the bytes on the wire against f32le
    # and costs nothing in ASR quality at 16 kHz.
    audio_encoding: Literal["s16le", "f32le"] = "s16le"


class LanguageIdentificationConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    min_confidence: float = Field(0.60, ge=0.0, le=1.0)
    min_duration_s: float = Field(1.0, ge=0.0, le=10.0)


class AsrConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    backend: Literal["vllm", "transformers"] = "vllm"
    model_id: str = Field("Qwen/Qwen3-ASR-0.6B-hf", min_length=1, max_length=512)
    model_revision: str = Field(
        "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
        pattern=r"^[0-9a-f]{40}$",
    )
    vllm_model_id: str = Field("Qwen/Qwen3-ASR-0.6B", min_length=1, max_length=512)
    vllm_served_model_name: str = Field("kotonoha-asr", min_length=1, max_length=128)
    vllm_realtime_architecture: Literal["qwen3_asr", "voxtral"] = "qwen3_asr"
    dtype: str = Field("float16", min_length=1, max_length=64)
    vllm_gpu_memory_utilization: float = Field(0.80, gt=0.0, le=1.0)
    vllm_max_model_len: int = Field(4096, ge=512, le=131072)
    vllm_enforce_eager: bool = True
    n_best: int = Field(5, ge=5, le=5)
    num_beams: int = Field(5, ge=5, le=10)
    max_new_tokens: int = Field(256, ge=1, le=512)
    timeout_s: float = Field(15.0, gt=0.0, le=300.0)
    avg_logprob_threshold: float = Field(-0.55, ge=-100.0, le=0.0)
    lid: LanguageIdentificationConfig = Field(default_factory=LanguageIdentificationConfig)


class AsrVerificationConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    enabled: bool = True
    # §5.5 makes verification conditional because it costs 0.8 s on the Orin.
    # The policy remains configurable for measured remote deployments.
    mode: Literal["conditional", "always"] = "conditional"
    backend: Literal["faster_whisper", "whisper_cpp"] = "faster_whisper"
    model_id: str = Field("large-v3", min_length=1, max_length=512)
    # The Jetson AArch64 wheel uses the CPU path. Remote overlays can select CUDA.
    compute_type: str = Field("int8", min_length=1, max_length=64)
    device: str = Field("cpu", min_length=1, max_length=64)
    beam_size: int = Field(5, ge=1, le=20)
    timeout_s: float = Field(3.0, gt=0.0, le=300.0)
    divergence_cer: float = Field(0.25, ge=0.0, le=1.0)


class LanguageModelProfile(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    repo: str = Field(min_length=1, max_length=256)
    directory: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    quantization: str | None = Field(None, max_length=64)
    dtype: str = Field("bfloat16", min_length=1, max_length=64)


class LanguageModelConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    profile: Literal["translategemma"] = "translategemma"
    profiles: dict[str, LanguageModelProfile] = Field(min_length=1, max_length=8)
    models_dir: Path = Path("./models/llm")
    served_model_name: str = Field("kotonoha-translation", min_length=1, max_length=128)
    max_model_len: int = Field(2048, ge=512, le=131072)
    gpu_memory_utilization: float = Field(0.35, gt=0.0, le=1.0)
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"] = "auto"
    max_num_seqs: int = Field(1, ge=1, le=1024)
    max_num_batched_tokens: int | None = Field(None, ge=1, le=1048576)
    enable_prefix_caching: bool = False
    limit_mm_per_prompt: dict[Literal["image", "audio", "video"], int] | None = None
    compilation_mode: int | None = Field(None, ge=0, le=3)
    compilation_cudagraph_capture_sizes: tuple[int, ...] = Field(
        default_factory=tuple,
        max_length=128,
    )
    compilation_cache_dir: Path | None = None
    enforce_eager: bool = True
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, gt=0.0, le=10.0)
    max_tokens: int = Field(512, ge=1, le=2048)
    stream: bool = True
    timeout_s: float = Field(3.0, gt=0.0, le=300.0)
    min_tok_per_s: float = Field(5.0, gt=0.0, le=1000.0)

    @model_validator(mode="after")
    def _check_engine_limits(
        self,
        /,
    ) -> LanguageModelConfig:
        if self.profile not in self.profiles:
            raise ValueError(f"llm.profiles does not define the active profile: {self.profile}")
        if self.limit_mm_per_prompt is not None and any(
            value < 0 or value > 16 for value in self.limit_mm_per_prompt.values()
        ):
            raise ValueError("llm.limit_mm_per_prompt values must be between 0 and 16")
        capture_sizes = self.compilation_cudagraph_capture_sizes
        if any(value <= 0 or value > 1024 for value in capture_sizes):
            raise ValueError(
                "llm.compilation_cudagraph_capture_sizes values must be between 1 and 1024"
            )
        if len(capture_sizes) != len(set(capture_sizes)):
            raise ValueError("llm.compilation_cudagraph_capture_sizes must not contain duplicates")
        return self

    @property
    def active(
        self,
        /,
    ) -> LanguageModelProfile:
        return self.profiles[self.profile]

    @property
    def model_path(
        self,
        /,
    ) -> Path:
        return self.models_dir / self.active.directory


class TextToSpeechVoices(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    ko: KoreanVoice = "Sohee"
    en: EnglishVoice = "Ryan"
    ja: JapaneseVoice = "Ono_Anna"
    zh_tw: ChineseVoice = Field(
        "Vivian",
        validation_alias=AliasChoices("zh_tw", "zh-TW"),
        serialization_alias="zh-TW",
    )

    def for_language(
        self,
        language: str,
        /,
    ) -> QwenVoice:
        voices: dict[str, QwenVoice] = {
            "ko": self.ko,
            "en": self.en,
            "ja": self.ja,
            "zh-TW": self.zh_tw,
        }
        return voices.get(language, self.en)


class TextToSpeechConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    backend: Literal["vllm_omni"] = "vllm_omni"
    served_model_name: str = Field("kotonoha-tts", min_length=1, max_length=128)
    task_type: Literal["CustomVoice"] = "CustomVoice"
    sample_rate: int = Field(24000, ge=24000, le=24000)
    timeout_s: float = Field(5.0, gt=0.0, le=300.0)
    max_new_tokens: int = Field(2048, ge=1, le=4096)
    max_audio_seconds: float = Field(30.0, gt=0.0, le=120.0)
    playback_buffer_seconds: float = Field(5.0, ge=1.0, le=30.0)
    voices: TextToSpeechVoices = Field(default_factory=TextToSpeechVoices)


class TraditionalChineseConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    opencc_config: Literal["s2twp"] = "s2twp"
    apply_to: list[Literal["asr", "translation"]] = Field(
        default_factory=lambda: ["asr", "translation"],
        max_length=2,
    )

    @model_validator(mode="after")
    def _check_unique_targets(
        self,
        /,
    ) -> TraditionalChineseConfig:
        if len(self.apply_to) != len(set(self.apply_to)):
            raise ValueError("zh.apply_to must not contain duplicates")
        return self


class ContextConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    history_turns: int = Field(6, ge=0, le=100)
    glossary_max_terms: int = Field(64, ge=0, le=1000)


class UserInterfaceConfig(BaseModel):
    # auto follows KOTONOHA_LANG, then the system locale, then English.
    __slots__: ClassVar[tuple[str, ...]] = ()
    language: Literal["auto", "en", "ko", "ja", "zh-TW"] = "auto"
    # Completed turns appear below the live Web interpreter. 0 hides the panel.
    history_turns: int = Field(20, ge=0, le=200)


class StoreConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    path: Path = Path("./data/kotonoha.db")
    maximum_turns: int = Field(10_000, ge=100, le=1_000_000)
    maximum_sessions: int = Field(1_000, ge=10, le=100_000)


class LoggingConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # Application logs and turn metrics go to separate files. Mixed together, the
    # turn log (§11) can no longer be parsed as-is and every reader needs a filter.
    log_path: Path = Path("./data/logs/kotonoha.jsonl")
    turn_log_path: Path = Path("./data/logs/turns.jsonl")
    max_bytes: int = Field(64 * 1024 * 1024, ge=1024 * 1024, le=10 * 1024**3)
    backup_count: int = Field(5, ge=1, le=100)
    console: bool = True
    # Console records render as "[   12.345678] LEVEL service: event key=value", the
    # kernel ring buffer layout. The JSONL files keep the structured form either way,
    # because the metrics and evaluation readers parse them.
    console_format: Literal["dmesg", "json"] = "dmesg"
    # Debug mode lowers the effective level to DEBUG and turns on the per-stage
    # detail in the terminal interface. It does not change what is persisted.
    debug: bool = False
    prometheus_port: int | None = Field(None, ge=1024, le=65535)

    def effective_level(
        self,
        /,
    ) -> str:
        return "DEBUG" if self.debug else self.level


class LatencyBudgetConfig(BaseModel):
    """Latency budget in milliseconds (§6)."""
    __slots__: ClassVar[tuple[str, ...]] = ()

    silence: int = Field(800, ge=0, le=600_000)
    frontend: int = Field(100, ge=0, le=600_000)
    asr: int = Field(900, ge=0, le=600_000)
    verify: int = Field(100, ge=0, le=600_000)
    llm_first_clause: int = Field(700, ge=0, le=600_000)
    tts_first_packet: int = Field(300, ge=0, le=600_000)
    total: int = Field(2900, ge=0, le=600_000)

    @model_validator(mode="after")
    def _check_total(
        self,
        /,
    ) -> LatencyBudgetConfig:
        if self.total < self.silence:
            raise ValueError("budget_ms.total must not be less than budget_ms.silence")
        return self


class Settings(BaseSettings):
    __slots__: ClassVar[tuple[str, ...]] = ()
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="KOTONOHA__",
        env_nested_delimiter="__",
        extra="forbid",
    )

    # onboard | hybrid | remote | custom — see PERF_PLACEMENT.
    perf_mode: Literal["onboard", "hybrid", "remote", "custom"] = "onboard"
    # Custom mode uses this map for each resident model role. Other modes retain
    # their preset and accept explicit per-role overrides for compatibility.
    placement: dict[str, Placement] = Field(default_factory=dict, max_length=4)

    session: SessionConfig = Field(default_factory=SessionConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    frontend: FrontendConfig = Field(default_factory=FrontendConfig)
    shm: SharedMemoryConfig = Field(default_factory=SharedMemoryConfig)
    services: ServiceEndpointsConfig = Field(default_factory=ServiceEndpointsConfig)
    remote: RemoteConfig = Field(default_factory=RemoteConfig)
    accelerator: AcceleratorConfig = Field(default_factory=AcceleratorConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    asr_verify: AsrVerificationConfig = Field(default_factory=AsrVerificationConfig)
    llm: LanguageModelConfig
    tts: TextToSpeechConfig = Field(default_factory=TextToSpeechConfig)
    zh: TraditionalChineseConfig = Field(default_factory=TraditionalChineseConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    ui: UserInterfaceConfig = Field(default_factory=UserInterfaceConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    budget_ms: LatencyBudgetConfig = Field(default_factory=LatencyBudgetConfig)

    # Kept so relative paths can be resolved against the repository root.
    root: Path = REPO_ROOT

    @model_validator(mode="after")
    def _check_audio_capacity(
        self,
        /,
    ) -> Settings:
        unknown_roles = set(self.placement) - set(ROLES)
        if unknown_roles:
            raise ValueError(
                f"unknown role in placement: {', '.join(sorted(unknown_roles))}"
            )
        required_milliseconds = (
            self.frontend.vad.max_utterance_ms + self.frontend.vad.frame_ms
        )
        if self.shm.slot_seconds * 1000 < required_milliseconds:
            raise ValueError(
                "shm.slot_seconds must hold frontend.vad.max_utterance_ms "
                "plus one VAD frame"
            )
        return self

    def resolve(
        self,
        /,
        path: Path,
    ) -> Path:
        return path if path.is_absolute() else (self.root / path).resolve()

    # -- role placement ----------------------------------------------------
    def resolved_placement(
        self,
        /,
    ) -> dict[str, Placement]:
        """Where each role actually runs.

        With the remote disabled everything collapses to local, whatever
        perf_mode says. A mode that silently points at an unreachable box would
        just turn into a per-turn timeout.
        """
        resolved = dict(PERF_PLACEMENT[self.perf_mode])
        for role, side in self.placement.items():
            if role not in ROLES:
                raise ValueError(f"unknown role in placement: {role}")
            resolved[role] = side
        if not self.remote.enabled:
            return dict.fromkeys(ROLES, "local")
        return resolved

    def url_for(
        self,
        /,
        role: str,
        side: Placement,
    ) -> str:
        services = self.remote.services if side == "remote" else self.services
        return getattr(services, role)

    @property
    def audio_leaves_device(
        self,
        /,
    ) -> bool:
        """Return whether utterance audio leaves the application host."""
        placement = self.resolved_placement()
        return placement["asr"] == "remote" or placement["asr_verify"] == "remote"

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        /,
        settings_cls: Any,
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> Any:
        """Environment variables beat the YAML file.

        pydantic-settings defaults to init (i.e. the loaded YAML) outranking env,
            which silently ignores one-off overrides such as
            KOTONOHA__LLM__PROFILE=translategemma.
        Per-device tuning and the spikes lean on those overrides, so flip the order.
        """
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)


def deep_merge(
    base: dict[str, Any],
    /,
    overlay: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_yaml(
    path: Path,
    /,
) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    llm_data = data.get("llm")
    legacy_memory_profile = (
        llm_data.pop("vllm_memory_profile", None) if isinstance(llm_data, dict) else None
    )
    if legacy_memory_profile is not None:
        legacy_profiles = {
            "jetson": "nvidia.jetson.agx-orin",
            "a6000": "nvidia.rtx.a6000",
        }
        if legacy_memory_profile in legacy_profiles:
            accelerator_data = data.setdefault("accelerator", {})
            if isinstance(accelerator_data, dict):
                accelerator_data["profile"] = legacy_profiles[legacy_memory_profile]
    return data


def accelerator_profile_path(
    profile: str,
    /,
) -> Path:
    """Resolve a dotted accelerator profile identifier to its YAML source."""
    components = profile.split(".")
    invalid_component = any(
        ACCELERATOR_PROFILE_COMPONENT.fullmatch(component) is None
        for component in components
    )
    if len(components) < 3 or invalid_component:
        raise ValueError(
            "accelerator profile must use <vendor>.<family>.<model> naming"
        )
    path = ACCELERATOR_PROFILES_ROOT.joinpath(*components[:-1], f"{components[-1]}.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"accelerator profile not found: {profile} ({path})")
    return path


def _profile_identifier(
    default_data: dict[str, Any],
    /,
    *,
    chosen_data: dict[str, Any],
    local_data: dict[str, Any],
    local_override: dict[str, Any] | None = None,
) -> str:
    candidates = (
        os.environ.get("KOTONOHA__ACCELERATOR__PROFILE"),
        (local_override or {}).get("accelerator", {}).get("profile"),
        local_data.get("accelerator", {}).get("profile"),
        chosen_data.get("accelerator", {}).get("profile"),
        default_data.get("accelerator", {}).get("profile"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return AcceleratorConfig().profile


def _configuration_layers(
    path: str | Path | None = None,
    /,
    *,
    local_override: dict[str, Any] | None = None,
) -> list[Path]:
    chosen = Path(path or os.environ.get("KOTONOHA_CONFIG") or DEFAULT_CONFIG)
    if not chosen.exists() and (path is not None or os.environ.get("KOTONOHA_CONFIG")):
        raise FileNotFoundError(f"config not found: {chosen}")

    default_data = read_yaml(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else {}
    chosen_data = read_yaml(chosen) if chosen.exists() else {}
    local = local_config_path()
    skip_local = bool(os.environ.get("KOTONOHA_SKIP_LOCAL_CONFIG")) and not os.environ.get(
        "KOTONOHA_LOCAL_CONFIG"
    )
    local_data = read_yaml(local) if local.exists() and not skip_local else {}
    profile = accelerator_profile_path(
        _profile_identifier(
            default_data,
            chosen_data=chosen_data,
            local_data=local_data,
            local_override=local_override,
        )
    )

    layers: list[Path] = []
    if DEFAULT_CONFIG.exists():
        layers.append(DEFAULT_CONFIG)
    if profile.resolve() not in {layer.resolve() for layer in layers}:
        layers.append(profile)
    if chosen.exists() and chosen.resolve() != DEFAULT_CONFIG.resolve():
        layers.append(chosen)
    return layers


def load_settings(
    path: str | Path | None = None,
    /,
) -> Settings:
    """Read the YAML layers and build Settings.

    Layers, each merged over the previous:

      1. config/default.yaml                         the full baseline
      2. the selected accelerator profile            measured device defaults
      3. the file given by --config or KOTONOHA_CONFIG, if it is a different one
      4. config/local.yaml                           per-device overrides, if present

    Layer 2 exists so files like performance.yaml can be small overlays that say
    only what differs, instead of duplicating the whole baseline and drifting
    from it. Environment variables still beat all three.

    KOTONOHA_SKIP_LOCAL_CONFIG drops layer 3 when it resolves to this machine's own
    config/local.yaml. The test suite sets it: that file carries a real device's
    remote endpoints and token, and a suite that read it would dial the external
    server instead of running offline. An explicit KOTONOHA_LOCAL_CONFIG still
    applies, so the layer itself remains testable.
    """
    layers = _configuration_layers(path)
    # The skip applies to this machine's own file, not to the mechanism: a caller
    # that names a path with KOTONOHA_LOCAL_CONFIG still gets that layer, which is
    # how the management API and its tests exercise it.
    skip_local = bool(os.environ.get("KOTONOHA_SKIP_LOCAL_CONFIG")) and not os.environ.get(
        "KOTONOHA_LOCAL_CONFIG"
    )
    if not skip_local:
        local = local_config_path()
        if local.exists():
            layers.append(local)

    data: dict[str, Any] = {}
    for layer in layers:
        data = deep_merge(data, read_yaml(layer))

    data.setdefault("root", str(REPO_ROOT))
    return Settings(**data)


def config_layers(
    path: str | Path | None = None,
    /,
    *,
    local_override: dict[str, Any] | None = None,
) -> list[Path]:
    """The YAML files load_settings would merge, in order.

    The configuration editor needs this to validate a candidate local.yaml against
    the same profile and overlay layering the runtime uses.
    """
    return _configuration_layers(path, local_override=local_override)


LOCAL_CONFIG = DEFAULT_CONFIG.parent / "local.yaml"


def local_config_path() -> Path:
    """Return the host-specific override path.

    The Orin uses config/local.yaml. Remote containers set KOTONOHA_LOCAL_CONFIG
    to a separate file, so editing the A6000 cannot overwrite the Orin's values
    when both trees are mounted from the same development checkout.
    """
    return Path(os.environ.get("KOTONOHA_LOCAL_CONFIG", LOCAL_CONFIG))
