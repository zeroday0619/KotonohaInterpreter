"""Configuration loading — YAML plus environment variables (pydantic-settings).

Precedence: environment (KOTONOHA__*) > local YAML > the selected overlay >
the selected accelerator profile > config/default.yaml. Nested keys use a double
underscore for overrides,
e.g. KOTONOHA__LLM__PROFILE=translategemma
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from kotonoha._typing import override

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"
ACCELERATOR_PROFILES_ROOT = REPO_ROOT / "config" / "profiles" / "accelerators"

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
    pair: list[SupportedLanguage] = ["ko", "en"]
    fixed_target: SupportedLanguage = "en"
    broadcast_targets: list[SupportedLanguage] = ["ko", "en", "zh-TW", "ja"]
    languages: list[SupportedLanguage] = ["ko", "en", "zh-TW", "ja"]

    @model_validator(mode="after")
    def _check_pair(
        self,
        /,
    ) -> SessionConfig:
        if self.routing == "pair" and len(self.pair) != 2:
            raise ValueError("session.pair must contain exactly 2 languages")
        return self


class AudioConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    input_device: int | str | None = None
    output_device: int | str | None = None
    capture_sample_rate: int = 48000
    capture_block_ms: int = 20
    channels: int = 1
    work_sample_rate: int = 16000
    playback_sample_rate: int = 24000

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
    post_filter_beta: float = 0.02


class VadConfig(BaseModel):
    # "energy" is a development-machine fallback only; the device uses silero_onnx.
    __slots__: ClassVar[tuple[str, ...]] = ()
    backend: Literal["silero_onnx", "energy"] = "silero_onnx"
    model_path: Path = Path("./models/silero_vad.onnx")
    threshold: float = 0.5
    neg_threshold: float = 0.35
    preroll_ms: int = Field(300, ge=200, le=500, description="non-negotiable, see §5.1")
    min_speech_ms: int = 120
    silence_ms: int = 800
    max_utterance_ms: int = 30000
    frame_ms: int = 32


class FrontendConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    denoise: DenoiseConfig = DenoiseConfig()
    vad: VadConfig = VadConfig()


class SharedMemoryConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    name: str = "kotonoha_audio"
    slots: int = 8
    slot_seconds: int = 30
    sample_rate: int = 16000


class ServiceEndpointsConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    asr: str = "http://127.0.0.1:8001"
    asr_verify: str = "http://127.0.0.1:8002"
    llm: str = "http://127.0.0.1:8003"
    tts: str = "http://127.0.0.1:8004"


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
PERF_PLACEMENT: dict[str, dict[str, Placement]] = {
    "onboard": {"asr": "local", "asr_verify": "local", "llm": "local", "tts": "local"},
    "hybrid": {"asr": "local", "asr_verify": "local", "llm": "remote", "tts": "local"},
    "remote": {"asr": "remote", "asr_verify": "remote", "llm": "remote", "tts": "remote"},
}


class RemoteConfig(BaseModel):
    """The external RTX A6000 box.

    This server is private infrastructure rather than a cloud API, so §12 still
    applies. In `remote` mode, utterance audio leaves the device. The `hybrid`
    mode keeps audio processing on the device.
    """
    __slots__: ClassVar[tuple[str, ...]] = ()

    enabled: bool = False
    services: ServiceEndpointsConfig = ServiceEndpointsConfig(
        asr="http://a6000.lan:8001",
        asr_verify="http://a6000.lan:8002",
        llm="http://a6000.lan:8003",
        tts="http://a6000.lan:8004",
    )
    token: str | None = None  # bearer token; prefer KOTONOHA__REMOTE__TOKEN
    verify_tls: bool = True
    ca_bundle: Path | None = None
    connect_timeout_s: float = 1.5

    # Failover (§10 applied to the link): drop to the on-board service after
    # this many consecutive transport failures, and only return once the remote
    # has been healthy again for recover_after_s.
    failover_after: int = 2
    recover_after_s: float = 30.0
    health_interval_s: float = 10.0

    # Audio upload encoding. s16le halves the bytes on the wire against f32le
    # and costs nothing in ASR quality at 16 kHz.
    audio_encoding: Literal["s16le", "f32le"] = "s16le"


class LanguageIdentificationConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    min_confidence: float = 0.60
    min_duration_s: float = 1.0


class AsrConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    backend: Literal["vllm", "transformers"] = "vllm"
    model_id: str = "Qwen/Qwen3-ASR-0.6B-hf"
    vllm_model_id: str = "Qwen/Qwen3-ASR-0.6B"
    vllm_served_model_name: str = "kotonoha-asr"
    vllm_realtime_architecture: Literal["qwen3_asr", "voxtral"] = "qwen3_asr"
    dtype: str = "float16"
    vllm_gpu_memory_utilization: float = Field(0.80, gt=0.0, le=1.0)
    vllm_max_model_len: int = Field(4096, ge=512)
    vllm_enforce_eager: bool = True
    n_best: int = 5
    num_beams: int = 5
    max_new_tokens: int = 256
    timeout_s: float = 4.0
    avg_logprob_threshold: float = -0.55
    lid: LanguageIdentificationConfig = LanguageIdentificationConfig()


class AsrVerificationConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    enabled: bool = True
    # §5.5 makes verification conditional because it costs 0.8 s on the Orin.
    # The policy remains configurable for measured remote deployments.
    mode: Literal["conditional", "always"] = "conditional"
    backend: Literal["faster_whisper", "whisper_cpp"] = "faster_whisper"
    model_id: str = "large-v3"
    # The Jetson AArch64 wheel uses the CPU path. Remote overlays can select CUDA.
    compute_type: str = "int8"
    device: str = "cpu"
    beam_size: int = 5
    timeout_s: float = 3.0
    divergence_cer: float = 0.25


class LanguageModelProfile(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    repo: str
    directory: str
    quantization: str | None = None
    dtype: str = "bfloat16"


class LanguageModelConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    profile: Literal["translategemma"] = "translategemma"
    profiles: dict[str, LanguageModelProfile]
    models_dir: Path = Path("./models/llm")
    served_model_name: str = "kotonoha-translation"
    max_model_len: int = Field(2048, ge=512)
    gpu_memory_utilization: float = Field(0.35, gt=0.0, le=1.0)
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"] = "auto"
    max_num_seqs: int = Field(1, ge=1)
    max_num_batched_tokens: int | None = Field(None, ge=1)
    enable_prefix_caching: bool = False
    limit_mm_per_prompt: dict[Literal["image", "audio", "video"], int] | None = None
    compilation_mode: int | None = Field(None, ge=0, le=3)
    compilation_cudagraph_capture_sizes: tuple[int, ...] = ()
    compilation_cache_dir: Path | None = None
    enforce_eager: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_tokens: int = 512
    stream: bool = True
    timeout_s: float = 3.0
    min_tok_per_s: float = 5.0

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
    served_model_name: str = "kotonoha-tts"
    task_type: Literal["CustomVoice"] = "CustomVoice"
    sample_rate: int = Field(24000, ge=24000, le=24000)
    timeout_s: float = 5.0
    voices: TextToSpeechVoices = Field(default_factory=TextToSpeechVoices)


class TraditionalChineseConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    opencc_config: str = "s2twp"
    apply_to: list[Literal["asr", "translation"]] = ["asr", "translation"]


class ContextConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    history_turns: int = 6
    glossary_max_terms: int = 64


class UserInterfaceConfig(BaseModel):
    # auto follows KOTONOHA_LANG, then the system locale, then English.
    __slots__: ClassVar[tuple[str, ...]] = ()
    language: Literal["auto", "en", "ko", "ja", "zh-TW"] = "auto"
    refresh_hz: int = Field(60, ge=15, le=60)
    # Completed turns appear as messages below the live panes. 0 hides the panel.
    history_turns: int = Field(20, ge=0, le=200)


class StoreConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    path: Path = Path("./data/kotonoha.db")


class LoggingConfig(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    level: str = "INFO"
    # Application logs and turn metrics go to separate files. Mixed together, the
    # turn log (§11) can no longer be parsed as-is and every reader needs a filter.
    log_path: Path = Path("./data/logs/kotonoha.jsonl")
    turn_log_path: Path = Path("./data/logs/turns.jsonl")
    console: bool = True


class LatencyBudgetConfig(BaseModel):
    """Latency budget in milliseconds (§6)."""
    __slots__: ClassVar[tuple[str, ...]] = ()

    silence: int = 800
    frontend: int = 100
    asr: int = 900
    verify: int = 100
    llm_first_clause: int = 700
    tts_first_packet: int = 300
    total: int = 2900


class Settings(BaseSettings):
    __slots__: ClassVar[tuple[str, ...]] = ()
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="KOTONOHA__",
        env_nested_delimiter="__",
        extra="forbid",
    )

    # onboard | hybrid | remote — see PERF_PLACEMENT.
    perf_mode: Literal["onboard", "hybrid", "remote"] = "onboard"
    # Explicit per-role override. Anything omitted follows perf_mode.
    placement: dict[str, Placement] = {}

    session: SessionConfig = SessionConfig()
    audio: AudioConfig = AudioConfig()
    frontend: FrontendConfig = FrontendConfig()
    shm: SharedMemoryConfig = SharedMemoryConfig()
    services: ServiceEndpointsConfig = ServiceEndpointsConfig()
    remote: RemoteConfig = RemoteConfig()
    accelerator: AcceleratorConfig = AcceleratorConfig()
    asr: AsrConfig = AsrConfig()
    asr_verify: AsrVerificationConfig = AsrVerificationConfig()
    llm: LanguageModelConfig
    tts: TextToSpeechConfig = TextToSpeechConfig()
    zh: TraditionalChineseConfig = TraditionalChineseConfig()
    context: ContextConfig = ContextConfig()
    ui: UserInterfaceConfig = UserInterfaceConfig()
    store: StoreConfig = StoreConfig()
    logging: LoggingConfig = LoggingConfig()
    budget_ms: LatencyBudgetConfig = LatencyBudgetConfig()

    # Kept so relative paths can be resolved against the repository root.
    root: Path = REPO_ROOT

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
        """True when utterance audio is sent off the box. Surfaced in the TUI."""
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
        not component or component in {".", ".."} for component in components
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
