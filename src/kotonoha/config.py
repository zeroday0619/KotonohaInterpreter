"""Configuration loading — YAML plus environment variables (pydantic-settings).

Precedence: environment (KOTONOHA__*) > the YAML given with --config >
config/default.yaml. Nested keys are overridden with a double underscore,
e.g. KOTONOHA__LLM__PROFILE=moe
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
    # "energy" is a development-machine fallback only; the device uses silero_onnx.
    backend: Literal["silero_onnx", "energy"] = "silero_onnx"
    model_path: Path = Path("./models/silero_vad.onnx")
    threshold: float = 0.5
    neg_threshold: float = 0.35
    preroll_ms: int = Field(300, ge=200, le=500, description="non-negotiable, see §5.1")
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


ROLES = ("asr", "asr_verify", "llm", "tts")
Placement = Literal["local", "remote"]

# Which side runs each role, per performance mode.
#
#   onboard  everything on the Orin. The original design.
#   hybrid   only the LLM goes to the A6000. It is the single biggest latency
#            win and it is text-only, so no audio ever leaves the device.
#   remote   ASR, verification and TTS move too. Fastest when the link is good,
#            but utterance audio now crosses the network.
PERF_PLACEMENT: dict[str, dict[str, Placement]] = {
    "onboard": {"asr": "local", "asr_verify": "local", "llm": "local", "tts": "local"},
    "hybrid": {"asr": "local", "asr_verify": "local", "llm": "remote", "tts": "local"},
    "remote": {"asr": "remote", "asr_verify": "remote", "llm": "remote", "tts": "remote"},
}


class RemoteCfg(BaseModel):
    """The external RTX A6000 box.

    This is our own machine on our own network, not a cloud API, so §12 still
    holds. But in `remote` mode the utterance audio does leave the device —
    that is a deliberate trade, and `hybrid` exists for when it is not acceptable.
    """

    enabled: bool = False
    services: ServicesCfg = ServicesCfg(
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
    # §5.5 makes verification conditional because it costs 0.8 s on the Orin.
    # On the A6000 it is cheap enough to run every turn, which strictly improves
    # accuracy — so the policy is configurable rather than hard-coded.
    mode: Literal["conditional", "always"] = "conditional"
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


class UiCfg(BaseModel):
    # auto follows KOTONOHA_LANG, then the system locale, then English.
    language: Literal["auto", "en", "ko", "ja", "zh-TW"] = "auto"


class StoreCfg(BaseModel):
    path: Path = Path("./data/kotonoha.db")


class LoggingCfg(BaseModel):
    level: str = "INFO"
    # Application logs and turn metrics go to separate files. Mixed together, the
    # turn log (§11) can no longer be parsed as-is and every reader needs a filter.
    log_path: Path = Path("./data/logs/kotonoha.jsonl")
    turn_log_path: Path = Path("./data/logs/turns.jsonl")
    console: bool = False


class BudgetCfg(BaseModel):
    """Latency budget in milliseconds (§6)."""

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

    # onboard | hybrid | remote — see PERF_PLACEMENT.
    perf_mode: Literal["onboard", "hybrid", "remote"] = "onboard"
    # Explicit per-role override. Anything omitted follows perf_mode.
    placement: dict[str, Placement] = {}

    session: SessionCfg = SessionCfg()
    audio: AudioCfg = AudioCfg()
    frontend: FrontendCfg = FrontendCfg()
    shm: ShmCfg = ShmCfg()
    services: ServicesCfg = ServicesCfg()
    remote: RemoteCfg = RemoteCfg()
    asr: AsrCfg = AsrCfg()
    asr_verify: AsrVerifyCfg = AsrVerifyCfg()
    llm: LlmCfg
    tts: TtsCfg = TtsCfg()
    zh: ZhCfg = ZhCfg()
    context: ContextCfg = ContextCfg()
    ui: UiCfg = UiCfg()
    store: StoreCfg = StoreCfg()
    logging: LoggingCfg = LoggingCfg()
    budget_ms: BudgetCfg = BudgetCfg()

    # Kept so relative paths can be resolved against the repository root.
    root: Path = REPO_ROOT

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else (self.root / p).resolve()

    # -- role placement ----------------------------------------------------
    def resolved_placement(self) -> dict[str, Placement]:
        """Where each role actually runs.

        With the remote disabled everything collapses to local, whatever
        perf_mode says. A mode that silently points at an unreachable box would
        just turn into a per-turn timeout.
        """
        base = dict(PERF_PLACEMENT[self.perf_mode])
        for role, side in self.placement.items():
            if role not in ROLES:
                raise ValueError(f"unknown role in placement: {role}")
            base[role] = side
        if not self.remote.enabled:
            return dict.fromkeys(ROLES, "local")
        return base

    def url_for(self, role: str, side: Placement) -> str:
        svc = self.remote.services if side == "remote" else self.services
        return getattr(svc, role)

    @property
    def audio_leaves_device(self) -> bool:
        """True when utterance audio is sent off the box. Surfaced in the TUI."""
        p = self.resolved_placement()
        return p["asr"] == "remote" or p["asr_verify"] == "remote"

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        """Environment variables beat the YAML file.

        pydantic-settings defaults to init (i.e. the loaded YAML) outranking env,
        which silently ignores one-off overrides like KOTONOHA__LLM__PROFILE=moe.
        Per-device tuning and the spikes lean on those overrides, so flip the order.
        """
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def read_yaml(p: Path) -> dict[str, Any]:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_settings(path: str | Path | None = None) -> Settings:
    """Read the YAML layers and build Settings.

    Layers, each merged over the previous:

      1. config/default.yaml       the full baseline
      2. the file given by --config or KOTONOHA_CONFIG, if it is a different one
      3. config/local.yaml         per-device overrides, if present

    Layer 2 exists so files like performance.yaml can be small overlays that say
    only what differs, instead of duplicating the whole baseline and drifting
    from it. Environment variables still beat all three.
    """
    chosen = Path(path or os.environ.get("KOTONOHA_CONFIG") or DEFAULT_CONFIG)
    if not chosen.exists() and (path is not None or os.environ.get("KOTONOHA_CONFIG")):
        raise FileNotFoundError(f"config not found: {chosen}")

    layers: list[Path] = []
    if DEFAULT_CONFIG.exists():
        layers.append(DEFAULT_CONFIG)
    if chosen.exists() and chosen.resolve() != DEFAULT_CONFIG.resolve():
        layers.append(chosen)
    local = local_config_path()
    if local.exists():
        layers.append(local)

    data: dict[str, Any] = {}
    for layer in layers:
        data = deep_merge(data, read_yaml(layer))

    data.setdefault("root", str(REPO_ROOT))
    return Settings(**data)


def config_layers(path: str | Path | None = None) -> list[Path]:
    """The YAML files load_settings would merge, in order.

    The configuration editor needs this to validate a candidate local.yaml against
    the same layering the runtime uses.
    """
    chosen = Path(path or os.environ.get("KOTONOHA_CONFIG") or DEFAULT_CONFIG)
    layers: list[Path] = []
    if DEFAULT_CONFIG.exists():
        layers.append(DEFAULT_CONFIG)
    if chosen.exists() and chosen.resolve() != DEFAULT_CONFIG.resolve():
        layers.append(chosen)
    return layers


LOCAL_CONFIG = DEFAULT_CONFIG.parent / "local.yaml"


def local_config_path() -> Path:
    """Return the host-specific override path.

    The Orin uses config/local.yaml. Remote containers set KOTONOHA_LOCAL_CONFIG
    to a separate file, so editing the A6000 cannot overwrite the Orin's values
    when both trees are mounted from the same development checkout.
    """
    return Path(os.environ.get("KOTONOHA_LOCAL_CONFIG", LOCAL_CONFIG))

__all__ = [
    "Settings",
    "load_settings",
    "config_layers",
    "deep_merge",
    "read_yaml",
    "Lang",
    "Placement",
    "ROLES",
    "REPO_ROOT",
    "DEFAULT_CONFIG",
    "LOCAL_CONFIG",
    "local_config_path",
]
