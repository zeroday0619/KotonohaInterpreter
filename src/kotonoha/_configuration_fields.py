"""UI-neutral metadata for editable Kotonoha settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from kotonoha._config import Settings
from kotonoha._i18n import LOCALE_NAMES, N_, _


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Describe one editable leaf in :class:`Settings`."""

    path: str
    section: str
    kind: str
    choices: tuple[str, ...] = ()
    optional: bool = False
    value_kind: str = "text"


LANGUAGE_CHOICES = ("auto", *LOCALE_NAMES)

SECTIONS = (
    "interface",
    "session",
    "audio",
    "frontend",
    "runtime",
    "remote",
    "asr",
    "asr_verify",
    "llm",
    "tts",
    "language",
    "data",
    "observability",
)

TOP_LEVEL_SECTIONS = {
    "accelerator": "runtime",
    "ui": "interface",
    "session": "session",
    "audio": "audio",
    "frontend": "frontend",
    "shm": "runtime",
    "services": "runtime",
    "perf_mode": "remote",
    "placement": "remote",
    "remote": "remote",
    "asr": "asr",
    "asr_verify": "asr_verify",
    "llm": "llm",
    "tts": "tts",
    "zh": "language",
    "context": "data",
    "store": "data",
    "logging": "observability",
    "budget_ms": "observability",
}

SECTION_LABELS: dict[str, str] = {
    "asr": N_("Primary ASR"),
    "asr_verify": N_("Verification ASR"),
    "audio": N_("Audio devices"),
    "data": N_("Context and storage"),
    "frontend": N_("Audio frontend"),
    "interface": N_("Interface"),
    "language": N_("Language processing"),
    "llm": N_("Translation LLM"),
    "observability": N_("Logging and latency budgets"),
    "remote": N_("External server"),
    "runtime": N_("Runtime services"),
    "session": N_("Session"),
    "tts": N_("Speech synthesis"),
}

VALUE_KIND_DESCRIPTIONS: dict[str, str] = {
    "collection": N_("YAML list or mapping for {path}."),
    "number": N_("Numeric value for {path}."),
    "path": N_("Filesystem path for {path}."),
    "text": N_("Text value for {path}."),
}

FIELD_DESCRIPTIONS: dict[str, str] = {
    "asr.backend": N_("vLLM is the default; Transformers is the fallback."),
    "asr.n_best": N_("Hypotheses returned per utterance. The correction pass consumes all."),
    "asr_verify.mode": N_(
        "conditional verifies only low-confidence turns; always verifies every turn."
    ),
    "audio.input_device": N_(
        "Host audio input used by command-line replay and diagnostics. Browser audio "
        "devices are selected on the Interpreter page."
    ),
    "audio.output_device": N_(
        "Host audio output used by command-line diagnostics. Browser audio devices are "
        "selected on the Interpreter page."
    ),
    "frontend.denoise.enabled": N_("DeepFilterNet3 noise suppression."),
    "frontend.vad.backend": N_("silero_onnx on the device; energy is a workstation fallback."),
    "frontend.vad.preroll_ms": N_(
        "Audio retained before speech onset. Below 200 ms the first syllable is clipped."
    ),
    "frontend.vad.silence_ms": N_("Silence required before end-of-utterance."),
    "frontend.vad.threshold": N_("Speech onset probability, 0 to 1."),
    "llm.profile": N_("Selects the target-specific TranslateGemma translation model."),
    "logging.console": N_(
        "Publish structured application logs to the Web log panel. Model services emit "
        "JSON to their console."
    ),
    "logging.prometheus_port": N_(
        "Optional localhost port for Prometheus turn metrics. Empty disables the exporter."
    ),
    "perf_mode": N_(
        "onboard runs everything locally. hybrid moves only the LLM and keeps audio on the "
        "device. remote moves every model. custom selects each role independently."
    ),
    "placement": N_(
        "Custom mode placement for ASR, verification ASR, translation LLM, and TTS."
    ),
    "remote.audio_encoding": N_("s16le halves the bytes on the wire against f32le."),
    "remote.enabled": N_("When false, every role runs locally regardless of perf_mode."),
    "remote.failover_after": N_("Consecutive transport failures before a role falls back."),
    "remote.services.asr": N_("ASR service URL on the external server."),
    "remote.services.asr_verify": N_("Verification service URL on the external server."),
    "remote.services.llm": N_("Translation service URL on the external server."),
    "remote.services.tts": N_("Speech synthesis service URL on the external server."),
    "session.mode": N_(
        "push_to_talk requires a key press, auto segments on the VAD, and text closes the "
        "microphone and takes utterances from the keyboard."
    ),
    "session.routing": N_(
        "pair alternates between two languages; fixed always uses one target language."
    ),
    "session.text_source_language": N_(
        "Source language for typed input. auto reads it from the script."
    ),
    "tts.backend": N_("vllm_omni requires successful vLLM-Omni Spike 2 validation."),
    "ui.language": N_("Interface language. auto follows the system locale."),
}


def _without_none(
    annotation: Any,
    /,
) -> tuple[Any, bool]:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        arguments = tuple(
            argument for argument in get_args(annotation) if argument is not type(None)
        )
        optional = len(arguments) != len(get_args(annotation))
        if len(arguments) == 1:
            return arguments[0], optional
        return annotation, optional
    return annotation, False


def _nested_model(
    annotation: Any,
    /,
) -> type[BaseModel] | None:
    annotation, _optional = _without_none(annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _field_spec(
    path: str,
    /,
    section: str,
    annotation: Any,
) -> FieldSpec:
    if path in {"audio.input_device", "audio.output_device"}:
        return FieldSpec(path, section, "device", optional=True)
    if path == "placement":
        return FieldSpec(path, section, "placement", value_kind="collection")
    annotation, optional = _without_none(annotation)
    origin = get_origin(annotation)
    if origin is Literal:
        choices = tuple(str(value) for value in get_args(annotation))
        return FieldSpec(path, section, "select", choices)
    if annotation is bool:
        return FieldSpec(path, section, "bool", optional=optional)
    if origin in (list, dict, tuple, set):
        return FieldSpec(path, section, "value", optional=optional, value_kind="collection")
    if annotation is Path:
        return FieldSpec(path, section, "value", optional=optional, value_kind="path")
    if annotation in (int, float):
        return FieldSpec(path, section, "value", optional=optional, value_kind="number")
    return FieldSpec(path, section, "value", optional=optional)


def _build_fields() -> tuple[FieldSpec, ...]:
    fields: list[FieldSpec] = []

    def visit(
        model: type[BaseModel],
        /,
        prefix: str,
        section: str | None = None,
    ) -> None:
        for name, model_field in model.model_fields.items():
            if not prefix and name == "root":
                continue
            path = f"{prefix}.{name}" if prefix else name
            field_section = section or TOP_LEVEL_SECTIONS[name]
            nested = _nested_model(model_field.annotation)
            if nested is not None:
                visit(nested, path, field_section)
            else:
                fields.append(_field_spec(path, field_section, model_field.annotation))

    visit(Settings, "")
    return tuple(fields)


FIELDS = _build_fields()


def effective_value(
    settings: Settings,
    /,
    path: str,
) -> Any:
    """Return the value used by the validated runtime settings model."""
    node: Any = settings
    for part in path.split("."):
        node = getattr(node, part)
    return node


def field_description(
    specification: FieldSpec,
    /,
) -> str:
    """Return an operator-facing description for a field."""
    specific = FIELD_DESCRIPTIONS.get(specification.path)
    if specific:
        return _(specific)
    generic = VALUE_KIND_DESCRIPTIONS[specification.value_kind]
    return _(generic, path=specification.path)
