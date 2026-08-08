"""Local and remote configuration editor, reached with `kotonoha config`.

Local edits are written to config/local.yaml. Remote edits use the authenticated
management API on the A6000 and are written to remote-server.local.yaml there. Both
paths validate the complete Settings model before persistence.

The field list is reflected from the pydantic model. Collections remain leaf fields and
use YAML flow syntax, which keeps arbitrary mappings such as voice tables and LLM
profiles editable without inventing a second schema for the interface.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Any, ClassVar, Literal, Union, get_args, get_origin

import yaml
from pydantic import BaseModel
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    ListItem,
    ListView,
    Select,
    Static,
    Switch,
)

from kotonoha._config import (
    PERF_PLACEMENT,
    ROLES,
    Settings,
    load_settings,
    local_config_path,
)
from kotonoha._config_store import (
    apply_changes,
    get_path,
)
from kotonoha._i18n import LOCALE_NAMES, N_, _
from kotonoha._typing import override
from kotonoha.audio._devices import (
    AudioDevice,
    AudioProbeResult,
    probe_audio_devices,
    query_audio_devices,
)
from kotonoha.clients._base import ServiceError
from kotonoha.clients._config_admin import RemoteConfigClient, RemoteConfigSnapshot


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One editable leaf in Settings."""

    path: str
    section: str
    kind: str  # select | bool | device | placement | value
    choices: tuple[str, ...] = ()
    optional: bool = False
    value_kind: str = "text"  # text | number | path | collection


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
    annotation, _ = _without_none(annotation)
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


# Category titles. N_ marks them for extraction without translating at import
# time, so the active locale is applied when a label is rendered.
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

ROLE_LABELS: dict[str, str] = {
    "asr": N_("ASR"),
    "asr_verify": N_("Verification ASR"),
    "llm": N_("LLM"),
    "tts": N_("TTS"),
}

# Fallback description, by value kind, for a field with no specific note.
VALUE_KIND_DESCRIPTIONS: dict[str, str] = {
    "collection": N_("YAML list or mapping for {path}."),
    "number": N_("Numeric value for {path}."),
    "path": N_("Filesystem path for {path}."),
    "text": N_("Text value for {path}."),
}

# Per-setting notes, keyed by dotted configuration path. A path absent here
# falls back to VALUE_KIND_DESCRIPTIONS.
FIELD_DESCRIPTIONS: dict[str, str] = {
    "asr.backend": N_("vLLM is the default; Transformers is the fallback."),
    "asr.n_best": N_("Hypotheses returned per utterance. The correction pass consumes all."),
    "asr_verify.mode": N_(
            "conditional verifies only low-confidence turns; always verifies every turn."
    ),
    "audio.input_device": N_(
        "Stable microphone name and host API. Empty selects the system default."
    ),
    "audio.output_device": N_(
        "Stable speaker name and host API. Empty selects the system default."
    ),
    "frontend.denoise.enabled": N_("DeepFilterNet3 noise suppression."),
    "frontend.vad.backend": N_("silero_onnx on the device; energy is a workstation fallback."),
    "frontend.vad.preroll_ms": N_(
            "Audio retained before speech onset. Below 200 ms the first syllable is clipped."
    ),
    "frontend.vad.silence_ms": N_("Silence required before end-of-utterance."),
    "frontend.vad.threshold": N_("Speech onset probability, 0 to 1."),
    "llm.profile": N_(
            "Selects the target-specific TranslateGemma translation model."
    ),
    "logging.console": N_(
            "Show structured application logs in the TUI. Model services emit JSON to their "
            "console."
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
    "remote.enabled": N_(
            "When false, every role runs locally regardless of perf_mode."
    ),
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
    "tts.backend": N_(
            "vllm_omni requires successful vLLM-Omni Spike 2 validation."
    ),
    "ui.language": N_("Interface language. auto follows the system locale."),
    "ui.refresh_hz": N_(
            "Maximum TUI frame scheduler rate. Idle frames do not repaint the terminal."
    ),
}


def effective_value(
    settings: Settings,
    /,
    path: str,
) -> Any:
    """Read the value the runtime would use, through the pydantic model."""
    node: Any = settings
    for part in path.split("."):
        node = getattr(node, part)
    return node


def field_description(
    specification: FieldSpec,
    /,
) -> str:
    specific = FIELD_DESCRIPTIONS.get(specification.path)
    if specific:
        return _(specific)
    generic = VALUE_KIND_DESCRIPTIONS[specification.value_kind]
    return _(generic, path=specification.path)


def _format_value(
    value: Any,
    /,
) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (BaseModel, dict, list, tuple, set)):
        return yaml.safe_dump(
            _plain_value(value),
            allow_unicode=True,
            default_flow_style=True,
            width=10_000,
        ).strip()
    return str(value)


def _plain_value(
    value: Any,
    /,
) -> Any:
    """Convert nested pydantic values into types accepted by YAML and JSON."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class FieldRow(Static):
    """One setting: path, editor, description, and an origin marker."""
    __slots__: ClassVar[tuple[str, ...]] = ()

    specification: FieldSpec
    current: Any
    from_override: bool
    editor: Select | Switch | Input | None
    device_options: tuple[tuple[str, int | str], ...]
    placement_defaults: dict[str, str]
    placement_initial: dict[str, str]
    placement_editors: dict[str, Select]

    @override
    def __init__(
        self,
        /,
        specification: FieldSpec,
        current: Any,
        from_override: bool,
        device_options: tuple[tuple[str, int | str], ...] = (),
        placement_defaults: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.specification = specification
        self.current = current
        self.from_override = from_override
        self.editor = None
        self.device_options = device_options
        self.placement_defaults = placement_defaults or {}
        self.placement_initial = self._placement_state(current)
        self.placement_editors = {}

    @override
    def compose(
        self,
        /,
    ) -> ComposeResult:
        yield Static(self._label(), classes="fieldlabel")
        with Horizontal(classes="fieldrow"):
            if self.specification.kind == "placement":
                for role in ROLES:
                    with Vertical(classes="placement-control"):
                        yield Static(_(ROLE_LABELS[role]), classes="placement-label")
                        editor = self._make_placement_editor(role)
                        self.placement_editors[role] = editor
                        yield editor
            else:
                yield self._make_editor()
        yield Static(field_description(self.specification), classes="fielddesc")

    def _label(
        self,
        /,
    ) -> Text:
        label = Text(self.specification.path, style="bold")
        if self.from_override:
            label.append(f"  [{_('modified')}]", style="yellow")
        return label

    def _make_editor(
        self,
        /,
    ) -> Any:
        specification = self.specification
        if specification.kind == "bool":
            self.editor = Switch(value=bool(self.current))
        elif specification.kind == "select":
            options = [(choice, choice) for choice in specification.choices]
            value = (
                self.current
                if self.current in specification.choices
                else specification.choices[0]
            )
            self.editor = Select(options, value=value, allow_blank=False)
        elif specification.kind == "device":
            self.editor = Select(
                self._device_options(self.current),
                value=self._device_value(self.current),
                prompt=_("Select an audio device"),
                allow_blank=False,
            )
        elif specification.kind == "placement":
            self.editor = None
        else:
            self.editor = Input(
                value=_format_value(self.current),
                placeholder="null" if specification.optional else "",
                password=specification.path == "remote.token",
            )
        return self.editor

    def set_state(
        self,
        /,
        current: Any,
        from_override: bool,
        placement_defaults: dict[str, str] | None = None,
    ) -> None:
        self.current = current
        self.from_override = from_override
        if placement_defaults is not None:
            self.placement_defaults = placement_defaults
        self.query_one(".fieldlabel", Static).update(self._label())
        if self.specification.kind == "bool":
            self.editor.value = bool(current)
        elif self.specification.kind == "select":
            choices = self.specification.choices
            self.editor.value = current if current in choices else choices[0]
        elif self.specification.kind == "device":
            self.editor.set_options(self._device_options(current))
            self.editor.value = self._device_value(current)
        elif self.specification.kind == "placement":
            self.placement_initial = self._placement_state(current)
            for role, editor in self.placement_editors.items():
                editor.value = self.placement_initial[role]
        else:
            self.editor.value = _format_value(current)

    def value(
        self,
        /,
    ) -> Any:
        """Return the parsed editor value; Settings performs final type validation."""
        specification = self.specification
        if specification.kind == "bool":
            return bool(self.editor.value)
        if specification.kind == "select":
            return str(self.editor.value)
        if specification.kind == "device":
            value = self.editor.value
            return None if value == "" else value
        if specification.kind == "placement":
            selected = {role: str(editor.value) for role, editor in self.placement_editors.items()}
            if selected == self.placement_initial:
                return dict(self.current or {})
            return {
                role: side
                for role, side in selected.items()
                if side != self.placement_defaults.get(role, "local")
            }

        raw = str(self.editor.value).strip()
        if not raw:
            if specification.optional:
                return None
            raise ValueError(_("{path}: value required", path=specification.path))
        try:
            return yaml.safe_load(raw)
        except yaml.YAMLError as error:
            raise ValueError(f"{specification.path}: {error}") from error

    def _device_options(
        self,
        current: Any,
        /,
    ) -> tuple[tuple[str, int | str], ...]:
        options = list(self.device_options)
        current_value = self._device_value(current)
        if current_value and current_value not in {value for _, value in options}:
            options.insert(
                1,
                (
                    _("Configured device: {device}", device=current_value),
                    current_value,
                ),
            )
        return tuple(options)

    def _device_value(
        self,
        current: Any,
        /,
    ) -> int | str:
        return "" if current is None else current

    def _make_placement_editor(
        self,
        role: str,
        /,
    ) -> Select:
        return Select(
            [
                (_("Local"), "local"),
                (_("Remote"), "remote"),
            ],
            value=self._placement_state(self.current)[role],
            allow_blank=False,
            id=f"placement-{role}",
        )

    def _placement_state(
        self,
        current: Any,
        /,
    ) -> dict[str, str]:
        configured = current if isinstance(current, dict) else {}
        return {
            role: str(configured.get(role, self.placement_defaults.get(role, "local")))
            for role in ROLES
        }


class CategoryItem(ListItem):
    """One category in the navigation menu."""
    __slots__: ClassVar[tuple[str, ...]] = ()

    section: str
    modified: int

    @override
    def __init__(
        self,
        /,
        section: str,
        modified: int,
    ) -> None:
        super().__init__(id=f"category-{section}")
        self.section = section
        self.modified = modified

    @override
    def compose(
        self,
        /,
    ) -> ComposeResult:
        yield Static(self._label())

    def _label(
        self,
        /,
    ) -> Text:
        label = Text(_(SECTION_LABELS[self.section]))
        if self.modified:
            label.append(f"  {self.modified}", style="yellow bold")
        return label

    def set_modified(
        self,
        /,
        modified: int,
    ) -> None:
        self.modified = modified
        self.query_one(Static).update(self._label())


class ConfigApp(App):
    __slots__: ClassVar[tuple[str, ...]] = ()
    config_path: Path | None
    local_path: Path
    client_settings: Settings
    settings: Settings
    overrides: dict[str, Any]
    target: str
    remote_path: str | None
    remote_editable_paths: set[str]
    remote_client: RemoteConfigClient | None
    current_section: str
    status: Static
    title: str
    sub_title: str
    _changing_target: bool
    _rows: list[FieldRow]
    _bindings: BindingsMap
    audio_devices: tuple[AudioDevice, ...]
    audio_device_error: str | None

    CSS: ClassVar[str] = """
    Screen { layout: vertical; }
    #workspace { height: 1fr; }
    #navigation {
        width: 30;
        height: 1fr;
        border-right: solid $primary;
        background: $surface;
    }
    #target-select { width: 26; margin: 0 2 1 2; }
    #navigation-title {
        height: 3;
        padding: 1 2 0 2;
        text-style: bold;
        color: $text-muted;
    }
    #category-list { height: 1fr; }
    #category-list ListItem { padding: 1 2; }
    #category-list ListItem.-highlight { background: $accent; color: $text; }
    #content { width: 1fr; height: 1fr; }
    .category-panel { width: 1fr; height: 1fr; padding: 0 3 1 3; }
    .category-title {
        padding: 1 0;
        text-style: bold;
        color: $accent;
        border-bottom: solid $primary-darken-2;
    }
    .fieldlabel { padding: 1 0 0 0; }
    .fieldrow { height: 3; }
    .placement-control { width: 1fr; }
    .placement-label { height: 1; color: $text-muted; }
    .fielddesc { color: $text-muted; }
    #status { height: 2; padding: 0 2; }
    #audio-test-controls { height: 3; }
    #audio-test { width: 32; }
    #audio-test-status { padding: 1 2; color: $text-muted; }
    Input { width: 100%; }
    Select { width: 100%; }
    """

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("s", "save", ""),
        ("r", "reload", ""),
        ("m", "menu", ""),
        ("q", "quit", ""),
    ]

    @override
    def __init__(
        self,
        /,
        config_path: Path | None = None,
        local_path: Path | None = None,
        settings: Settings | None = None,
        overrides: dict | None = None,
    ) -> None:
        super().__init__()
        self.config_path = config_path
        self.local_path = local_path or local_config_path()
        self.client_settings = settings or load_settings(config_path)
        self.settings = self.client_settings
        self.overrides = overrides if overrides is not None else self._read_local_overrides()
        self.target = "local"
        self.remote_path = None
        self.remote_editable_paths = set()
        self.remote_client = None
        self._changing_target = False
        self._rows = []
        try:
            self.audio_devices = query_audio_devices()
            self.audio_device_error = None
        except Exception as error:  # noqa: BLE001 - the editor remains usable without PortAudio
            self.audio_devices = ()
            self.audio_device_error = str(error)
        self.current_section = SECTIONS[0]
        self._bindings = BindingsMap(
            [
                Binding("s", "save", _("Save")),
                Binding("r", "reload", _("Reload")),
                Binding("m", "menu", _("Categories")),
                Binding("q", "quit", _("Quit")),
            ]
        )

    def _read_local_overrides(
        self,
        /,
    ) -> dict:
        from kotonoha._config import read_yaml

        return read_yaml(self.local_path) if self.local_path.exists() else {}

    @override
    def compose(
        self,
        /,
    ) -> ComposeResult:
        yield Header()
        with Horizontal(id="workspace"):
            with Container(id="navigation"):
                yield Select(
                    [(_("Local device"), "local"), (_("Remote A6000"), "remote")],
                    value="local",
                    allow_blank=False,
                    id="target-select",
                )
                yield Static(_("Categories"), id="navigation-title")
                with ListView(id="category-list"):
                    for section in SECTIONS:
                        yield CategoryItem(section, self._modified_count(section))
            with Container(id="content"):
                for section in SECTIONS:
                    with VerticalScroll(id=f"panel-{section}", classes="category-panel"):
                        yield Static(_(SECTION_LABELS[section]), classes="category-title")
                        for specification in FIELDS:
                            if specification.section != section:
                                continue
                            row = FieldRow(
                                specification,
                                effective_value(self.settings, specification.path),
                                get_path(self.overrides, specification.path) is not None,
                                self._device_options(specification),
                                PERF_PLACEMENT[self.settings.perf_mode],
                            )
                            self._rows.append(row)
                            yield row
                        if section == "audio":
                            with Horizontal(id="audio-test-controls"):
                                yield Button(
                                    _("Test selected devices"),
                                    id="audio-test",
                                )
                                yield Static("", id="audio-test-status")
        self.status = Static("", id="status")
        yield self.status
        yield Footer()

    def on_mount(
        self,
        /,
    ) -> None:
        self.title = _("Kotonoha configuration")
        self._update_subtitle()
        self._show_section(self.current_section)
        self.query_one("#category-list", ListView).index = 0
        if self.audio_device_error:
            self._say(
                _(
                    "Audio device list unavailable: {error}",
                    error=self.audio_device_error,
                ),
                "yellow",
            )

    async def on_unmount(
        self,
        /,
    ) -> None:
        if self.remote_client is not None:
            await self.remote_client.aclose()

    @on(Button.Pressed, "#audio-test")
    async def audio_test_pressed(
        self,
        /,
        event: Button.Pressed,
    ) -> None:
        del event
        if self.target != "local":
            self._say(_("Audio device testing is available only for the local device"), "yellow")
            return
        try:
            input_device = self._row_value("audio.input_device")
            output_device = self._row_value("audio.output_device")
        except ValueError as error:
            self._say(str(error), "red")
            return

        button = self.query_one("#audio-test", Button)
        button.disabled = True
        self._say(_("Speak while the microphone and speaker test runs..."), "yellow")
        try:
            result = await asyncio.to_thread(
                probe_audio_devices,
                input_device,
                output_device,
                capture_sample_rate=self.settings.audio.capture_sample_rate,
                playback_sample_rate=self.settings.audio.playback_sample_rate,
                channels=self.settings.audio.channels,
            )
        finally:
            button.disabled = False
        self._report_audio_test(result)

    @on(Select.Changed, "#target-select")
    async def target_changed(
        self,
        /,
        event: Select.Changed,
    ) -> None:
        if self._changing_target or event.value == Select.BLANK:
            return
        requested = str(event.value)
        if requested == self.target:
            return
        if requested == "remote":
            await self._load_remote()
        else:
            await self._load_local()

    def on_list_view_selected(
        self,
        /,
        event: ListView.Selected,
    ) -> None:
        if isinstance(event.item, CategoryItem):
            self._show_section(event.item.section)

    def _show_section(
        self,
        /,
        section: str,
    ) -> None:
        if section not in self._visible_sections():
            return
        self.current_section = section
        for candidate in SECTIONS:
            self.query_one(f"#panel-{candidate}").display = candidate == section

    async def _load_remote(
        self,
        /,
    ) -> None:
        self._say(_("Loading configuration from the remote server"), "dim")
        if self.remote_client is None:
            remote = self.client_settings.remote
            self.remote_client = RemoteConfigClient(remote.services.asr, remote)
        try:
            snapshot = await self.remote_client.read()
        except ServiceError as error:
            self._say(_("Remote configuration failed: {error}", error=str(error)), "red")
            self._set_target_selector("local")
            return
        self._apply_remote_snapshot(snapshot)
        self._say(_("Loaded remote configuration from {path}", path=snapshot.path), "green")

    async def _load_local(
        self,
        /,
    ) -> None:
        self.target = "local"
        self._set_target_selector("local")
        self.settings = await asyncio.to_thread(load_settings, self.config_path)
        self.overrides = await asyncio.to_thread(self._read_local_overrides)
        self._refresh_rows()
        self._update_subtitle()
        self._say(_("Reloaded from disk"), "dim")

    def _apply_remote_snapshot(
        self,
        /,
        snapshot: RemoteConfigSnapshot,
    ) -> None:
        self.target = "remote"
        self._set_target_selector("remote")
        self.remote_path = snapshot.path
        self.settings = Settings.model_validate(snapshot.config)
        self.overrides = snapshot.overrides
        self.remote_editable_paths = set(snapshot.editable_paths)
        self._refresh_rows()
        self._update_subtitle()

    def _set_target_selector(
        self,
        /,
        target: str,
    ) -> None:
        self._changing_target = True
        self.query_one("#target-select", Select).value = target
        self._changing_target = False

    def _update_subtitle(
        self,
        /,
    ) -> None:
        path = self.local_path if self.target == "local" else self.remote_path or "remote"
        self.sub_title = _("Changes are written to {path}", path=path)

    def _modified_count(
        self,
        /,
        section: str,
    ) -> int:
        return sum(
            get_path(self.overrides, specification.path) is not None
            for specification in FIELDS
            if specification.section == section and self._field_visible(specification)
        )

    def _row_value(
        self,
        path: str,
        /,
    ) -> Any:
        row = next(
            (candidate for candidate in self._rows if candidate.specification.path == path),
            None,
        )
        if row is None:
            raise ValueError(_("Audio setting is not available: {path}", path=path))
        return row.value()

    def _report_audio_test(
        self,
        result: AudioProbeResult,
        /,
    ) -> None:
        if result.ok:
            self._say(_("Audio device test passed"), "green")
            level = (
                f"{result.input_peak_dbfs:.1f} dBFS"
                if result.input_peak_dbfs is not None
                else _("unknown level")
            )
            self.query_one("#audio-test-status", Static).update(
                Text(
                    _(
                        "Microphone signal detected at {level}; speaker test tone sent",
                        level=level,
                    ),
                    style="green",
                )
            )
            return
        messages: list[str] = []
        if result.input_error:
            messages.append(_("Input device test failed: {error}", error=result.input_error))
        elif result.input_signal_detected is False:
            messages.append(_("Input stream opened but no microphone signal was detected"))
        if result.output_error:
            messages.append(_("Output device test failed: {error}", error=result.output_error))
        message = " ".join(messages)
        self._say(message, "red")
        self.query_one("#audio-test-status", Static).update(Text(message, style="red"))

    def _field_visible(
        self,
        /,
        specification: FieldSpec,
    ) -> bool:
        return self.target == "local" or specification.path in self.remote_editable_paths

    def _visible_sections(
        self,
        /,
    ) -> tuple[str, ...]:
        return tuple(
            section
            for section in SECTIONS
            if any(
                specification.section == section and self._field_visible(specification)
                for specification in FIELDS
            )
        )

    def _refresh_visibility(
        self,
        /,
    ) -> None:
        visible_sections = self._visible_sections()
        for row in self._rows:
            row.display = self._field_visible(row.specification)
        for section in SECTIONS:
            self.query_one(f"#category-{section}", CategoryItem).display = (
                section in visible_sections
            )
        self.query_one("#audio-test-controls").display = self.target == "local"
        if self.current_section not in visible_sections:
            first_section = visible_sections[0]
            self.query_one("#category-list", ListView).index = SECTIONS.index(first_section)
            self._show_section(first_section)

    def _device_options(
        self,
        specification: FieldSpec,
        /,
    ) -> tuple[tuple[str, int | str], ...]:
        if specification.kind != "device":
            return ()
        direction = "input" if specification.path == "audio.input_device" else "output"
        options: list[tuple[str, int | str]] = [(_("System default"), "")]
        options.extend(
            (device.label, device.selector)
            for device in self.audio_devices
            if (direction == "input" and device.input_channels > 0)
            or (direction == "output" and device.output_channels > 0)
        )
        return tuple(options)

    def _refresh_rows(
        self,
        /,
    ) -> None:
        for row in self._rows:
            row.set_state(
                effective_value(self.settings, row.specification.path),
                get_path(self.overrides, row.specification.path) is not None,
                PERF_PLACEMENT[self.settings.perf_mode],
            )
        for section in SECTIONS:
            self.query_one(f"#category-{section}", CategoryItem).set_modified(
                self._modified_count(section)
            )
        self._refresh_visibility()

    def _collect_changes(
        self,
        /,
    ) -> dict[str, Any]:
        changes: dict[str, Any] = {}
        for row in self._rows:
            if not self._field_visible(row.specification):
                continue
            value = row.value()
            current = effective_value(self.settings, row.specification.path)
            if value != _plain_value(current):
                changes[row.specification.path] = value
        return changes

    async def action_save(
        self,
        /,
    ) -> None:
        try:
            changes = self._collect_changes()
        except ValueError as error:
            self._say(
                _(
                    "Rejected because the configuration would be invalid: {error}",
                    error=str(error),
                ),
                "red",
            )
            return
        if not changes:
            self._say(_("No changes to save"), "dim")
            return

        if self.target == "remote":
            await self._save_remote(changes)
            return

        result = await asyncio.to_thread(
            apply_changes,
            changes,
            self.config_path,
            self.local_path,
        )
        if not result.written:
            self._say(
                _(
                    "Rejected because the configuration would be invalid: {error}",
                    error=result.error,
                ),
                "red",
            )
            return
        self.settings = await asyncio.to_thread(load_settings, self.config_path)
        self.overrides = await asyncio.to_thread(self._read_local_overrides)
        self._refresh_rows()
        self._say(
            _("Saved {count} settings to {path}", count=len(changes), path=self.local_path)
            + "  "
            + _("Restart the interpreter for these values to take effect"),
            "green",
        )

    async def _save_remote(
        self,
        /,
        changes: dict[str, Any],
    ) -> None:
        if self.remote_client is None:
            self._say(_("The remote configuration service is not connected"), "red")
            return
        try:
            snapshot = await self.remote_client.update(changes)
        except ServiceError as error:
            self._say(_("Remote configuration failed: {error}", error=str(error)), "red")
            return
        self._apply_remote_snapshot(snapshot)
        self._say(
            _(
                "Saved {count} settings to the remote file {path}",
                count=len(changes),
                path=snapshot.path,
            )
            + "  "
            + _("Restart the remote services to apply these values"),
            "green",
        )

    async def action_reload(
        self,
        /,
    ) -> None:
        if self.target == "remote":
            await self._load_remote()
        else:
            await self._load_local()

    def action_menu(
        self,
        /,
    ) -> None:
        self.query_one("#category-list", ListView).focus()

    def _say(
        self,
        /,
        message: str,
        style: str,
    ) -> None:
        self.status.update(Text(message, style=style))
