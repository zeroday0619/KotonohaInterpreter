"""Configuration editor, reached with `kotonoha config`.

Edits are written to `config/local.yaml`, the third and highest-priority YAML layer.
`config/default.yaml` and any overlay passed with --config are never modified, so the
committed baseline stays intact and a device keeps its own values across updates.

The field list is curated rather than reflected from the pydantic model. A reflected
list would run to more than a hundred entries, most of which are not operator settings.
Fields here are the ones changed when deploying or tuning a unit.

A candidate configuration is validated by constructing `Settings` from the same layer
order the runtime uses. Nothing is written unless that succeeds, so the editor cannot
leave a device with a configuration that fails to load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, Input, Select, Static, Switch

from ..config import LOCAL_CONFIG, Settings, config_layers, deep_merge, load_settings, read_yaml
from ..i18n import LOCALE_NAMES, t


@dataclass(frozen=True)
class FieldSpec:
    """One editable setting.

    `path` is the dotted location in the merged YAML. The label shown is the path
    itself; engineers read it directly, and it is also what has to be typed into a
    YAML file, so translating it would be counterproductive. The description is
    localized under `cfg.f.<path>`.
    """

    path: str
    section: str
    kind: str  # select | bool | int | float | text
    choices: tuple[str, ...] = ()
    optional: bool = False  # empty input means null


LANGUAGE_CHOICES = ("auto", *LOCALE_NAMES)

FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("ui.language", "interface", "select", LANGUAGE_CHOICES),
    FieldSpec("session.mode", "session", "select", ("push_to_talk", "auto")),
    FieldSpec("session.routing", "session", "select", ("pair", "fixed", "broadcast")),
    FieldSpec("audio.input_device", "audio", "text", optional=True),
    FieldSpec("audio.output_device", "audio", "text", optional=True),
    FieldSpec("frontend.denoise.enabled", "frontend", "bool"),
    FieldSpec("frontend.vad.backend", "frontend", "select", ("silero_onnx", "energy")),
    FieldSpec("frontend.vad.threshold", "frontend", "float"),
    FieldSpec("frontend.vad.preroll_ms", "frontend", "int"),
    FieldSpec("frontend.vad.silence_ms", "frontend", "int"),
    FieldSpec("asr.backend", "models", "select", ("transformers", "vllm")),
    FieldSpec("asr.n_best", "models", "int"),
    FieldSpec("asr_verify.mode", "models", "select", ("conditional", "always")),
    FieldSpec("llm.profile", "models", "select", ("moe", "dense")),
    FieldSpec("tts.backend", "models", "select", ("qwen3", "melo")),
    FieldSpec("perf_mode", "remote", "select", ("onboard", "hybrid", "remote")),
    FieldSpec("remote.enabled", "remote", "bool"),
    FieldSpec("remote.services.llm", "remote", "text"),
    FieldSpec("remote.services.asr", "remote", "text"),
    FieldSpec("remote.services.asr_verify", "remote", "text"),
    FieldSpec("remote.services.tts", "remote", "text"),
    FieldSpec("remote.audio_encoding", "remote", "select", ("s16le", "f32le")),
    FieldSpec("remote.failover_after", "remote", "int"),
)

SECTIONS = ("interface", "session", "audio", "frontend", "models", "remote")


# -- dotted-path helpers ---------------------------------------------------
def get_path(data: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data


def set_path(data: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        data = data.setdefault(part, {})
    data[parts[-1]] = value


def effective_value(settings: Settings, path: str) -> Any:
    """Read the value the runtime would use, through the pydantic model."""
    node: Any = settings
    for part in path.split("."):
        node = getattr(node, part)
    return node


# -- validation and persistence -------------------------------------------
def validate_candidate(config_path: Path | None, local: dict) -> str | None:
    """Return None when the candidate loads, otherwise a one-line reason."""
    merged: dict = {}
    for layer in config_layers(config_path):
        merged = deep_merge(merged, read_yaml(layer))
    merged = deep_merge(merged, local)
    try:
        Settings(**merged)
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        return f"{loc}: {first['msg']}"
    except Exception as e:  # noqa: BLE001
        return repr(e)
    return None


def write_local(path: Path, local: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Written by `kotonoha config`. Host-specific overrides.\n"
        "# This is the third configuration layer and overrides config/default.yaml\n"
        "# and any overlay passed with --config.\n\n"
    )
    path.write_text(
        header + yaml.safe_dump(local, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


# -- widgets ---------------------------------------------------------------
class FieldRow(Static):
    """One setting: path, editor, description, and an origin marker."""

    def __init__(self, spec: FieldSpec, current: Any, from_local: bool, **kw):
        super().__init__(**kw)
        self.spec = spec
        self.current = current
        self.from_local = from_local
        self.editor: Select | Switch | Input | None = None

    def compose(self) -> ComposeResult:
        yield Static(self._label(), classes="fieldlabel")
        with Horizontal(classes="fieldrow"):
            yield self._make_editor()
        yield Static(t(f"cfg.f.{self.spec.path}"), classes="fielddesc")

    def _label(self) -> Text:
        text = Text(self.spec.path, style="bold")
        if self.from_local:
            text.append(f"  [{t('cfg.modified')}]", style="yellow")
        return text

    def _make_editor(self):
        s = self.spec
        if s.kind == "bool":
            self.editor = Switch(value=bool(self.current))
        elif s.kind == "select":
            options = [(c, c) for c in s.choices]
            value = self.current if self.current in s.choices else s.choices[0]
            self.editor = Select(options, value=value, allow_blank=False)
        else:
            self.editor = Input(
                value="" if self.current is None else str(self.current),
                placeholder="null" if s.optional else "",
            )
        return self.editor

    def value(self) -> Any:
        """Parsed widget value. Raises ValueError when the text does not parse."""
        s = self.spec
        if s.kind == "bool":
            return bool(self.editor.value)
        if s.kind == "select":
            return str(self.editor.value)

        raw = str(self.editor.value).strip()
        if not raw:
            if s.optional:
                return None
            raise ValueError(f"{s.path}: value required")
        if s.kind == "int":
            return int(raw)
        if s.kind == "float":
            return float(raw)
        # A device may be given either an index or a name.
        if s.optional and raw.lstrip("-").isdigit():
            return int(raw)
        return raw


class ConfigApp(App):
    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; padding: 0 2; }
    .section { padding: 1 0 0 0; text-style: bold underline; color: $accent; }
    .fieldlabel { padding: 1 0 0 0; }
    .fieldrow { height: 3; }
    .fielddesc { color: $text-muted; }
    #status { height: 2; padding: 0 2; }
    Input { width: 60; }
    Select { width: 60; }
    """

    # Descriptions are localized per instance below. BINDINGS is a class attribute
    # evaluated at import time, before --lang has been parsed.
    BINDINGS = [
        ("s", "save", ""),
        ("r", "reload", ""),
        ("q", "quit", ""),
    ]

    def __init__(self, config_path: Path | None = None, local_path: Path | None = None):
        super().__init__()
        self.config_path = config_path
        self.local_path = local_path or LOCAL_CONFIG
        self.settings = load_settings(config_path)
        self.local = read_yaml(self.local_path) if self.local_path.exists() else {}
        self._rows: list[FieldRow] = []
        # Replace the map rather than calling bind() on it: Textual builds the map
        # from the class attribute, so mutating it would leak one instance's locale
        # into the next.
        self._bindings = BindingsMap(
            [
                Binding("s", "save", t("cfg.key.save")),
                Binding("r", "reload", t("cfg.key.reload")),
                Binding("q", "quit", t("cfg.key.quit")),
            ]
        )

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="body"):
            for section in SECTIONS:
                yield Static(t(f"cfg.section.{section}"), classes="section")
                for spec in FIELDS:
                    if spec.section != section:
                        continue
                    row = FieldRow(
                        spec,
                        effective_value(self.settings, spec.path),
                        get_path(self.local, spec.path) is not None,
                    )
                    self._rows.append(row)
                    yield row
        self.status = Static("", id="status")
        yield self.status
        yield Footer()

    def on_mount(self) -> None:
        self.title = t("cfg.title")
        self.sub_title = t("cfg.subtitle", path=self.local_path)

    # -- actions -----------------------------------------------------------
    def action_save(self) -> None:
        candidate = dict(self.local)
        changed = 0
        try:
            for row in self._rows:
                value = row.value()
                if value != effective_value(self.settings, row.spec.path):
                    set_path(candidate, row.spec.path, value)
                    changed += 1
        except ValueError as e:
            self._say(t("cfg.invalid", error=str(e)), "red")
            return

        if not changed:
            self._say(t("cfg.no_changes"), "dim")
            return

        problem = validate_candidate(self.config_path, candidate)
        if problem:
            self._say(t("cfg.invalid", error=problem), "red")
            return

        write_local(self.local_path, candidate)
        self.local = candidate
        self.settings = load_settings(self.config_path)
        self._say(
            t("cfg.saved", count=changed, path=self.local_path)
            + "  "
            + t("cfg.restart_required"),
            "green",
        )

    def action_reload(self) -> None:
        self.settings = load_settings(self.config_path)
        self.local = read_yaml(self.local_path) if self.local_path.exists() else {}
        for row in self._rows:
            current = effective_value(self.settings, row.spec.path)
            if row.spec.kind == "bool":
                row.editor.value = bool(current)
            elif row.spec.kind == "select":
                row.editor.value = current if current in row.spec.choices else row.spec.choices[0]
            else:
                row.editor.value = "" if current is None else str(current)
        self._say(t("cfg.reloaded"), "dim")

    def _say(self, message: str, style: str) -> None:
        self.status.update(Text(message, style=style))


@dataclass
class HeadlessResult:
    """Result of a non-interactive save, used by the tests."""

    written: bool
    error: str | None = None
    changed: list[str] = field(default_factory=list)


def apply_changes(
    changes: dict[str, Any],
    config_path: Path | None = None,
    local_path: Path | None = None,
) -> HeadlessResult:
    """Validate and persist changes without starting the interface.

    The editor and this function share `validate_candidate` and `write_local`, so the
    guarantee that an invalid configuration is never written is testable without
    driving widgets.
    """
    target = local_path or LOCAL_CONFIG
    candidate = read_yaml(target) if target.exists() else {}
    for path, value in changes.items():
        set_path(candidate, path, value)

    problem = validate_candidate(config_path, candidate)
    if problem:
        return HeadlessResult(written=False, error=problem)

    write_local(target, candidate)
    return HeadlessResult(written=True, changed=sorted(changes))
