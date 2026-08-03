"""Local and remote configuration editor, reached with `kotonoha config`.

Local edits are written to config/local.yaml. Remote edits use the authenticated
management API on the A6000 and are written to remote-server.local.yaml there. Both
paths validate the complete Settings model before persistence.

The field list is reflected from the pydantic model. Collections remain leaf fields and
use YAML flow syntax, which keeps arbitrary mappings such as voice tables and LLM
profiles editable without inventing a second schema for the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

import yaml
from pydantic import BaseModel
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Footer, Header, Input, ListItem, ListView, Select, Static, Switch

from ..clients.base import ServiceError
from ..clients.config_admin import RemoteConfigClient, RemoteConfigSnapshot
from ..config import Settings, load_settings, local_config_path
from ..config_store import (
    ApplyResult,
    apply_changes,
    get_path,
    set_path,
    validate_candidate,
    write_local,
)
from ..i18n import CATALOGS, DEFAULT_LOCALE, LOCALE_NAMES, t


@dataclass(frozen=True)
class FieldSpec:
    """One editable leaf in Settings."""

    path: str
    section: str
    kind: str  # select | bool | value
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


def _without_none(annotation: Any) -> tuple[Any, bool]:
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


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    annotation, _ = _without_none(annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _field_spec(path: str, section: str, annotation: Any) -> FieldSpec:
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

    def visit(model: type[BaseModel], prefix: str, section: str | None = None) -> None:
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


def effective_value(settings: Settings, path: str) -> Any:
    """Read the value the runtime would use, through the pydantic model."""
    node: Any = settings
    for part in path.split("."):
        node = getattr(node, part)
    return node


def field_description(specification: FieldSpec) -> str:
    specific = f"cfg.f.{specification.path}"
    if specific in CATALOGS[DEFAULT_LOCALE]:
        return t(specific)
    return t(f"cfg.field.{specification.value_kind}", path=specification.path)


def _format_value(value: Any) -> str:
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


def _plain_value(value: Any) -> Any:
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

    def __init__(self, specification: FieldSpec, current: Any, from_override: bool, **kwargs):
        super().__init__(**kwargs)
        self.specification = specification
        self.current = current
        self.from_override = from_override
        self.editor: Select | Switch | Input | None = None

    @property
    def spec(self) -> FieldSpec:
        """Compatibility alias used by existing evaluation and TUI tests."""
        return self.specification

    def compose(self) -> ComposeResult:
        yield Static(self._label(), classes="fieldlabel")
        with Horizontal(classes="fieldrow"):
            yield self._make_editor()
        yield Static(field_description(self.specification), classes="fielddesc")

    def _label(self) -> Text:
        label = Text(self.specification.path, style="bold")
        if self.from_override:
            label.append(f"  [{t('cfg.modified')}]", style="yellow")
        return label

    def _make_editor(self):
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
        else:
            self.editor = Input(
                value=_format_value(self.current),
                placeholder="null" if specification.optional else "",
            )
        return self.editor

    def set_state(self, current: Any, from_override: bool) -> None:
        self.current = current
        self.from_override = from_override
        self.query_one(".fieldlabel", Static).update(self._label())
        if self.specification.kind == "bool":
            self.editor.value = bool(current)
        elif self.specification.kind == "select":
            choices = self.specification.choices
            self.editor.value = current if current in choices else choices[0]
        else:
            self.editor.value = _format_value(current)

    def value(self) -> Any:
        """Return the parsed editor value; Settings performs final type validation."""
        specification = self.specification
        if specification.kind == "bool":
            return bool(self.editor.value)
        if specification.kind == "select":
            return str(self.editor.value)

        raw = str(self.editor.value).strip()
        if not raw:
            if specification.optional:
                return None
            raise ValueError(t("cfg.value_required", path=specification.path))
        try:
            return yaml.safe_load(raw)
        except yaml.YAMLError as error:
            raise ValueError(f"{specification.path}: {error}") from error


class CategoryItem(ListItem):
    """One category in the navigation menu."""

    def __init__(self, section: str, modified: int):
        super().__init__(id=f"category-{section}")
        self.section = section
        self.modified = modified

    def compose(self) -> ComposeResult:
        yield Static(self._label())

    def _label(self) -> Text:
        label = Text(t(f"cfg.section.{self.section}"))
        if self.modified:
            label.append(f"  {self.modified}", style="yellow bold")
        return label

    def set_modified(self, modified: int) -> None:
        self.modified = modified
        self.query_one(Static).update(self._label())


class ConfigApp(App):
    CSS = """
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
    .fielddesc { color: $text-muted; }
    #status { height: 2; padding: 0 2; }
    Input { width: 100%; }
    Select { width: 100%; }
    """

    BINDINGS = [
        ("s", "save", ""),
        ("r", "reload", ""),
        ("m", "menu", ""),
        ("q", "quit", ""),
    ]

    def __init__(self, config_path: Path | None = None, local_path: Path | None = None):
        super().__init__()
        self.config_path = config_path
        self.local_path = local_path or local_config_path()
        self.client_settings = load_settings(config_path)
        self.settings = self.client_settings
        self.overrides = self._read_local_overrides()
        self.target = "local"
        self.remote_path: str | None = None
        self.remote_editable_paths: set[str] = set()
        self.remote_client: RemoteConfigClient | None = None
        self._changing_target = False
        self._rows: list[FieldRow] = []
        self.current_section = SECTIONS[0]
        self._bindings = BindingsMap(
            [
                Binding("s", "save", t("cfg.key.save")),
                Binding("r", "reload", t("cfg.key.reload")),
                Binding("m", "menu", t("cfg.key.menu")),
                Binding("q", "quit", t("cfg.key.quit")),
            ]
        )

    @property
    def local(self) -> dict:
        """Compatibility alias for the active override mapping."""
        return self.overrides

    @local.setter
    def local(self, value: dict) -> None:
        self.overrides = value

    def _read_local_overrides(self) -> dict:
        from ..config import read_yaml

        return read_yaml(self.local_path) if self.local_path.exists() else {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="workspace"):
            with Container(id="navigation"):
                yield Select(
                    [(t("cfg.target.local"), "local"), (t("cfg.target.remote"), "remote")],
                    value="local",
                    allow_blank=False,
                    id="target-select",
                )
                yield Static(t("cfg.categories"), id="navigation-title")
                with ListView(id="category-list"):
                    for section in SECTIONS:
                        yield CategoryItem(section, self._modified_count(section))
            with Container(id="content"):
                for section in SECTIONS:
                    with VerticalScroll(id=f"panel-{section}", classes="category-panel"):
                        yield Static(t(f"cfg.section.{section}"), classes="category-title")
                        for specification in FIELDS:
                            if specification.section != section:
                                continue
                            row = FieldRow(
                                specification,
                                effective_value(self.settings, specification.path),
                                get_path(self.overrides, specification.path) is not None,
                            )
                            self._rows.append(row)
                            yield row
        self.status = Static("", id="status")
        yield self.status
        yield Footer()

    def on_mount(self) -> None:
        self.title = t("cfg.title")
        self._update_subtitle()
        self._show_section(self.current_section)
        self.query_one("#category-list", ListView).index = 0

    async def on_unmount(self) -> None:
        if self.remote_client is not None:
            await self.remote_client.aclose()

    @on(Select.Changed, "#target-select")
    async def target_changed(self, event: Select.Changed) -> None:
        if self._changing_target or event.value == Select.BLANK:
            return
        requested = str(event.value)
        if requested == self.target:
            return
        if requested == "remote":
            await self._load_remote()
        else:
            self._load_local()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, CategoryItem):
            self._show_section(event.item.section)

    def _show_section(self, section: str) -> None:
        if section not in self._visible_sections():
            return
        self.current_section = section
        for candidate in SECTIONS:
            self.query_one(f"#panel-{candidate}").display = candidate == section

    async def _load_remote(self) -> None:
        self._say(t("cfg.remote.loading"), "dim")
        if self.remote_client is None:
            remote = self.client_settings.remote
            self.remote_client = RemoteConfigClient(remote.services.asr, remote)
        try:
            snapshot = await self.remote_client.read()
        except ServiceError as error:
            self._say(t("cfg.remote.failed", error=str(error)), "red")
            self._set_target_selector("local")
            return
        self._apply_remote_snapshot(snapshot)
        self._say(t("cfg.remote.loaded", path=snapshot.path), "green")

    def _load_local(self) -> None:
        self.target = "local"
        self._set_target_selector("local")
        self.settings = load_settings(self.config_path)
        self.overrides = self._read_local_overrides()
        self._refresh_rows()
        self._update_subtitle()
        self._say(t("cfg.reloaded"), "dim")

    def _apply_remote_snapshot(self, snapshot: RemoteConfigSnapshot) -> None:
        self.target = "remote"
        self._set_target_selector("remote")
        self.remote_path = snapshot.path
        self.settings = Settings.model_validate(snapshot.config)
        self.overrides = snapshot.overrides
        self.remote_editable_paths = set(snapshot.editable_paths)
        self._refresh_rows()
        self._update_subtitle()

    def _set_target_selector(self, target: str) -> None:
        self._changing_target = True
        self.query_one("#target-select", Select).value = target
        self._changing_target = False

    def _update_subtitle(self) -> None:
        path = self.local_path if self.target == "local" else self.remote_path or "remote"
        self.sub_title = t("cfg.subtitle", path=path)

    def _modified_count(self, section: str) -> int:
        return sum(
            get_path(self.overrides, specification.path) is not None
            for specification in FIELDS
            if specification.section == section and self._field_visible(specification)
        )

    def _field_visible(self, specification: FieldSpec) -> bool:
        return self.target == "local" or specification.path in self.remote_editable_paths

    def _visible_sections(self) -> tuple[str, ...]:
        return tuple(
            section
            for section in SECTIONS
            if any(
                specification.section == section and self._field_visible(specification)
                for specification in FIELDS
            )
        )

    def _refresh_visibility(self) -> None:
        visible_sections = self._visible_sections()
        for row in self._rows:
            row.display = self._field_visible(row.specification)
        for section in SECTIONS:
            self.query_one(f"#category-{section}", CategoryItem).display = (
                section in visible_sections
            )
        if self.current_section not in visible_sections:
            first_section = visible_sections[0]
            self.query_one("#category-list", ListView).index = SECTIONS.index(first_section)
            self._show_section(first_section)

    def _refresh_rows(self) -> None:
        for row in self._rows:
            row.set_state(
                effective_value(self.settings, row.specification.path),
                get_path(self.overrides, row.specification.path) is not None,
            )
        for section in SECTIONS:
            self.query_one(f"#category-{section}", CategoryItem).set_modified(
                self._modified_count(section)
            )
        self._refresh_visibility()

    def _collect_changes(self) -> dict[str, Any]:
        changes: dict[str, Any] = {}
        for row in self._rows:
            if not self._field_visible(row.specification):
                continue
            value = row.value()
            current = effective_value(self.settings, row.specification.path)
            if value != _plain_value(current):
                changes[row.specification.path] = value
        return changes

    async def action_save(self) -> None:
        try:
            changes = self._collect_changes()
        except ValueError as error:
            self._say(t("cfg.invalid", error=str(error)), "red")
            return
        if not changes:
            self._say(t("cfg.no_changes"), "dim")
            return

        if self.target == "remote":
            await self._save_remote(changes)
            return

        result = apply_changes(changes, self.config_path, self.local_path)
        if not result.written:
            self._say(t("cfg.invalid", error=result.error), "red")
            return
        self.settings = load_settings(self.config_path)
        self.overrides = self._read_local_overrides()
        self._refresh_rows()
        self._say(
            t("cfg.saved", count=len(changes), path=self.local_path)
            + "  "
            + t("cfg.restart_required"),
            "green",
        )

    async def _save_remote(self, changes: dict[str, Any]) -> None:
        if self.remote_client is None:
            self._say(t("cfg.remote.not_connected"), "red")
            return
        try:
            snapshot = await self.remote_client.update(changes)
        except ServiceError as error:
            self._say(t("cfg.remote.failed", error=str(error)), "red")
            return
        self._apply_remote_snapshot(snapshot)
        self._say(
            t("cfg.remote.saved", count=len(changes), path=snapshot.path)
            + "  "
            + t("cfg.remote.restart_required"),
            "green",
        )

    async def action_reload(self) -> None:
        if self.target == "remote":
            await self._load_remote()
        else:
            self._load_local()

    def action_menu(self) -> None:
        self.query_one("#category-list", ListView).focus()

    def _say(self, message: str, style: str) -> None:
        self.status.update(Text(message, style=style))


# The old names remain imports from this module because the test suite and external
# maintenance scripts used them before persistence moved into config_store.py.
HeadlessResult = ApplyResult


__all__ = [
    "FIELDS",
    "SECTIONS",
    "ConfigApp",
    "FieldSpec",
    "HeadlessResult",
    "apply_changes",
    "effective_value",
    "field_description",
    "get_path",
    "set_path",
    "validate_candidate",
    "write_local",
]
