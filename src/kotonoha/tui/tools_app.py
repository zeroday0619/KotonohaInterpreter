"""Command runner for operations that do not have a dedicated terminal interface."""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
from asyncio.subprocess import DEVNULL, PIPE, STDOUT, Process
from collections.abc import Mapping
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Select, Static

from ..i18n import current_locale, t

OPERATIONS = (
    "replay",
    "devices",
    "serve",
    "doctor",
    "netcheck",
    "glossary_import",
    "glossary_list",
    "completion_show",
    "completion_install",
)

OPERATION_FIELDS: dict[str, tuple[str, ...]] = {
    "replay": ("wav", "replay-seconds"),
    "devices": (),
    "serve": ("service", "host", "port"),
    "doctor": (),
    "netcheck": ("samples", "netcheck-seconds"),
    "glossary_import": ("glossary-path",),
    "glossary_list": (),
    "completion_show": (),
    "completion_install": (),
}


class ToolInputError(ValueError):
    """Report invalid command input before a child process starts."""


def _positive_integer(value: str, field: str, *, maximum: int | None = None) -> str:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ToolInputError(t("tools.error.positive_integer", field=field)) from error
    if parsed <= 0:
        raise ToolInputError(t("tools.error.positive_integer", field=field))
    if maximum is not None and parsed > maximum:
        raise ToolInputError(t("tools.error.maximum", field=field, maximum=maximum))
    return str(parsed)


def _positive_number(value: str, field: str) -> str:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ToolInputError(t("tools.error.positive_number", field=field)) from error
    if parsed <= 0:
        raise ToolInputError(t("tools.error.positive_number", field=field))
    return str(parsed)


def _existing_file(value: str, field: str) -> str:
    path = Path(value).expanduser()
    if not value.strip() or not path.is_file():
        raise ToolInputError(t("tools.error.file", field=field))
    return str(path)


def build_tool_command(
    operation: str,
    values: Mapping[str, str],
    config_path: Path | None = None,
) -> list[str]:
    """Build a validated CLI invocation without involving a shell."""
    if operation not in OPERATIONS:
        raise ToolInputError(t("tools.error.operation"))

    command = [sys.executable, "-m", "kotonoha.cli"]
    if config_path is not None:
        command.extend(("--config", str(config_path)))
    command.extend(("--lang", current_locale()))

    if operation == "replay":
        command.extend(
            (
                "replay",
                _existing_file(values.get("wav", ""), t("tools.field.wav")),
                "--seconds",
                _positive_number(
                    values.get("replay-seconds", ""), t("tools.field.seconds")
                ),
            )
        )
    elif operation == "devices":
        command.append("devices")
    elif operation == "serve":
        service = values.get("service", "")
        if service not in {"asr", "verify", "tts"}:
            raise ToolInputError(t("tools.error.service"))
        host = values.get("host", "").strip()
        if not host:
            raise ToolInputError(t("tools.error.required", field=t("tools.field.host")))
        command.extend(("serve", service, "--host", host))
        port = values.get("port", "").strip()
        if port:
            command.extend(
                ("--port", _positive_integer(port, t("tools.field.port"), maximum=65535))
            )
    elif operation == "doctor":
        command.append("doctor")
    elif operation == "netcheck":
        command.extend(
            (
                "netcheck",
                "--samples",
                _positive_integer(values.get("samples", ""), t("tools.field.samples")),
                "--seconds",
                _positive_number(
                    values.get("netcheck-seconds", ""), t("tools.field.seconds")
                ),
            )
        )
    elif operation == "glossary_import":
        command.extend(
            (
                "glossary",
                "import",
                _existing_file(
                    values.get("glossary-path", ""), t("tools.field.glossary_path")
                ),
            )
        )
    elif operation == "glossary_list":
        command.extend(("glossary", "list"))
    elif operation == "completion_show":
        command.append("--show-completion")
    else:
        command.append("--install-completion")
    return command


class ToolsApp(App[None]):
    """Expose every non-interactive CLI operation and its options in Textual."""

    CSS = """
    Screen { layout: vertical; }
    #tools-workspace { height: 1fr; }
    #tools-options {
        width: 42;
        min-width: 36;
        border-right: solid $primary;
        padding: 1 2;
    }
    #tools-output { width: 1fr; padding: 1 2; }
    #operation-label, .field-label { color: $text-muted; margin-top: 1; }
    #operation-description { margin: 1 0; min-height: 3; }
    .tool-field { height: auto; }
    #tool-actions { height: auto; margin-top: 1; }
    #tool-actions Button { margin-right: 1; }
    #output-title { text-style: bold; margin-bottom: 1; }
    #tool-log { height: 1fr; border: round $secondary; }
    #tool-status { height: 1; margin-top: 1; color: $text-muted; }
    """

    BINDINGS = [
        ("r", "execute", ""),
        ("x", "stop", ""),
        ("c", "clear", ""),
        ("q", "back", ""),
    ]

    def __init__(self, config_path: Path | None = None):
        super().__init__()
        self.config_path = config_path
        self.process: Process | None = None
        self._bindings = BindingsMap(
            [
                Binding("r", "execute", t("tools.key.run")),
                Binding("x", "stop", t("tools.key.stop")),
                Binding("c", "clear", t("tools.key.clear")),
                Binding("q", "back", t("tools.key.back")),
            ]
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="tools-workspace"):
            with VerticalScroll(id="tools-options"):
                yield Label(t("tools.operation"), id="operation-label")
                yield Select(
                    [(t(f"tools.operation.{operation}"), operation) for operation in OPERATIONS],
                    value=OPERATIONS[0],
                    allow_blank=False,
                    id="tool-operation",
                )
                yield Static(t("tools.description.replay"), id="operation-description")
                yield from self._input_field("wav", "tools.field.wav", "tools.placeholder.wav")
                yield from self._input_field(
                    "replay-seconds", "tools.field.seconds", value="30.0", input_type="number"
                )
                with Container(id="field-service", classes="tool-field"):
                    yield Label(t("tools.field.service"), classes="field-label")
                    yield Select(
                        [("ASR", "asr"), (t("tools.service.verify"), "verify"), ("TTS", "tts")],
                        value="asr",
                        allow_blank=False,
                        id="service",
                    )
                yield from self._input_field("host", "tools.field.host", value="0.0.0.0")
                yield from self._input_field(
                    "port", "tools.field.port", "tools.placeholder.port", input_type="integer"
                )
                yield from self._input_field(
                    "samples", "tools.field.samples", value="10", input_type="integer"
                )
                yield from self._input_field(
                    "netcheck-seconds",
                    "tools.field.seconds",
                    value="6.0",
                    input_type="number",
                )
                yield from self._input_field(
                    "glossary-path", "tools.field.glossary_path", "tools.placeholder.glossary"
                )
                with Horizontal(id="tool-actions"):
                    yield Button(t("tools.run"), id="tool-run", variant="primary")
                    yield Button(t("tools.stop"), id="tool-stop", disabled=True)
                    yield Button(t("tools.back"), id="tool-back")
            with Container(id="tools-output"):
                yield Static(t("tools.output"), id="output-title")
                yield RichLog(id="tool-log", wrap=True, markup=False)
                yield Static(t("tools.status.ready"), id="tool-status")
        yield Footer()

    def _input_field(
        self,
        field_id: str,
        label_key: str,
        placeholder_key: str | None = None,
        *,
        value: str = "",
        input_type: str = "text",
    ) -> ComposeResult:
        with Container(id=f"field-{field_id}", classes="tool-field"):
            yield Label(t(label_key), classes="field-label")
            yield Input(
                value=value,
                placeholder=t(placeholder_key) if placeholder_key else "",
                type=input_type,
                id=field_id,
            )

    def on_mount(self) -> None:
        self.title = t("tools.title")
        self.sub_title = t(
            "tools.subtitle", config=str(self.config_path or "config/default.yaml")
        )
        self._show_fields(OPERATIONS[0])

    async def on_unmount(self) -> None:
        await self._terminate_process()

    @on(Select.Changed, "#tool-operation")
    def operation_changed(self, event: Select.Changed) -> None:
        if event.value == Select.BLANK:
            return
        self._show_fields(str(event.value))

    @on(Button.Pressed)
    async def button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "tool-run":
            self.action_execute()
        elif event.button.id == "tool-stop":
            await self.action_stop()
        else:
            await self.action_back()

    def _show_fields(self, operation: str) -> None:
        visible = set(OPERATION_FIELDS[operation])
        for field_id in {field for fields in OPERATION_FIELDS.values() for field in fields}:
            self.query_one(f"#field-{field_id}").display = field_id in visible
        self.query_one("#operation-description", Static).update(
            t(f"tools.description.{operation}")
        )

    def _values(self) -> dict[str, str]:
        values = {
            field_id: self.query_one(f"#{field_id}", Input).value
            for field_id in (
                "wav",
                "replay-seconds",
                "host",
                "port",
                "samples",
                "netcheck-seconds",
                "glossary-path",
            )
        }
        service = self.query_one("#service", Select).value
        values["service"] = "" if service == Select.BLANK else str(service)
        return values

    def action_execute(self) -> None:
        if self.process is None:
            self.run_worker(self._execute_tool(), group="tool-process", exclusive=True)

    async def _execute_tool(self) -> None:
        operation_value = self.query_one("#tool-operation", Select).value
        if operation_value == Select.BLANK:
            self._write(t("tools.error.operation"), "red")
            return
        try:
            command = build_tool_command(str(operation_value), self._values(), self.config_path)
        except ToolInputError as error:
            self._write(str(error), "red")
            return

        self._write(f"$ {shlex.join(command)}", "cyan")
        environment = os.environ.copy()
        environment["KOTONOHA_LANG"] = current_locale()
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdin=DEVNULL,
                stdout=PIPE,
                stderr=STDOUT,
                env=environment,
            )
        except OSError as error:
            self._write(t("tools.error.start", error=error), "red")
            return

        self._set_running(True)
        return_code = -1
        try:
            if self.process.stdout is not None:
                while line := await self.process.stdout.readline():
                    self._write(line.decode(errors="replace").rstrip())
            return_code = await self.process.wait()
        except asyncio.CancelledError:
            await self._terminate_process()
            raise
        finally:
            self.process = None
            if self.is_running:
                self._set_running(False)
                self.query_one("#tool-status", Static).update(
                    t("tools.status.finished", code=return_code)
                )

    def _set_running(self, running: bool) -> None:
        self.query_one("#tool-run", Button).disabled = running
        self.query_one("#tool-stop", Button).disabled = not running
        self.query_one("#tool-operation", Select).disabled = running
        self.query_one("#tool-status", Static).update(
            t("tools.status.running") if running else t("tools.status.ready")
        )

    def _write(self, message: str, style: str | None = None) -> None:
        output = Text(message, style=style) if style else message
        self.query_one("#tool-log", RichLog).write(output)

    async def action_stop(self) -> None:
        if self.process is not None:
            self._write(t("tools.status.stopping"), "yellow")
            await self._terminate_process()

    async def _terminate_process(self) -> None:
        process = self.process
        if process is None or process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                return
            await process.wait()

    def action_clear(self) -> None:
        self.query_one("#tool-log", RichLog).clear()

    async def action_back(self) -> None:
        await self._terminate_process()
        self.exit(None)
