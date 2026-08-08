"""Command runner for operations that do not have a dedicated terminal interface."""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
from asyncio.subprocess import DEVNULL, PIPE, STDOUT, Process
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Select, Static

from kotonoha._i18n import N_, _, current_locale
from kotonoha._typing import override

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
    __slots__: ClassVar[tuple[str, ...]] = ()


def _positive_integer(
    value: str,
    /,
    field: str,
    *,
    maximum: int | None = None,
) -> str:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ToolInputError(_("{field} must be a positive integer.", field=field)) from error
    if parsed <= 0:
        raise ToolInputError(_("{field} must be a positive integer.", field=field))
    if maximum is not None and parsed > maximum:
        raise ToolInputError(_("{field} must not exceed {maximum}.", field=field, maximum=maximum))
    return str(parsed)


def _positive_number(
    value: str,
    /,
    field: str,
) -> str:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ToolInputError(_("{field} must be a positive number.", field=field)) from error
    if parsed <= 0:
        raise ToolInputError(_("{field} must be a positive number.", field=field))
    return str(parsed)


def _existing_file(
    value: str,
    /,
    field: str,
) -> str:
    path = Path(value).expanduser()
    if not value.strip() or not path.is_file():
        raise ToolInputError(_("{field} must reference an existing file.", field=field))
    return str(path)


def build_tool_command(
    operation: str,
    /,
    values: Mapping[str, str],
    config_path: Path | None = None,
) -> list[str]:
    """Build a validated CLI invocation without involving a shell."""
    if operation not in OPERATIONS:
        raise ToolInputError(_("Select a valid operation."))

    command = [sys.executable, "-m", "kotonoha._cli"]
    if config_path is not None:
        command.extend(("--config", str(config_path)))
    command.extend(("--lang", current_locale()))

    if operation == "replay":
        command.extend(
            (
                "replay",
                _existing_file(values.get("wav", ""), _("WAV file")),
                "--seconds",
                _positive_number(
                    values.get("replay-seconds", ""), _("Duration in seconds")
                ),
            )
        )
    elif operation == "devices":
        command.append("devices")
    elif operation == "serve":
        service = values.get("service", "")
        if service not in {"asr", "verify"}:
            raise ToolInputError(_("Select a valid service."))
        host = values.get("host", "").strip()
        if not host:
            raise ToolInputError(_("{field} is required.", field=_("Bind address")))
        command.extend(("serve", service, "--host", host))
        port = values.get("port", "").strip()
        if port:
            command.extend(
                ("--port", _positive_integer(port, _("Port"), maximum=65535))
            )
    elif operation == "doctor":
        command.append("doctor")
    elif operation == "netcheck":
        command.extend(
            (
                "netcheck",
                "--samples",
                _positive_integer(values.get("samples", ""), _("Measurements per role")),
                "--seconds",
                _positive_number(
                    values.get("netcheck-seconds", ""), _("Duration in seconds")
                ),
            )
        )
    elif operation == "glossary_import":
        command.extend(
            (
                "glossary",
                "import",
                _existing_file(
                    values.get("glossary-path", ""), _("Glossary YAML file")
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


# Command names and notes. N_ marks them for extraction without translating at
# import time, so the active locale is applied when the widget is built.
OPERATION_LABELS: dict[str, str] = {
    "completion_install": N_("Install shell completion"),
    "completion_show": N_("Show shell completion"),
    "devices": N_("List audio devices"),
    "doctor": N_("Run environment diagnostics"),
    "glossary_import": N_("Import a glossary"),
    "glossary_list": N_("List glossary entries"),
    "netcheck": N_("Measure the external link"),
    "replay": N_("Replay a WAV file"),
    "serve": N_("Start a model service"),
}

OPERATION_DESCRIPTIONS: dict[str, str] = {
    "completion_install": N_("Install completion for the active shell."),
    "completion_show": N_("Print the completion script for the active shell."),
    "devices": N_("Print available audio devices and system defaults."),
    "doctor": N_("Inspect dependencies, placement, models, and service health."),
    "glossary_import": N_("Load glossary terms and Chinese conversion rules."),
    "glossary_list": N_("Print every term stored in the local glossary."),
    "netcheck": N_("Measure remote service latency and audio upload throughput."),
    "replay": N_("Run the full pipeline from a 16-bit PCM WAV file."),
    "serve": N_("Start one Python ASR service."),
}

FIELD_LABELS: dict[str, str] = {
    "glossary_path": N_("Glossary YAML file"),
    "host": N_("Bind address"),
    "port": N_("Port"),
    "samples": N_("Measurements per role"),
    "seconds": N_("Duration in seconds"),
    "service": N_("Service"),
    "wav": N_("WAV file"),
}

FIELD_PLACEHOLDERS: dict[str, str] = {
    "glossary": N_("/path/to/glossary.yaml"),
    "port": N_("Use the service default when empty"),
    "wav": N_("/path/to/probe.wav"),
}


class ToolsApp(App[None]):
    """Expose every non-interactive CLI operation and its options in Textual."""
    __slots__: ClassVar[tuple[str, ...]] = ()

    config_path: Path | None
    process: Process | None
    title: str
    sub_title: str
    _bindings: BindingsMap

    CSS: ClassVar[str] = """
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

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("r", "execute", ""),
        ("x", "stop", ""),
        ("c", "clear", ""),
        ("q", "back", ""),
    ]

    @override
    def __init__(
        self,
        /,
        config_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.config_path = config_path
        self.process: Process | None = None
        self._bindings = BindingsMap(
            [
                Binding("r", "execute", _("Run")),
                Binding("x", "stop", _("Stop")),
                Binding("c", "clear", _("Clear output")),
                Binding("q", "back", _("Back")),
            ]
        )

    @override
    def compose(
        self,
        /,
    ) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="tools-workspace"):
            with VerticalScroll(id="tools-options"):
                yield Label(_("Operation"), id="operation-label")
                yield Select(
                    [(_(OPERATION_LABELS[operation]), operation) for operation in OPERATIONS],
                    value=OPERATIONS[0],
                    allow_blank=False,
                    id="tool-operation",
                )
                yield Static(
                    _("Run the full pipeline from a 16-bit PCM WAV file."),
                    id="operation-description",
                )
                yield from self._input_field("wav", "wav", "wav")
                yield from self._input_field(
                    "replay-seconds", "seconds", value="30.0", input_type="number"
                )
                with Container(id="field-service", classes="tool-field"):
                    yield Label(_("Service"), classes="field-label")
                    yield Select(
                        [("ASR", "asr"), (_("Verification ASR"), "verify")],
                        value="asr",
                        allow_blank=False,
                        id="service",
                    )
                yield from self._input_field("host", "host", value="127.0.0.1")
                yield from self._input_field(
                    "port", "port", "port", input_type="integer"
                )
                yield from self._input_field(
                    "samples", "samples", value="10", input_type="integer"
                )
                yield from self._input_field(
                    "netcheck-seconds",
                    "seconds",
                    value="6.0",
                    input_type="number",
                )
                yield from self._input_field(
                    "glossary-path", "glossary_path", "glossary"
                )
                with Horizontal(id="tool-actions"):
                    yield Button(_("Run"), id="tool-run", variant="primary")
                    yield Button(_("Stop"), id="tool-stop", disabled=True)
                    yield Button(_("Back"), id="tool-back")
            with Container(id="tools-output"):
                yield Static(_("Command output"), id="output-title")
                yield RichLog(id="tool-log", wrap=True, markup=False)
                yield Static(_("Ready"), id="tool-status")
        yield Footer()

    def _input_field(
        self,
        /,
        field_id: str,
        field: str,
        placeholder: str | None = None,
        *,
        value: str = "",
        input_type: str = "text",
    ) -> ComposeResult:
        with Container(id=f"field-{field_id}", classes="tool-field"):
            yield Label(_(FIELD_LABELS[field]), classes="field-label")
            yield Input(
                value=value,
                placeholder=_(FIELD_PLACEHOLDERS[placeholder]) if placeholder else "",
                type=input_type,
                id=field_id,
            )

    def on_mount(
        self,
        /,
    ) -> None:
        self.title = _("Kotonoha operations")
        self.sub_title = _(
            "Configuration: {config}",
            config=str(self.config_path or "config/default.yaml"),
        )
        self._show_fields(OPERATIONS[0])

    async def on_unmount(
        self,
        /,
    ) -> None:
        await self._terminate_process()

    @on(Select.Changed, "#tool-operation")
    def operation_changed(
        self,
        /,
        event: Select.Changed,
    ) -> None:
        if event.value == Select.BLANK:
            return
        self._show_fields(str(event.value))

    @on(Button.Pressed)
    async def button_pressed(
        self,
        /,
        event: Button.Pressed,
    ) -> None:
        if event.button.id == "tool-run":
            self.action_execute()
        elif event.button.id == "tool-stop":
            await self.action_stop()
        else:
            await self.action_back()

    def _show_fields(
        self,
        /,
        operation: str,
    ) -> None:
        visible = set(OPERATION_FIELDS[operation])
        for field_id in {field for fields in OPERATION_FIELDS.values() for field in fields}:
            self.query_one(f"#field-{field_id}").display = field_id in visible
        self.query_one("#operation-description", Static).update(
            _(OPERATION_DESCRIPTIONS[operation])
        )

    def _values(
        self,
        /,
    ) -> dict[str, str]:
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

    def action_execute(
        self,
        /,
    ) -> None:
        if self.process is None:
            self.run_worker(self._execute_tool(), group="tool-process", exclusive=True)

    async def _execute_tool(
        self,
        /,
    ) -> None:
        operation_value = self.query_one("#tool-operation", Select).value
        if operation_value == Select.BLANK:
            self._write(_("Select a valid operation."), "red")
            return
        try:
            command = await asyncio.to_thread(
                build_tool_command,
                str(operation_value),
                self._values(),
                self.config_path,
            )
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
            self._write(_("Failed to start the process: {error}", error=error), "red")
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
                    _("Process finished with exit code {code}", code=return_code)
                )

    def _set_running(
        self,
        /,
        running: bool,
    ) -> None:
        self.query_one("#tool-run", Button).disabled = running
        self.query_one("#tool-stop", Button).disabled = not running
        self.query_one("#tool-operation", Select).disabled = running
        self.query_one("#tool-status", Static).update(
            _("Running") if running else _("Ready")
        )

    def _write(
        self,
        /,
        message: str,
        style: str | None = None,
    ) -> None:
        output = Text(message, style=style) if style else message
        self.query_one("#tool-log", RichLog).write(output)

    async def action_stop(
        self,
        /,
    ) -> None:
        if self.process is not None:
            self._write(_("Stopping the process"), "yellow")
            await self._terminate_process()

    async def _terminate_process(
        self,
        /,
    ) -> None:
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

    def action_clear(
        self,
        /,
    ) -> None:
        self.query_one("#tool-log", RichLog).clear()

    @override
    async def action_back(
        self,
        /,
    ) -> None:
        await self._terminate_process()
        self.exit(None)
