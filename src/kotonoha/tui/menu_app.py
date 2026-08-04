"""Control center for the integrated terminal interface."""

from __future__ import annotations

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Container
from textual.widgets import Button, Footer, Header, Static

from ..config import Settings
from ..i18n import _


class TuiMenuApp(App[str | None]):
    CSS = """
    Screen { align: center middle; }
    #control-center {
        width: 72;
        height: auto;
        border: round $primary;
        padding: 1 3;
    }
    #menu-title { text-style: bold; color: $accent; margin-bottom: 1; }
    #menu-subtitle { color: $text-muted; margin-bottom: 1; }
    .runtime-detail { color: $text-muted; }
    .menu-button { width: 100%; margin-top: 1; }
    .menu-description { color: $text-muted; margin: 0 1; }
    """

    BINDINGS = [
        ("i", "interpreter", ""),
        ("s", "configuration", ""),
        ("h", "history", ""),
        ("t", "tools", ""),
        ("l", "license", ""),
        ("q", "quit", ""),
    ]

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self._bindings = BindingsMap(
            [
                Binding("i", "interpreter", _("Interpreter")),
                Binding("s", "configuration", _("Configuration")),
                Binding("h", "history", _("Interpretation history")),
                Binding("t", "tools", _("Operations")),
                Binding("l", "license", _("License")),
                Binding("q", "quit", _("Exit")),
            ]
        )

    def compose(self) -> ComposeResult:
        placement = "  ".join(
            f"{role}={side}" for role, side in self.settings.resolved_placement().items()
        )
        yield Header(show_clock=True)
        with Container(id="control-center"):
            yield Static(_("Control center"), id="menu-title")
            yield Static(_("Select an operation"), id="menu-subtitle")
            yield Static(
                _("Performance mode: {mode}", mode=self.settings.perf_mode),
                classes="runtime-detail",
            )
            yield Static(
                _("Placement: {placement}", placement=placement),
                classes="runtime-detail",
            )
            yield Button(
                _("Interpreter"),
                id="interpreter",
                variant="primary",
                classes="menu-button",
            )
            yield Static(
                _("Start microphone capture and interpretation"), classes="menu-description"
            )
            yield Button(
                _("Configuration"),
                id="configuration",
                classes="menu-button",
            )
            yield Static(_("Edit local and remote settings"), classes="menu-description")
            yield Button(_("Interpretation history"), id="history", classes="menu-button")
            yield Static(_("Search past turns and export them"), classes="menu-description")
            yield Button(_("Operations"), id="tools", classes="menu-button")
            yield Static(
                _("Run diagnostics, services, replay, and glossary commands"),
                classes="menu-description",
            )
            yield Button(_("License"), id="license", classes="menu-button")
            yield Static(
                _("Review project and installed dependency licenses"),
                classes="menu-description",
            )
            yield Button(_("Exit"), id="quit", classes="menu-button")
        yield Footer()

    def on_mount(self) -> None:
        self.title = _("Kotonoha Interpreter")
        self.sub_title = _("Control center")

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "interpreter":
            self.action_interpreter()
        elif event.button.id == "configuration":
            self.action_configuration()
        elif event.button.id == "history":
            self.action_history()
        elif event.button.id == "tools":
            self.action_tools()
        elif event.button.id == "license":
            self.action_license()
        else:
            self.action_quit()

    def action_interpreter(self) -> None:
        self.exit("interpreter")

    def action_configuration(self) -> None:
        self.exit("configuration")

    def action_history(self) -> None:
        self.exit("history")

    def action_tools(self) -> None:
        self.exit("tools")

    def action_license(self) -> None:
        self.exit("license")

    def action_quit(self) -> None:
        self.exit(None)
