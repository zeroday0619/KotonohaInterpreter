"""Control center for the integrated terminal interface."""

from __future__ import annotations

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Container
from textual.widgets import Button, Footer, Header, Static

from ..config import Settings
from ..i18n import t


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
        ("t", "tools", ""),
        ("l", "license", ""),
        ("q", "quit", ""),
    ]

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self._bindings = BindingsMap(
            [
                Binding("i", "interpreter", t("tui.menu.key.interpreter")),
                Binding("s", "configuration", t("tui.menu.key.configuration")),
                Binding("t", "tools", t("tui.menu.key.tools")),
                Binding("l", "license", t("tui.menu.key.license")),
                Binding("q", "quit", t("tui.menu.key.quit")),
            ]
        )

    def compose(self) -> ComposeResult:
        placement = "  ".join(
            f"{role}={side}" for role, side in self.settings.resolved_placement().items()
        )
        yield Header(show_clock=True)
        with Container(id="control-center"):
            yield Static(t("tui.menu.title"), id="menu-title")
            yield Static(t("tui.menu.subtitle"), id="menu-subtitle")
            yield Static(
                t("tui.menu.performance", mode=self.settings.perf_mode),
                classes="runtime-detail",
            )
            yield Static(
                t("tui.menu.placement", placement=placement),
                classes="runtime-detail",
            )
            yield Button(
                t("tui.menu.interpreter"),
                id="interpreter",
                variant="primary",
                classes="menu-button",
            )
            yield Static(t("tui.menu.interpreter.description"), classes="menu-description")
            yield Button(
                t("tui.menu.configuration"),
                id="configuration",
                classes="menu-button",
            )
            yield Static(t("tui.menu.configuration.description"), classes="menu-description")
            yield Button(t("tui.menu.tools"), id="tools", classes="menu-button")
            yield Static(t("tui.menu.tools.description"), classes="menu-description")
            yield Button(t("tui.menu.license"), id="license", classes="menu-button")
            yield Static(t("tui.menu.license.description"), classes="menu-description")
            yield Button(t("tui.menu.quit"), id="quit", classes="menu-button")
        yield Footer()

    def on_mount(self) -> None:
        self.title = t("tui.title")
        self.sub_title = t("tui.menu.title")

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "interpreter":
            self.action_interpreter()
        elif event.button.id == "configuration":
            self.action_configuration()
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

    def action_tools(self) -> None:
        self.exit("tools")

    def action_license(self) -> None:
        self.exit("license")

    def action_quit(self) -> None:
        self.exit(None)
