"""TUI (Textual).

What the screen has to show:
  · The current state — that the microphone is shut during SPEAKING must be
    visible.
  · The detected language and where it came from (lid / inherited), as §10
    requires. Interpreting from an inherited language with no indication on
    screen makes users think the device is broken.
  · The five latency marks and whether the budget was blown — without them
    there is no way to see where time is going (§11).

Terminals do not deliver key-release events, so push-to-talk is a space-bar
toggle: press to start, press again to finish.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, ClassVar

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static

from kotonoha._config import LatencyBudgetConfig
from kotonoha._i18n import _
from kotonoha._logging_setup import drain_terminal_interface_logs
from kotonoha._typing import override
from kotonoha.core._events import UiEvent
from kotonoha.core._orchestrator import Orchestrator
from kotonoha.store._db import HistoryEntry
from kotonoha.tui._log_panel import format_json_log
from kotonoha.tui._rendering import FrameAccumulator

STATE_STYLE = {
    "IDLE": "dim white",
    "LISTENING": "bold cyan",
    "PROCESSING": "bold yellow",
    "SPEAKING": "bold green",
}
METER_WIDTH = 8
METER_PARTIALS = " ▏▎▍▌▋▊▉"
LOG_RECORDS_PER_FRAME = 32


def level_meter_units(
    level: float,
    /,
) -> int:
    """Map the useful microphone RMS range onto sub-character meter units."""
    return round(min(1.0, max(0.0, level) * 12.0) * METER_WIDTH * 8)


def level_meter_text(
    units: int,
    /,
) -> str:
    """Render one eighth-cell precision without changing the status-bar width."""
    bounded_units = min(METER_WIDTH * 8, max(0, units))
    full_cells, partial_units = divmod(bounded_units, 8)
    meter = "█" * full_cells
    if full_cells < METER_WIDTH and partial_units:
        meter += METER_PARTIALS[partial_units]
    return meter.ljust(METER_WIDTH, "·")


class StatusBar(Static):
    __slots__: ClassVar[tuple[str, ...]] = ()
    level: float
    _meter_units: int

    state: ClassVar[Any] = reactive("IDLE")
    mode: ClassVar[Any] = reactive("push_to_talk")
    routing: ClassVar[Any] = reactive("pair")
    language: ClassVar[Any] = reactive("—")
    lang_source: ClassVar[Any] = reactive("")
    mic: ClassVar[Any] = reactive(True)
    perf: ClassVar[Any] = reactive("onboard")
    offbox_audio: ClassVar[Any] = reactive(False)

    @override
    def __init__(
        self,
        /,
        **widget_options: Any,
    ) -> None:
        super().__init__(**widget_options)
        self.level = 0.0
        self._meter_units = 0

    def set_level(
        self,
        /,
        level: float,
    ) -> None:
        self.level = level
        meter_units = level_meter_units(level)
        if meter_units != self._meter_units:
            self._meter_units = meter_units
            self.refresh()

    @override
    def render(
        self,
        /,
    ) -> Text:
        text = Text()
        text.append(" ● ", style=STATE_STYLE.get(self.state, "white"))
        text.append(f"{self.state:<11}", style=STATE_STYLE.get(self.state, "white"))
        text.append("│ mic ", style="dim")
        label = _("OPEN") if self.mic else _("SHUT")
        text.append(f"{label} ", style="green" if self.mic else "red bold")
        text.append(level_meter_text(self._meter_units), style="cyan")
        text.append(" │ ", style="dim")
        text.append(f"{self.mode}", style="magenta")
        text.append(" / ", style="dim")
        text.append(f"{self.routing}", style="magenta")
        text.append(" │ lang ", style="dim")
        text.append(self.language, style="bold")
        if self.lang_source and self.lang_source != "lid":
            text.append(f" ({self.lang_source})", style="yellow bold")
        text.append(" │ ", style="dim")
        text.append(
            self.perf,
            style="bold blue" if self.perf != "onboard" else "dim",
        )
        if self.offbox_audio:
            # Audio placement must remain visible while remote mode is active.
            text.append(" ⇗" + _(" audio off-box"), style="yellow bold")
        return text


class Pane(Static):
    """Plain text panel for the active interpretation turn."""
    __slots__: ClassVar[tuple[str, ...]] = ()

    _title: str
    _style: str
    _lines: list[str]

    @override
    def __init__(
        self,
        /,
        title: str,
        style: str = "white",
        **widget_options: Any,
    ) -> None:
        super().__init__(**widget_options)
        self._title = title
        self._style = style
        self._lines: list[str] = []

    def push(
        self,
        /,
        line: str,
    ) -> None:
        self._lines.append(line)
        self._lines = self._lines[-12:]
        self.refresh()

    def replace_last(
        self,
        /,
        line: str,
    ) -> None:
        if self._lines:
            self._lines[-1] = line
        else:
            self._lines.append(line)
        self.refresh()

    def clear(
        self,
        /,
    ) -> None:
        if self._lines:
            self._lines.clear()
            self.refresh()

    @override
    def render(
        self,
        /,
    ) -> Text:
        text = Text()
        text.append(f"{self._title}\n", style="bold underline")
        for line in self._lines:
            text.append(line + "\n", style=self._style)
        return text


class HistoryPane(Static):
    """Completed turns, seeded from the database and appended as turns finish.

    The live panes are cleared at the start of every utterance, so without this
    the screen forgets the conversation the moment the next one begins. It is
    seeded across sessions: after a restart the preceding exchanges are still
    the context the operator is working in.
    """
    __slots__: ClassVar[tuple[str, ...]] = ()

    _title: str
    _limit: int
    _entries: list[tuple[float, str, str, str, str]]

    @override
    def __init__(
        self,
        /,
        title: str,
        limit: int,
        **widget_options: Any,
    ) -> None:
        super().__init__(**widget_options)
        self._title = title
        self._limit = max(1, limit)
        self._entries: list[tuple[float, str, str, str, str]] = []

    def load(
        self,
        /,
        entries: list[HistoryEntry],
    ) -> None:
        self._entries = [
            (
                entry.ts,
                entry.src_lang or "?",
                entry.tgt_lang or "?",
                entry.source_text or "",
                entry.translation or "",
            )
            for entry in entries
        ][-self._limit :]
        self.refresh()

    def append(
        self,
        /,
        timestamp: float,
        source_language: str | None,
        target_language: str | None,
        source: str,
        translation: str,
    ) -> None:
        self._entries.append(
            (
                timestamp,
                source_language or "?",
                target_language or "?",
                source,
                translation,
            )
        )
        self._entries = self._entries[-self._limit :]
        self.refresh()

    def clear_entries(
        self,
        /,
    ) -> None:
        self._entries.clear()
        self.refresh()

    @override
    def render(
        self,
        /,
    ) -> Text:
        text = Text()
        text.append(f"{self._title}\n", style="bold underline")
        if not self._entries:
            text.append(_("No past turns") + "\n", style="dim")
            return text
        # Newest last, so the eye lands on the most recent exchange at the bottom.
        for timestamp, source_language, target_language, source, translation in self._entries:
            text.append(
                datetime.fromtimestamp(timestamp).strftime("%H:%M:%S "),
                style="dim",
            )
            text.append(f"{source_language}→{target_language}\n", style="magenta")
            if source:
                text.append(f"  {source}\n", style="white")
            if translation:
                text.append(f"  {translation}\n", style="cyan")
        return text


class LatencyPanel(Static):
    """§6 budget comparison and the five §11 marks."""
    __slots__: ClassVar[tuple[str, ...]] = ()

    budget: LatencyBudgetConfig
    marks: dict
    stages: dict
    over: dict
    extra: dict

    @override
    def __init__(
        self,
        /,
        budget: LatencyBudgetConfig,
        **widget_options: Any,
    ) -> None:
        super().__init__(**widget_options)
        self.budget = budget
        self.marks: dict = {}
        self.stages: dict = {}
        self.over: dict = {}
        self.extra: dict = {}

    def update_turn(
        self,
        /,
        record: dict,
    ) -> None:
        self.marks = record.get("marks_ms") or {}
        self.stages = record.get("stages_ms") or {}
        self.over = record.get("over_budget_ms") or {}
        self.extra = {
            "logprob": record.get("asr_avg_logprob"),
            "verify": record.get("cross_verify_fired"),
            "tok/s": record.get("tok_per_s"),
            "audio_s": record.get("audio_seconds"),
            "out_tok": record.get("output_tokens"),
            "outcome": record.get("outcome"),
        }
        self.refresh()

    @override
    def render(
        self,
        /,
    ) -> Text:
        budget = self.budget
        rows = [
            (
                _("ASR (+verify)"),
                self.stages.get("asr"),
                budget.asr + budget.verify,
            ),
            (
                _("LLM first clause"),
                self.stages.get("llm_first_clause"),
                budget.llm_first_clause,
            ),
            (
                _("TTS first packet"),
                self.stages.get("tts_first_packet"),
                budget.tts_first_packet,
            ),
            (
                _("EOU to audio"),
                self.stages.get("total_to_first_audio"),
                budget.total - budget.silence,
            ),
        ]
        text = Text()
        text.append(
            _("Latency (ms)        measured / budget") + "\n",
            style="bold underline",
        )
        for name, measured, limit in rows:
            text.append(f"{name:<16}")
            if measured is None:
                text.append("     —\n", style="dim")
                continue
            style = "red bold" if measured > limit else "green"
            text.append(f"{measured:>8.0f}", style=style)
            text.append(f" / {limit:<6}\n", style="dim")
        if self.over:
            exceeded = ", ".join(
                f"{stage} +{duration:.0f}ms"
                for stage, duration in self.over.items()
            )
            text.append(_("Over: ") + exceeded + "\n", style="red bold")
        if self.extra:
            text.append("\n")
            for key, value in self.extra.items():
                if value is None:
                    continue
                text.append(f"{key}={value}  ", style="dim")
        return text


class ServicePanel(Static):
    __slots__: ClassVar[tuple[str, ...]] = ()
    services: dict[str, dict]
    errors: list[str]

    @override
    def __init__(
        self,
        /,
        **widget_options: Any,
    ) -> None:
        super().__init__(**widget_options)
        self.services: dict[str, dict] = {}
        self.errors: list[str] = []

    def set_service(
        self,
        /,
        name: str,
        ok: bool,
        detail: dict,
        side: str = "local",
        degraded: bool = False,
    ) -> None:
        self.services[name] = {"ok": ok, "detail": detail, "side": side, "degraded": degraded}
        self.refresh()

    def push_error(
        self,
        /,
        where: str,
        message: str,
    ) -> None:
        self.errors.append(f"{where}: {message}")
        self.errors = self.errors[-4:]
        self.refresh()

    @override
    def render(
        self,
        /,
    ) -> Text:
        text = Text()
        text.append(_("Services") + "\n", style="bold underline")
        for name in ("asr", "asr-verify", "llm", "tts"):
            service = self.services.get(name)
            if service is None:
                text.append(f"  {name:<11} ? \n", style="dim")
                continue
            text.append(f"  {name:<11}")
            text.append(
                "UP  " if service["ok"] else "DOWN",
                style="green" if service["ok"] else "red bold",
            )
            side = service.get("side", "local")
            if service.get("degraded"):
                side_style = "yellow bold"  # fell back off the A6000
            else:
                side_style = "blue" if side == "remote" else "dim"
            text.append(f" {side:<6}", style=side_style)
            detail = service["detail"] or {}
            tag = detail.get("backend") or detail.get("error")
            if tag:
                text.append(f" {str(tag)[:26]}", style="dim")
            text.append("\n")
        if self.errors:
            text.append(
                "\n" + _("Recent errors") + "\n",
                style="bold underline red",
            )
            for error in self.errors:
                text.append(f"  {error[:60]}\n", style="red")
        return text


class KotonohaApp(App):
    __slots__: ClassVar[tuple[str, ...]] = ()
    orchestrator: Orchestrator
    status: StatusBar
    source_pane: Pane
    translation_pane: Pane
    history_pane: HistoryPane
    latency_panel: LatencyPanel
    service_panel: ServicePanel
    text_input: Input
    log_output: RichLog
    title: str
    sub_title: str
    _talking: bool
    _voice_mode: str
    _frame_accumulator: FrameAccumulator
    _bindings: BindingsMap

    CSS: ClassVar[str] = """
    Screen { layout: vertical; }
    StatusBar { height: 1; background: $panel; }
    #panes { height: 1fr; }
    #src, #tgt { width: 1fr; border: round $primary; padding: 0 1; }
    #hist { width: 1fr; border: round $secondary; padding: 0 1; overflow-y: auto; }
    #bottom { height: 9; }
    #lat, #svc { width: 1fr; border: round $secondary; padding: 0 1; }
    #text-input { height: 3; border: round $accent; }
    #logs { height: 7; border: round $secondary; padding: 0 1; }
    #log-title { height: 1; text-style: bold underline; }
    #log-output { height: 1fr; }
    """

    # Descriptions are localized in __init__: BINDINGS is a class attribute and is
    # evaluated before the locale can be resolved from --lang.
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("space", "talk", ""),
        ("a", "toggle_mode", ""),
        ("r", "cycle_routing", ""),
        ("c", "clear", ""),
        ("h", "toggle_history", ""),
        ("t", "text_mode", ""),
        ("escape", "exit_text_mode", ""),
        ("q", "quit", ""),
    ]

    @override
    def __init__(
        self,
        /,
        orchestrator: Orchestrator,
    ) -> None:
        super().__init__()
        self.orchestrator = orchestrator
        self._talking = False
        # Restored when text mode is left, so `t` is a round trip rather than a
        # one-way switch out of whatever the operator had configured.
        self._voice_mode = (
            orchestrator.settings.session.mode
            if orchestrator.settings.session.mode != "text"
            else "push_to_talk"
        )
        self._frame_accumulator = FrameAccumulator()
        # Replace the map rather than calling bind() on it: Textual builds the map
        # from the class attribute, so mutating it would leak one instance's locale
        # into the next.
        self._bindings = BindingsMap(
            [
                Binding("space", "talk", _("Talk (toggle)")),
                Binding("a", "toggle_mode", _("PTT/auto")),
                Binding("r", "cycle_routing", _("Routing")),
                Binding("c", "clear", _("Clear")),
                Binding("h", "toggle_history", _("History")),
                Binding("t", "text_mode", _("Text input")),
                # priority, because the focused input consumes ordinary keys.
                # Without it there is no way out of text mode: `t` becomes a
                # character in the field rather than a binding.
                Binding(
                    "escape",
                    "exit_text_mode",
                    _("Leave text input"),
                    priority=True,
                ),
                Binding("q", "quit", _("Quit")),
            ]
        )

    @override
    def compose(
        self,
        /,
    ) -> ComposeResult:
        yield Header(show_clock=True)
        self.status = StatusBar()
        yield self.status
        with Horizontal(id="panes"):
            self.source_pane = Pane(_("Source (ASR)"), "white", id="src")
            self.translation_pane = Pane(_("Translation"), "bold cyan", id="tgt")
            self.history_pane = HistoryPane(
                _("History"), max(1, self.orchestrator.settings.ui.history_turns), id="hist"
            )
            yield self.source_pane
            yield self.translation_pane
            yield self.history_pane
        with Horizontal(id="bottom"):
            self.latency_panel = LatencyPanel(self.orchestrator.settings.budget_ms, id="lat")
            self.service_panel = ServicePanel(id="svc")
            yield self.latency_panel
            yield self.service_panel
        self.text_input = Input(
            placeholder=_("Type an utterance and press Enter. Press t to return to voice."),
            id="text-input",
        )
        yield self.text_input
        with Container(id="logs"):
            yield Static(_("Application logs"), id="log-title")
            self.log_output = RichLog(
                max_lines=500,
                wrap=True,
                markup=False,
                id="log-output",
            )
            yield self.log_output
        yield Footer()

    async def on_mount(
        self,
        /,
    ) -> None:
        self.title = _("Kotonoha Interpreter")
        self.sub_title = _("session {session}", session=self.orchestrator.session_id)
        self.status.mode = self.orchestrator.settings.session.mode
        self.status.routing = self.orchestrator.settings.session.routing
        self._show_text_input(self.orchestrator.settings.session.mode == "text")
        self.status.perf = self.orchestrator.settings.perf_mode
        self.status.offbox_audio = self.orchestrator.settings.audio_leaves_device
        if not self.orchestrator.settings.logging.console:
            self.log_output.write(
                Text(_("TUI logging is disabled by logging.console=false"), style="dim")
            )
        history_turns = self.orchestrator.settings.ui.history_turns
        self.history_pane.display = history_turns > 0
        if history_turns > 0:
            entries = await asyncio.to_thread(
                self.orchestrator.store.recent_history,
                history_turns,
            )
            self.history_pane.load(entries)
        await self.orchestrator.start()
        self.run_worker(self._drain(), exclusive=False)
        self.set_interval(
            1.0 / self.orchestrator.settings.ui.refresh_hz,
            self._render_frame,
            name="display-frame",
        )

    async def on_unmount(
        self,
        /,
    ) -> None:
        await self.orchestrator.stop()

    # -- event consumption -------------------------------------------------
    async def _drain(
        self,
        /,
    ) -> None:
        while True:
            try:
                first_event: UiEvent = await self.orchestrator.event_bus.get()
            except asyncio.CancelledError:
                return
            events = [first_event, *self.orchestrator.event_bus.drain_nowait()]
            with self.batch_update():
                for event in events:
                    self._apply(event)
            await asyncio.sleep(0)

    def _render_frame(
        self,
        /,
    ) -> None:
        update = self._frame_accumulator.advance()
        with self.batch_update():
            self.status.set_level(update.level)
            if update.translation_changed:
                self.translation_pane.replace_last(update.translation or "")
            if self.orchestrator.settings.logging.console:
                for raw_message in drain_terminal_interface_logs(LOG_RECORDS_PER_FRAME):
                    self.log_output.write(format_json_log(raw_message))

    def _apply(
        self,
        /,
        event: UiEvent,
    ) -> None:
        payload = event.payload
        if event.kind == "state":
            self.status.state = payload["state"]
            self.status.mic = payload["state"] != "SPEAKING"
            if payload["state"] == "LISTENING":
                self._clear_transcripts()
            elif payload["state"] == "IDLE":
                self._talking = False
        elif event.kind == "level":
            self._frame_accumulator.push_level(payload.get("rms", 0.0))
        elif event.kind == "lang":
            self.status.language = payload.get("lang") or "—"
            self.status.lang_source = payload.get("source") or ""
            if payload.get("note"):
                self.service_panel.push_error("lid", payload["note"])
        elif event.kind == "eou":
            self.source_pane.push(
                _("[{seconds}s, preroll {preroll}ms, {reason}]",
                    seconds=payload["seconds"],
                    preroll=payload["preroll_ms"],
                    reason=payload["ended_by"],
                )
            )
        elif event.kind == "text_submitted":
            self._clear_transcripts()
        elif event.kind == "asr":
            if payload.get("empty"):
                self.source_pane.replace_last(_("(silence, returning without playback)"))
            else:
                self.source_pane.replace_last(payload.get("text", ""))
        elif event.kind == "verify":
            if payload.get("state") == "done":
                mark = "≠" if payload.get("divergent") else "≈"
                self.source_pane.push(f"  {mark} whisper: {payload.get('text', '')[:70]}")
            elif payload.get("state") == "running":
                message = _("verifying ({reason})", reason=payload.get("reason", ""))
                self.source_pane.push("  … " + message)
        elif event.kind == "translation_delta":
            self._frame_accumulator.push_translation(payload.get("text", ""))
        elif event.kind == "clause":
            pass
        elif event.kind == "translation":
            self._frame_accumulator.discard_translation()
            if payload.get("timeout"):
                self.translation_pane.replace_last(_("(LLM timeout, transcript only, TTS skipped)"))
            else:
                self.translation_pane.replace_last(payload.get("text") or "")
                self.translation_pane.push("")
        elif event.kind == "history":
            self.history_pane.append(
                payload["ts"],
                payload.get("src_lang"),
                payload.get("tgt_lang"),
                payload.get("source_text") or "",
                payload.get("translation") or "",
            )
        elif event.kind == "turn":
            self.latency_panel.update_turn(payload)
        elif event.kind == "service":
            self.service_panel.set_service(
                payload["name"],
                payload["ok"],
                payload.get("detail", {}),
                side=payload.get("side", "local"),
                degraded=bool(payload.get("degraded")),
            )
        elif event.kind == "placement":
            # A role moved between the A6000 and the on-board service.
            self.service_panel.push_error(
                "placement",
                _(
                    "{role} to {side} ({reason})",
                    role=payload["role"],
                    side=payload["side"],
                    reason=payload["reason"],
                ),
            )
            self.status.offbox_audio = self.orchestrator.settings.audio_leaves_device
        elif event.kind == "privacy":
            self.status.offbox_audio = bool(payload.get("audio_leaves_device"))
        elif event.kind == "error":
            self.service_panel.push_error(payload.get("where", "?"), payload.get("message", ""))

    # -- keys ----------------------------------------------------------------
    def action_talk(
        self,
        /,
    ) -> None:
        if self.orchestrator.settings.session.mode != "push_to_talk":
            return
        if self._talking:
            self._talking = False
            self.orchestrator.ptt_up()
        else:
            self._talking = True
            self.orchestrator.ptt_down()

    def action_toggle_mode(
        self,
        /,
    ) -> None:
        """Cycle push_to_talk, auto and text."""
        order = ["push_to_talk", "auto", "text"]
        current = self.orchestrator.settings.session.mode
        self._set_mode(order[(order.index(current) + 1) % len(order)])

    def action_text_mode(
        self,
        /,
    ) -> None:
        """Enter typed input, or leave it when the field does not hold focus."""
        leaving = self.orchestrator.settings.session.mode == "text"
        self._set_mode(self._voice_mode if leaving else "text")

    def action_exit_text_mode(
        self,
        /,
    ) -> None:
        """Leave typed input. Bound to escape so it works from the focused field."""
        if self.orchestrator.settings.session.mode == "text":
            self._set_mode(self._voice_mode)

    def _set_mode(
        self,
        /,
        mode: str,
    ) -> None:
        if mode != "text":
            self._voice_mode = mode
        self.orchestrator.set_text_mode(mode == "text", previous=self._voice_mode)
        self.status.mode = self.orchestrator.settings.session.mode
        self._show_text_input(mode == "text")
        if mode == "text":
            self.text_input.focus()
        else:
            self.text_input.value = ""

    def _show_text_input(
        self,
        /,
        visible: bool,
    ) -> None:
        """Show or hide the field, and keep it out of the focus chain when hidden.

        display alone is not enough. Textual focuses the first focusable widget on
        mount, and a hidden Input is still focusable, so it would take focus and
        swallow every single-letter binding — including `t`, the key that reveals it.
        """
        self.text_input.display = visible
        self.text_input.can_focus = visible
        if not visible:
            self.set_focus(None)

    @on(Input.Submitted, "#text-input")
    def text_submitted(
        self,
        /,
        event: Input.Submitted,
    ) -> None:
        text = event.value.strip()
        if not text:
            return
        self.text_input.value = ""
        # submit_text runs the whole turn, so it cannot block the input handler.
        self.run_worker(self.orchestrator.submit_text(text), exclusive=False)

    def action_cycle_routing(
        self,
        /,
    ) -> None:
        session = self.orchestrator.settings.session
        order = ["pair", "fixed", "broadcast"]
        session.routing = order[(order.index(session.routing) + 1) % len(order)]
        self.status.routing = session.routing

    def action_clear(
        self,
        /,
    ) -> None:
        self._clear_transcripts()

    def action_toggle_history(
        self,
        /,
    ) -> None:
        """Reclaim the column when the live panes need the width."""
        self.history_pane.display = not self.history_pane.display

    def _clear_transcripts(
        self,
        /,
    ) -> None:
        self._frame_accumulator.discard_translation()
        self.source_pane.clear()
        self.translation_pane.clear()
