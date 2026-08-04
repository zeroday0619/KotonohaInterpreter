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

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static

from ..core.events import UiEvent
from ..i18n import _
from ..logging_setup import drain_terminal_interface_logs
from .log_panel import format_json_log
from .rendering import FrameAccumulator

STATE_STYLE = {
    "IDLE": "dim white",
    "LISTENING": "bold cyan",
    "PROCESSING": "bold yellow",
    "SPEAKING": "bold green",
}
METER_WIDTH = 8
METER_PARTIALS = " ▏▎▍▌▋▊▉"
LOG_RECORDS_PER_FRAME = 32


def level_meter_units(level: float) -> int:
    """Map the useful microphone RMS range onto sub-character meter units."""
    return round(min(1.0, max(0.0, level) * 12.0) * METER_WIDTH * 8)


def level_meter_text(units: int) -> str:
    """Render one eighth-cell precision without changing the status-bar width."""
    bounded_units = min(METER_WIDTH * 8, max(0, units))
    full_cells, partial_units = divmod(bounded_units, 8)
    meter = "█" * full_cells
    if full_cells < METER_WIDTH and partial_units:
        meter += METER_PARTIALS[partial_units]
    return meter.ljust(METER_WIDTH, "·")


class StatusBar(Static):
    state = reactive("IDLE")
    mode = reactive("push_to_talk")
    routing = reactive("pair")
    lang = reactive("—")
    lang_source = reactive("")
    mic = reactive(True)
    perf = reactive("onboard")
    offbox_audio = reactive(False)

    def __init__(self, **widget_options):
        super().__init__(**widget_options)
        self.level = 0.0
        self._meter_units = 0

    def set_level(self, level: float) -> None:
        self.level = level
        meter_units = level_meter_units(level)
        if meter_units != self._meter_units:
            self._meter_units = meter_units
            self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append(" ● ", style=STATE_STYLE.get(self.state, "white"))
        t.append(f"{self.state:<11}", style=STATE_STYLE.get(self.state, "white"))
        t.append("│ mic ", style="dim")
        label = _("OPEN") if self.mic else _("SHUT")
        t.append(f"{label} ", style="green" if self.mic else "red bold")
        t.append(level_meter_text(self._meter_units), style="cyan")
        t.append(" │ ", style="dim")
        t.append(f"{self.mode}", style="magenta")
        t.append(" / ", style="dim")
        t.append(f"{self.routing}", style="magenta")
        t.append(" │ lang ", style="dim")
        t.append(self.lang, style="bold")
        if self.lang_source and self.lang_source != "lid":
            t.append(f" ({self.lang_source})", style="yellow bold")
        t.append(" │ ", style="dim")
        t.append(self.perf, style="bold blue" if self.perf != "onboard" else "dim")
        if self.offbox_audio:
            # The operator should never have to guess whether audio is leaving.
            t.append(" ⇗" + _(" audio off-box"), style="yellow bold")
        return t


class Pane(Static):
    """Plain text panel for the active interpretation turn."""

    def __init__(self, title: str, style: str = "white", **kw):
        super().__init__(**kw)
        self._title = title
        self._style = style
        self._lines: list[str] = []

    def push(self, line: str) -> None:
        self._lines.append(line)
        self._lines = self._lines[-12:]
        self.refresh()

    def replace_last(self, line: str) -> None:
        if self._lines:
            self._lines[-1] = line
        else:
            self._lines.append(line)
        self.refresh()

    def clear(self) -> None:
        if self._lines:
            self._lines.clear()
            self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append(f"{self._title}\n", style="bold underline")
        for ln in self._lines:
            t.append(ln + "\n", style=self._style)
        return t


class HistoryPane(Static):
    """Completed turns, seeded from the database and appended as turns finish.

    The live panes are cleared at the start of every utterance, so without this
    the screen forgets the conversation the moment the next one begins. It is
    seeded across sessions: after a restart the preceding exchanges are still
    the context the operator is working in.
    """

    def __init__(self, title: str, limit: int, **kw):
        super().__init__(**kw)
        self._title = title
        self._limit = max(1, limit)
        self._entries: list[tuple[float, str, str, str, str]] = []

    def load(self, entries) -> None:
        self._entries = [
            (e.ts, e.src_lang or "?", e.tgt_lang or "?", e.source_text or "", e.translation or "")
            for e in entries
        ][-self._limit :]
        self.refresh()

    def append(
        self, ts: float, src_lang: str | None, tgt_lang: str | None, source: str, translation: str
    ) -> None:
        self._entries.append((ts, src_lang or "?", tgt_lang or "?", source, translation))
        self._entries = self._entries[-self._limit :]
        self.refresh()

    def clear_entries(self) -> None:
        self._entries.clear()
        self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append(f"{self._title}\n", style="bold underline")
        if not self._entries:
            t.append(_("No past turns") + "\n", style="dim")
            return t
        # Newest last, so the eye lands on the most recent exchange at the bottom.
        for ts, src, tgt, source, translation in self._entries:
            t.append(datetime.fromtimestamp(ts).strftime("%H:%M:%S "), style="dim")
            t.append(f"{src}→{tgt}\n", style="magenta")
            if source:
                t.append(f"  {source}\n", style="white")
            if translation:
                t.append(f"  {translation}\n", style="cyan")
        return t


class LatencyPanel(Static):
    """§6 budget comparison and the five §11 marks."""

    def __init__(self, budget, **kw):
        super().__init__(**kw)
        self.budget = budget
        self.marks: dict = {}
        self.stages: dict = {}
        self.over: dict = {}
        self.extra: dict = {}

    def update_turn(self, rec: dict) -> None:
        self.marks = rec.get("marks_ms") or {}
        self.stages = rec.get("stages_ms") or {}
        self.over = rec.get("over_budget_ms") or {}
        self.extra = {
            "logprob": rec.get("asr_avg_logprob"),
            "verify": rec.get("cross_verify_fired"),
            "tok/s": rec.get("tok_per_s"),
            "audio_s": rec.get("audio_seconds"),
            "out_tok": rec.get("output_tokens"),
            "outcome": rec.get("outcome"),
        }
        self.refresh()

    def render(self) -> Text:
        b = self.budget
        rows = [
            (_("ASR (+verify)"), self.stages.get("asr"), b.asr + b.verify),
            (_("LLM first clause"), self.stages.get("llm_first_clause"), b.llm_first_clause),
            (_("TTS first packet"), self.stages.get("tts_first_packet"), b.tts_first_packet),
            (_("EOU to audio"), self.stages.get("total_to_first_audio"), b.total - b.silence),
        ]
        t = Text()
        t.append(_("Latency (ms)        measured / budget") + "\n", style="bold underline")
        for name, v, lim in rows:
            t.append(f"{name:<16}")
            if v is None:
                t.append("     —\n", style="dim")
                continue
            style = "red bold" if v > lim else "green"
            t.append(f"{v:>8.0f}", style=style)
            t.append(f" / {lim:<6}\n", style="dim")
        if self.over:
            over = ", ".join(f"{k} +{v:.0f}ms" for k, v in self.over.items())
            t.append(_("Over: ") + over + "\n", style="red bold")
        if self.extra:
            t.append("\n")
            for k, v in self.extra.items():
                if v is None:
                    continue
                t.append(f"{k}={v}  ", style="dim")
        return t


class ServicePanel(Static):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.services: dict[str, dict] = {}
        self.errors: list[str] = []

    def set_service(self, name: str, ok: bool, detail: dict, side: str = "local",
                    degraded: bool = False) -> None:
        self.services[name] = {"ok": ok, "detail": detail, "side": side, "degraded": degraded}
        self.refresh()

    def push_error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")
        self.errors = self.errors[-4:]
        self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append(_("Services") + "\n", style="bold underline")
        for name in ("asr", "asr-verify", "llm", "tts"):
            s = self.services.get(name)
            if s is None:
                t.append(f"  {name:<11} ? \n", style="dim")
                continue
            t.append(f"  {name:<11}")
            t.append("UP  " if s["ok"] else "DOWN", style="green" if s["ok"] else "red bold")
            side = s.get("side", "local")
            if s.get("degraded"):
                side_style = "yellow bold"  # fell back off the A6000
            else:
                side_style = "blue" if side == "remote" else "dim"
            t.append(f" {side:<6}", style=side_style)
            d = s["detail"] or {}
            tag = d.get("backend") or d.get("error")
            if tag:
                t.append(f" {str(tag)[:26]}", style="dim")
            t.append("\n")
        if self.errors:
            t.append("\n" + _("Recent errors") + "\n", style="bold underline red")
            for e in self.errors:
                t.append(f"  {e[:60]}\n", style="red")
        return t


class KotonohaApp(App):
    CSS = """
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
    BINDINGS = [
        ("space", "talk", ""),
        ("a", "toggle_mode", ""),
        ("r", "cycle_routing", ""),
        ("c", "clear", ""),
        ("h", "toggle_history", ""),
        ("t", "text_mode", ""),
        ("escape", "exit_text_mode", ""),
        ("q", "quit", ""),
    ]

    def __init__(self, orch):
        super().__init__()
        self.orch = orch
        self._talking = False
        # Restored when text mode is left, so `t` is a round trip rather than a
        # one-way switch out of whatever the operator had configured.
        self._voice_mode = orch.s.session.mode if orch.s.session.mode != "text" else "push_to_talk"
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        self.status = StatusBar()
        yield self.status
        with Horizontal(id="panes"):
            self.src = Pane(_("Source (ASR)"), "white", id="src")
            self.tgt = Pane(_("Translation"), "bold cyan", id="tgt")
            self.hist = HistoryPane(
                _("History"), max(1, self.orch.s.ui.history_turns), id="hist"
            )
            yield self.src
            yield self.tgt
            yield self.hist
        with Horizontal(id="bottom"):
            self.lat = LatencyPanel(self.orch.s.budget_ms, id="lat")
            self.svc = ServicePanel(id="svc")
            yield self.lat
            yield self.svc
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

    async def on_mount(self) -> None:
        self.title = _("Kotonoha Interpreter")
        self.sub_title = _("session {session}", session=self.orch.session_id)
        self.status.mode = self.orch.s.session.mode
        self.status.routing = self.orch.s.session.routing
        self._show_text_input(self.orch.s.session.mode == "text")
        self.status.perf = self.orch.s.perf_mode
        self.status.offbox_audio = self.orch.s.audio_leaves_device
        if not self.orch.s.logging.console:
            self.log_output.write(
                Text(_("TUI logging is disabled by logging.console=false"), style="dim")
            )
        history_turns = self.orch.s.ui.history_turns
        self.hist.display = history_turns > 0
        if history_turns > 0:
            self.hist.load(self.orch.store.recent_history(history_turns))
        await self.orch.start()
        self.run_worker(self._drain(), exclusive=False)
        self.set_interval(
            1.0 / self.orch.s.ui.refresh_hz,
            self._render_frame,
            name="display-frame",
        )

    async def on_unmount(self) -> None:
        await self.orch.stop()

    # -- event consumption -------------------------------------------------
    async def _drain(self) -> None:
        while True:
            try:
                first_event: UiEvent = await self.orch.bus.get()
            except asyncio.CancelledError:
                return
            events = [first_event, *self.orch.bus.drain_nowait()]
            with self.batch_update():
                for event in events:
                    self._apply(event)
            await asyncio.sleep(0)

    def _render_frame(self) -> None:
        update = self._frame_accumulator.advance()
        with self.batch_update():
            self.status.set_level(update.level)
            if update.translation_changed:
                self.tgt.replace_last(update.translation or "")
            if self.orch.s.logging.console:
                for raw_message in drain_terminal_interface_logs(LOG_RECORDS_PER_FRAME):
                    self.log_output.write(format_json_log(raw_message))

    def _apply(self, ev: UiEvent) -> None:
        p = ev.payload
        if ev.kind == "state":
            self.status.state = p["state"]
            self.status.mic = p["state"] != "SPEAKING"
            if p["state"] == "LISTENING":
                self._clear_transcripts()
            elif p["state"] == "IDLE":
                self._talking = False
        elif ev.kind == "level":
            self._frame_accumulator.push_level(p.get("rms", 0.0))
        elif ev.kind == "lang":
            self.status.lang = p.get("lang") or "—"
            self.status.lang_source = p.get("source") or ""
            if p.get("note"):
                self.svc.push_error("lid", p["note"])
        elif ev.kind == "eou":
            self.src.push(
                _("[{seconds}s, preroll {preroll}ms, {reason}]",
                    seconds=p["seconds"],
                    preroll=p["preroll_ms"],
                    reason=p["ended_by"],
                )
            )
        elif ev.kind == "text_submitted":
            self._clear_transcripts()
        elif ev.kind == "asr":
            if p.get("empty"):
                self.src.replace_last(_("(silence, returning without playback)"))
            else:
                self.src.replace_last(p.get("text", ""))
        elif ev.kind == "verify":
            if p.get("state") == "done":
                mark = "≠" if p.get("divergent") else "≈"
                self.src.push(f"  {mark} whisper: {p.get('text', '')[:70]}")
            elif p.get("state") == "running":
                self.src.push("  … " + _("verifying ({reason})", reason=p.get("reason", "")))
        elif ev.kind == "translation_delta":
            self._frame_accumulator.push_translation(p.get("text", ""))
        elif ev.kind == "clause":
            pass
        elif ev.kind == "translation":
            self._frame_accumulator.discard_translation()
            if p.get("timeout"):
                self.tgt.replace_last(_("(LLM timeout, transcript only, TTS skipped)"))
            else:
                self.tgt.replace_last(p.get("text") or "")
                self.tgt.push("")
        elif ev.kind == "history":
            self.hist.append(
                p["ts"],
                p.get("src_lang"),
                p.get("tgt_lang"),
                p.get("source_text") or "",
                p.get("translation") or "",
            )
        elif ev.kind == "turn":
            self.lat.update_turn(p)
        elif ev.kind == "service":
            self.svc.set_service(
                p["name"],
                p["ok"],
                p.get("detail", {}),
                side=p.get("side", "local"),
                degraded=bool(p.get("degraded")),
            )
        elif ev.kind == "placement":
            # A role moved between the A6000 and the on-board service.
            self.svc.push_error(
                "placement",
                _(
                    "{role} to {side} ({reason})",
                    role=p["role"],
                    side=p["side"],
                    reason=p["reason"],
                ),
            )
            self.status.offbox_audio = self.orch.s.audio_leaves_device
        elif ev.kind == "privacy":
            self.status.offbox_audio = bool(p.get("audio_leaves_device"))
        elif ev.kind == "error":
            self.svc.push_error(p.get("where", "?"), p.get("message", ""))

    # -- keys ----------------------------------------------------------------
    def action_talk(self) -> None:
        if self.orch.s.session.mode != "push_to_talk":
            return
        if self._talking:
            self._talking = False
            self.orch.ptt_up()
        else:
            self._talking = True
            self.orch.ptt_down()

    def action_toggle_mode(self) -> None:
        """Cycle push_to_talk, auto and text."""
        order = ["push_to_talk", "auto", "text"]
        current = self.orch.s.session.mode
        self._set_mode(order[(order.index(current) + 1) % len(order)])

    def action_text_mode(self) -> None:
        """Enter typed input, or leave it when the field does not hold focus."""
        leaving = self.orch.s.session.mode == "text"
        self._set_mode(self._voice_mode if leaving else "text")

    def action_exit_text_mode(self) -> None:
        """Leave typed input. Bound to escape so it works from the focused field."""
        if self.orch.s.session.mode == "text":
            self._set_mode(self._voice_mode)

    def _set_mode(self, mode: str) -> None:
        if mode != "text":
            self._voice_mode = mode
        self.orch.set_text_mode(mode == "text", previous=self._voice_mode)
        self.status.mode = self.orch.s.session.mode
        self._show_text_input(mode == "text")
        if mode == "text":
            self.text_input.focus()
        else:
            self.text_input.value = ""

    def _show_text_input(self, visible: bool) -> None:
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
    def text_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        self.text_input.value = ""
        # submit_text runs the whole turn, so it cannot block the input handler.
        self.run_worker(self.orch.submit_text(text), exclusive=False)

    def action_cycle_routing(self) -> None:
        s = self.orch.s.session
        order = ["pair", "fixed", "broadcast"]
        s.routing = order[(order.index(s.routing) + 1) % len(order)]
        self.status.routing = s.routing

    def action_clear(self) -> None:
        self._clear_transcripts()

    def action_toggle_history(self) -> None:
        """Reclaim the column when the live panes need the width."""
        self.hist.display = not self.hist.display

    def _clear_transcripts(self) -> None:
        self._frame_accumulator.discard_translation()
        self.src.clear()
        self.tgt.clear()
