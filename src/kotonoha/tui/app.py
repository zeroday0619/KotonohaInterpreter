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

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from ..core.events import UiEvent
from ..i18n import t as _

STATE_STYLE = {
    "IDLE": "dim white",
    "LISTENING": "bold cyan",
    "PROCESSING": "bold yellow",
    "SPEAKING": "bold green",
}


class StatusBar(Static):
    state = reactive("IDLE")
    mode = reactive("push_to_talk")
    routing = reactive("pair")
    lang = reactive("—")
    lang_source = reactive("")
    mic = reactive(True)
    level = reactive(0.0)
    perf = reactive("onboard")
    offbox_audio = reactive(False)

    def render(self) -> Text:
        t = Text()
        t.append(" ● ", style=STATE_STYLE.get(self.state, "white"))
        t.append(f"{self.state:<11}", style=STATE_STYLE.get(self.state, "white"))
        t.append("│ mic ", style="dim")
        label = _("tui.mic.open") if self.mic else _("tui.mic.shut")
        t.append(f"{label} ", style="green" if self.mic else "red bold")
        bars = int(min(1.0, self.level * 12) * 10)
        t.append("▁▂▃▄▅▆▇█"[: max(0, bars // 2)].ljust(5, "·"), style="cyan")
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
            t.append(" ⇗" + _("tui.audio_offbox"), style="yellow bold")
        return t


class Pane(Static):
    """Plain text panel, no scrolling. Shows only the last N turns."""

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

    def render(self) -> Text:
        t = Text()
        t.append(f"{self._title}\n", style="bold underline")
        for ln in self._lines:
            t.append(ln + "\n", style=self._style)
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
            (_("tui.stage.asr"), self.stages.get("asr"), b.asr + b.verify),
            (_("tui.stage.llm"), self.stages.get("llm_first_clause"), b.llm_first_clause),
            (_("tui.stage.tts"), self.stages.get("tts_first_packet"), b.tts_first_packet),
            (_("tui.stage.total"), self.stages.get("total_to_first_audio"), b.total - b.silence),
        ]
        t = Text()
        t.append(_("tui.panel.latency") + "\n", style="bold underline")
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
            t.append(_("tui.over_budget") + over + "\n", style="red bold")
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
        t.append(_("tui.panel.services") + "\n", style="bold underline")
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
            t.append("\n" + _("tui.panel.errors") + "\n", style="bold underline red")
            for e in self.errors:
                t.append(f"  {e[:60]}\n", style="red")
        return t


class KotonohaApp(App):
    CSS = """
    Screen { layout: vertical; }
    StatusBar { height: 1; background: $panel; }
    #panes { height: 1fr; }
    #src, #tgt { width: 1fr; border: round $primary; padding: 0 1; }
    #bottom { height: 12; }
    #lat, #svc { width: 1fr; border: round $secondary; padding: 0 1; }
    """

    # Descriptions are localized in __init__: BINDINGS is a class attribute and is
    # evaluated before the locale can be resolved from --lang.
    BINDINGS = [
        ("space", "talk", ""),
        ("a", "toggle_mode", ""),
        ("r", "cycle_routing", ""),
        ("c", "clear", ""),
        ("q", "quit", ""),
    ]

    def __init__(self, orch):
        super().__init__()
        self.orch = orch
        self._talking = False
        # Replace the map rather than calling bind() on it: Textual builds the map
        # from the class attribute, so mutating it would leak one instance's locale
        # into the next.
        self._bindings = BindingsMap(
            [
                Binding("space", "talk", _("tui.key.talk")),
                Binding("a", "toggle_mode", _("tui.key.mode")),
                Binding("r", "cycle_routing", _("tui.key.routing")),
                Binding("c", "clear", _("tui.key.clear")),
                Binding("q", "quit", _("tui.key.quit")),
            ]
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        self.status = StatusBar()
        yield self.status
        with Horizontal(id="panes"):
            self.src = Pane(_("tui.pane.source"), "white", id="src")
            self.tgt = Pane(_("tui.pane.target"), "bold cyan", id="tgt")
            yield self.src
            yield self.tgt
        with Horizontal(id="bottom"):
            self.lat = LatencyPanel(self.orch.s.budget_ms, id="lat")
            self.svc = ServicePanel(id="svc")
            yield self.lat
            yield self.svc
        yield Footer()

    async def on_mount(self) -> None:
        self.title = _("tui.title")
        self.sub_title = _("tui.subtitle", session=self.orch.session_id)
        self.status.mode = self.orch.s.session.mode
        self.status.routing = self.orch.s.session.routing
        self.status.perf = self.orch.s.perf_mode
        self.status.offbox_audio = self.orch.s.audio_leaves_device
        await self.orch.start()
        self.run_worker(self._drain(), exclusive=False)

    async def on_unmount(self) -> None:
        await self.orch.stop()

    # -- event consumption -------------------------------------------------
    async def _drain(self) -> None:
        while True:
            try:
                ev: UiEvent = await self.orch.bus.get()
            except asyncio.CancelledError:
                return
            self._apply(ev)

    def _apply(self, ev: UiEvent) -> None:
        p = ev.payload
        if ev.kind == "state":
            self.status.state = p["state"]
            self.status.mic = p["state"] != "SPEAKING"
            if p["state"] == "IDLE":
                self._talking = False
        elif ev.kind == "level":
            self.status.level = p.get("rms", 0.0)
        elif ev.kind == "lang":
            self.status.lang = p.get("lang") or "—"
            self.status.lang_source = p.get("source") or ""
            if p.get("note"):
                self.svc.push_error("lid", p["note"])
        elif ev.kind == "eou":
            self.src.push(
                _(
                    "tui.eou",
                    seconds=p["seconds"],
                    preroll=p["preroll_ms"],
                    reason=p["ended_by"],
                )
            )
        elif ev.kind == "asr":
            if p.get("empty"):
                self.src.replace_last(_("tui.asr.empty"))
            else:
                self.src.replace_last(p.get("text", ""))
        elif ev.kind == "verify":
            if p.get("state") == "done":
                mark = "≠" if p.get("divergent") else "≈"
                self.src.push(f"  {mark} whisper: {p.get('text', '')[:70]}")
            elif p.get("state") == "running":
                self.src.push("  … " + _("tui.verify.running", reason=p.get("reason", "")))
        elif ev.kind == "translation_delta":
            self.tgt.replace_last(p.get("text", ""))
        elif ev.kind == "clause":
            pass
        elif ev.kind == "translation":
            if p.get("timeout"):
                self.tgt.replace_last(_("tui.llm.timeout"))
            else:
                self.tgt.replace_last(p.get("text") or "")
                self.tgt.push("")
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
                _("tui.placement.moved", role=p["role"], side=p["side"], reason=p["reason"]),
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
        s = self.orch.s.session
        s.mode = "auto" if s.mode == "push_to_talk" else "push_to_talk"
        self.status.mode = s.mode

    def action_cycle_routing(self) -> None:
        s = self.orch.s.session
        order = ["pair", "fixed", "broadcast"]
        s.routing = order[(order.index(s.routing) + 1) % len(order)]
        self.status.routing = s.routing

    def action_clear(self) -> None:
        self.src._lines.clear()
        self.tgt._lines.clear()
        self.src.refresh()
        self.tgt.refresh()
