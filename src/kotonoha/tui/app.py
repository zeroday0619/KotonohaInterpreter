"""TUI (Textual).

화면에 반드시 있어야 하는 것:
  · 현재 상태 — SPEAKING 중에 마이크가 닫혀 있다는 사실이 눈에 보여야 한다.
  · 판정 언어와 그 출처(lid / inherited) — §10 이 요구한다. 승계된 언어로
    통역 중인데 화면에 표시가 없으면 사용자는 기기가 고장난 줄 안다.
  · 다섯 지점 지연과 예산 초과 여부 — 없으면 어디서 새는지 알 수 없다(§11).

터미널은 키를 뗀 이벤트를 주지 않는다. 그래서 push-to-talk 은 스페이스 토글로
구현한다(누르면 시작, 다시 누르면 종료).
"""

from __future__ import annotations

import asyncio

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from ..core.events import UiEvent

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

    def render(self) -> Text:
        t = Text()
        t.append(" ● ", style=STATE_STYLE.get(self.state, "white"))
        t.append(f"{self.state:<11}", style=STATE_STYLE.get(self.state, "white"))
        t.append("│ mic ", style="dim")
        t.append("OPEN " if self.mic else "SHUT ", style="green" if self.mic else "red bold")
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
        return t


class Pane(Static):
    """스크롤 없는 단순 텍스트 패널. 최근 N턴만 보여준다."""

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
    """§6 예산 대조 + §11 다섯 지점."""

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
            ("ASR (+verify)", self.stages.get("asr"), b.asr + b.verify),
            ("LLM 첫 절", self.stages.get("llm_first_clause"), b.llm_first_clause),
            ("TTS 첫 패킷", self.stages.get("tts_first_packet"), b.tts_first_packet),
            ("EOU→첫 음성", self.stages.get("total_to_first_audio"), b.total - b.silence),
        ]
        t = Text()
        t.append("지연 (ms)            실측 / 예산\n", style="bold underline")
        for name, v, lim in rows:
            t.append(f"{name:<16}")
            if v is None:
                t.append("     —\n", style="dim")
                continue
            style = "red bold" if v > lim else "green"
            t.append(f"{v:>8.0f}", style=style)
            t.append(f" / {lim:<6}\n", style="dim")
        if self.over:
            t.append("초과: " + ", ".join(f"{k} +{v:.0f}ms" for k, v in self.over.items()) + "\n",
                     style="red bold")
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

    def set_service(self, name: str, ok: bool, detail: dict) -> None:
        self.services[name] = {"ok": ok, "detail": detail}
        self.refresh()

    def push_error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")
        self.errors = self.errors[-4:]
        self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append("서비스\n", style="bold underline")
        for name in ("asr", "asr-verify", "llm", "tts"):
            s = self.services.get(name)
            if s is None:
                t.append(f"  {name:<11} ? \n", style="dim")
                continue
            t.append(f"  {name:<11}")
            t.append("UP  " if s["ok"] else "DOWN", style="green" if s["ok"] else "red bold")
            d = s["detail"] or {}
            tag = d.get("backend") or d.get("error")
            if tag:
                t.append(f"  {str(tag)[:34]}", style="dim")
            t.append("\n")
        if self.errors:
            t.append("\n최근 오류\n", style="bold underline red")
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

    BINDINGS = [
        ("space", "talk", "말하기(토글)"),
        ("a", "toggle_mode", "PTT/자동"),
        ("r", "cycle_routing", "라우팅"),
        ("c", "clear", "지우기"),
        ("q", "quit", "종료"),
    ]

    def __init__(self, orch):
        super().__init__()
        self.orch = orch
        self._talking = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        self.status = StatusBar()
        yield self.status
        with Horizontal(id="panes"):
            self.src = Pane("원문 (ASR)", "white", id="src")
            self.tgt = Pane("번역", "bold cyan", id="tgt")
            yield self.src
            yield self.tgt
        with Horizontal(id="bottom"):
            self.lat = LatencyPanel(self.orch.s.budget_ms, id="lat")
            self.svc = ServicePanel(id="svc")
            yield self.lat
            yield self.svc
        yield Footer()

    async def on_mount(self) -> None:
        self.title = "Kotonoha Interpreter"
        self.sub_title = f"session {self.orch.session_id}"
        self.status.mode = self.orch.s.session.mode
        self.status.routing = self.orch.s.session.routing
        await self.orch.start()
        self.run_worker(self._drain(), exclusive=False)

    async def on_unmount(self) -> None:
        await self.orch.stop()

    # ── 이벤트 소비 ─────────────────────────────────────────────────────
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
            self.src.push(f"[{p['seconds']}s · preroll {p['preroll_ms']}ms · {p['ended_by']}]")
        elif ev.kind == "asr":
            if p.get("empty"):
                self.src.replace_last("(무음 — 재생 없이 복귀)")
            else:
                self.src.replace_last(p.get("text", ""))
        elif ev.kind == "verify":
            if p.get("state") == "done":
                mark = "≠" if p.get("divergent") else "≈"
                self.src.push(f"  {mark} whisper: {p.get('text', '')[:70]}")
            elif p.get("state") == "running":
                self.src.push(f"  … 교차검증 ({p.get('reason', '')})")
        elif ev.kind == "translation_delta":
            self.tgt.replace_last(p.get("text", ""))
        elif ev.kind == "clause":
            pass
        elif ev.kind == "translation":
            if p.get("timeout"):
                self.tgt.replace_last("(LLM 타임아웃 — 원문만 표시, TTS 생략)")
            else:
                self.tgt.replace_last(p.get("text") or "")
                self.tgt.push("")
        elif ev.kind == "turn":
            self.lat.update_turn(p)
        elif ev.kind == "service":
            self.svc.set_service(p["name"], p["ok"], p.get("detail", {}))
        elif ev.kind == "error":
            self.svc.push_error(p.get("where", "?"), p.get("message", ""))

    # ── 키 ──────────────────────────────────────────────────────────────
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
