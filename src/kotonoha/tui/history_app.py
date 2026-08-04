"""History browser, reached with `kotonoha history browse` or from the control center.

The interpreter panel shows the last few exchanges. This is the archive: every stored
turn, searchable across both languages, with the diagnostic columns that explain why a
turn came out the way it did.

Queries run against SQLite with LIMIT applied, so the table never holds more rows than
the page size regardless of how many turns the device has accumulated.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, Input, Select, Static

from kotonoha.config import Settings, load_settings
from kotonoha.i18n import _
from kotonoha.store.db import HistoryEntry, Store

PAGE_SIZE = 200
OUTCOMES = ("ok", "empty_asr", "llm_timeout", "tts_failed", "oom", "aborted")
ANY = "__any__"


@dataclass
class HistoryQuery:
    text: str = ""
    src_lang: str | None = None
    outcome: str | None = None
    offset: int = 0


def export_jsonl(entries: list[HistoryEntry], path: Path) -> Path:
    """Write the current result set as JSONL, one turn per line."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")
    return path


def default_export_path(settings: Settings) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return settings.resolve(Path("./data/exports")) / f"history-{stamp}.jsonl"


def excerpt(value: str | None, width: int = 60) -> str:
    text = (value or "").replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1] + "…"


class HistoryApp(App[None]):
    settings: Settings
    store: Store | None
    query: HistoryQuery
    entries: list[HistoryEntry]
    total: int
    table: DataTable
    detail: Static
    status: Static
    title: str
    sub_title: str
    _bindings: BindingsMap

    CSS = """
    Screen { layout: vertical; }
    #filters { height: 3; padding: 0 1; }
    #search { width: 1fr; }
    #filter-lang, #filter-outcome { width: 24; }
    #results { height: 1fr; }
    #detail { height: 14; border: round $secondary; padding: 0 1; overflow-y: auto; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    """

    BINDINGS = [
        ("slash", "focus_search", ""),
        ("r", "reload", ""),
        ("e", "export", ""),
        ("n", "next_page", ""),
        ("p", "previous_page", ""),
        ("escape", "back", ""),
        ("q", "quit", ""),
    ]

    def __init__(self, config_path: Path | None = None, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or load_settings(config_path)
        self.store = None
        self.query = HistoryQuery()
        self.entries: list[HistoryEntry] = []
        self.total = 0
        self._bindings = BindingsMap(
            [
                Binding("slash", "focus_search", _("Search")),
                Binding("r", "reload", _("Reload")),
                Binding("e", "export", _("Export")),
                Binding("n", "next_page", _("Next page")),
                Binding("p", "previous_page", _("Previous page")),
                Binding("escape", "back", _("Back")),
                Binding("q", "quit", _("Quit")),
            ]
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="filters"):
            yield Input(
                placeholder=_("Search source or translation, then press Enter"), id="search"
            )
            yield Select(
                [(_("All languages"), ANY)],
                value=ANY,
                allow_blank=False,
                id="filter-lang",
            )
            yield Select(
                [(_("All outcomes"), ANY)]
                + [(outcome, outcome) for outcome in OUTCOMES],
                value=ANY,
                allow_blank=False,
                id="filter-outcome",
            )
        self.table = DataTable(id="results", cursor_type="row", zebra_stripes=True)
        yield self.table
        self.detail = Static("", id="detail")
        yield self.detail
        self.status = Static("", id="status")
        yield self.status
        yield Footer()

    async def on_mount(self) -> None:
        self.store = await asyncio.to_thread(
            Store,
            self.settings.resolve(self.settings.store.path),
        )
        self.title = _("Interpretation history")
        self.sub_title = str(self.store.path)
        self.table.add_columns(
            _("Time"),
            _("Direction"),
            _("Source"),
            _("Translation"),
            _("Outcome"),
        )
        languages = await asyncio.to_thread(self.store.history_languages)
        self.query_one("#filter-lang", Select).set_options(
            [(_("All languages"), ANY)] + [(code, code) for code in languages]
        )
        await self.reload()

    async def on_unmount(self) -> None:
        if self.store is not None:
            await asyncio.to_thread(self.store.close)

    # -- data ---------------------------------------------------------------
    async def reload(self) -> None:
        filters = {
            "query": self.query.text or None,
            "src_lang": self.query.src_lang,
            "outcome": self.query.outcome,
        }
        self.total, self.entries = await asyncio.to_thread(
            self._load_page,
            filters,
        )

        self.table.clear()
        for entry in self.entries:
            self.table.add_row(
                entry.when.strftime("%m-%d %H:%M:%S"),
                f"{entry.src_lang or '?'}→{entry.tgt_lang or '?'}",
                excerpt(entry.source_text),
                excerpt(entry.translation),
                entry.outcome,
                key=entry.turn_id,
            )
        self._update_status()
        self._show_detail(self.entries[0] if self.entries else None)

    def _load_page(
        self,
        filters: dict[str, str | None],
    ) -> tuple[int, list[HistoryEntry]]:
        if self.store is None:
            return 0, []
        total = self.store.count_turns(**filters)
        entries = self.store.search_turns(
            **filters,
            limit=PAGE_SIZE,
            offset=self.query.offset,
        )
        return total, entries

    def _update_status(self) -> None:
        if not self.total:
            self.status.update(Text(_("No turns match"), style="dim"))
            return
        first = self.query.offset + 1
        last = self.query.offset + len(self.entries)
        self.status.update(
            Text(
                _("{first}-{last} of {total}", first=first, last=last, total=self.total),
                style="dim",
            )
        )

    def _show_detail(self, entry: HistoryEntry | None) -> None:
        if entry is None:
            self.detail.update("")
            return
        body = Text()
        body.append(entry.when.strftime("%Y-%m-%d %H:%M:%S  "), style="bold")
        body.append(f"{entry.src_lang or '?'} → {entry.tgt_lang or '?'}", style="magenta")
        body.append(f"   {entry.outcome}\n", style="green" if entry.outcome == "ok" else "red")
        body.append(_("Source") + "\n", style="bold underline")
        body.append((entry.source_text or "") + "\n\n")
        body.append(_("Translation") + "\n", style="bold underline")
        body.append((entry.translation or "") + "\n\n", style="cyan")

        # The diagnostics that explain a bad turn: which language was chosen and
        # how, how confident the ASR was, and whether the verifier ran.
        facts = [
            ("lang_source", entry.lang_source),
            ("lid_confidence", entry.lid_confidence),
            ("asr_avg_logprob", entry.asr_avg_logprob),
            ("cross_verified", entry.cross_verified),
            ("audio_seconds", entry.audio_seconds),
            ("session", entry.session_id),
            ("turn_id", entry.turn_id),
        ]
        body.append("  ".join(f"{k}={v}" for k, v in facts if v is not None), style="dim")
        self.detail.update(body)

    # -- events -------------------------------------------------------------
    @on(Input.Submitted, "#search")
    async def search_submitted(self, event: Input.Submitted) -> None:
        self.query.text = event.value.strip()
        self.query.offset = 0
        await self.reload()

    @on(Select.Changed, "#filter-lang")
    async def language_changed(self, event: Select.Changed) -> None:
        self.query.src_lang = None if event.value == ANY else str(event.value)
        self.query.offset = 0
        await self.reload()

    @on(Select.Changed, "#filter-outcome")
    async def outcome_changed(self, event: Select.Changed) -> None:
        self.query.outcome = None if event.value == ANY else str(event.value)
        self.query.offset = 0
        await self.reload()

    @on(DataTable.RowHighlighted)
    def row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if 0 <= event.cursor_row < len(self.entries):
            self._show_detail(self.entries[event.cursor_row])

    # -- actions ------------------------------------------------------------
    def action_focus_search(self) -> None:
        self.query_one("#search", Input).focus()

    async def action_reload(self) -> None:
        await self.reload()

    async def action_next_page(self) -> None:
        if self.query.offset + PAGE_SIZE < self.total:
            self.query.offset += PAGE_SIZE
            await self.reload()

    async def action_previous_page(self) -> None:
        if self.query.offset:
            self.query.offset = max(0, self.query.offset - PAGE_SIZE)
            await self.reload()

    async def action_export(self) -> None:
        """Export the rows currently on screen, filters included."""
        if not self.entries:
            self.status.update(Text(_("No turns match"), style="dim"))
            return
        path = await asyncio.to_thread(
            export_jsonl,
            self.entries,
            default_export_path(self.settings),
        )
        self.status.update(
            Text(
                _("Exported {count} turns to {path}", count=len(self.entries), path=path),
                style="green",
            )
        )

    def action_back(self) -> None:
        self.exit(None)
