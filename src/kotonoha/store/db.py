"""SQLite store — glossary, six-turn history, Traditional Chinese rules.

The store exposes synchronous operations. Async callers execute them through
`asyncio.to_thread` so SQLite and filesystem access cannot block the event loop.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SCHEMA = Path(__file__).with_name("schema.sql")


@dataclass(frozen=True)
class GlossaryEntry:
    src_lang: str
    src_term: str
    tgt_lang: str
    tgt_term: str
    kind: str = "term"
    note: str | None = None
    priority: int = 100


@dataclass(frozen=True)
class TurnRecord:
    turn_id: str
    ts: float
    src_lang: str | None
    tgt_lang: str | None
    source_text: str | None
    translation: str | None


@dataclass(frozen=True)
class HistoryEntry:
    """A stored turn with the diagnostic columns the browser displays.

    TurnRecord stays narrow because it is injected into the translation prompt,
    where every extra field costs context. This carries everything persisted.
    """

    turn_id: str
    ts: float
    session_id: str | None
    src_lang: str | None
    tgt_lang: str | None
    source_text: str | None
    translation: str | None
    lang_source: str | None
    lid_confidence: float | None
    asr_avg_logprob: float | None
    cross_verified: bool
    audio_seconds: float | None
    outcome: str

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.ts)

    def as_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "ts": self.ts,
            "time": self.when.isoformat(timespec="seconds"),
            "session_id": self.session_id,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "source_text": self.source_text,
            "translation": self.translation,
            "lang_source": self.lang_source,
            "lid_confidence": self.lid_confidence,
            "asr_avg_logprob": self.asr_avg_logprob,
            "cross_verified": self.cross_verified,
            "audio_seconds": self.audio_seconds,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    started_at: float
    routing: str | None
    turns: int
    first_ts: float | None
    last_ts: float | None


HISTORY_COLUMNS = (
    "turn_id, ts, session_id, src_lang, tgt_lang, source_text, translation, "
    "lang_source, lid_confidence, asr_avg_logprob, cross_verified, audio_seconds, outcome"
)


def _entry(row: sqlite3.Row) -> HistoryEntry:
    data = dict(row)
    data["cross_verified"] = bool(data["cross_verified"])
    return HistoryEntry(**data)


def _like(term: str) -> str:
    """Escape the LIKE wildcards so a search for "50%" does not match everything."""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class Store:
    path: Path
    connection: sqlite3.Connection

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA.read_text(encoding="utf-8"))
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    # -- glossary --------------------------------------------------------
    def upsert_glossary(self, entries: list[GlossaryEntry]) -> int:
        sql = """
        INSERT INTO glossary (src_lang, src_term, tgt_lang, tgt_term, kind, note, priority)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(src_lang, src_term, tgt_lang) DO UPDATE SET
            tgt_term = excluded.tgt_term,
            kind     = excluded.kind,
            note     = excluded.note,
            priority = excluded.priority,
            enabled  = 1
        """
        with self.connection:
            self.connection.executemany(
                sql,
                [
                    (
                        entry.src_lang,
                        entry.src_term,
                        entry.tgt_lang,
                        entry.tgt_term,
                        entry.kind,
                        entry.note,
                        entry.priority,
                    )
                    for entry in entries
                ],
            )
        return len(entries)

    def glossary_for(
        self, src_lang: str, tgt_lang: str, text: str, limit: int = 64
    ) -> list[GlossaryEntry]:
        """Return only the entries that actually appear in the transcript.

        Pasting the whole glossary into the prompt eats the 2048-token context
        fast, and unrelated terms contaminate the translation. Only matches go in.
        """
        rows = self.connection.execute(
            """
            SELECT src_lang, src_term, tgt_lang, tgt_term, kind, note, priority
            FROM glossary
            WHERE enabled = 1 AND tgt_lang = ? AND (src_lang = ? OR src_lang = '*')
            ORDER BY priority ASC, length(src_term) DESC
            """,
            (tgt_lang, src_lang),
        ).fetchall()

        hits: list[GlossaryEntry] = []
        for row in rows:
            if row["src_term"] and row["src_term"] in text:
                hits.append(GlossaryEntry(**dict(row)))
                if len(hits) >= limit:
                    break
        return hits

    def all_glossary(self) -> list[GlossaryEntry]:
        rows = self.connection.execute(
            "SELECT src_lang, src_term, tgt_lang, tgt_term, kind, note, priority "
            "FROM glossary WHERE enabled = 1 ORDER BY priority, src_term"
        ).fetchall()
        return [GlossaryEntry(**dict(row)) for row in rows]

    # -- history ---------------------------------------------------------
    def add_turn(
        self,
        turn_id: str,
        session_id: str,
        src_lang: str | None,
        tgt_lang: str | None,
        source_text: str | None,
        translation: str | None,
        lang_source: str = "lid",
        lid_confidence: float | None = None,
        asr_avg_logprob: float | None = None,
        cross_verified: bool = False,
        audio_seconds: float | None = None,
        outcome: str = "ok",
    ) -> float:
        """Persist one turn and return the timestamp written.

        The caller emits the same value to the interface, so the panel and the
        database agree instead of drifting by a round trip.
        """
        timestamp = time.time()
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO turns
                    (turn_id, ts, session_id, src_lang, tgt_lang, source_text, translation,
                     lang_source, lid_confidence, asr_avg_logprob, cross_verified,
                     audio_seconds, outcome)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    turn_id,
                    timestamp,
                    session_id,
                    src_lang,
                    tgt_lang,
                    source_text,
                    translation,
                    lang_source,
                    lid_confidence,
                    asr_avg_logprob,
                    1 if cross_verified else 0,
                    audio_seconds,
                    outcome,
                ),
            )
        return timestamp

    def recent_turns(self, session_id: str, limit: int = 6) -> list[TurnRecord]:
        rows = self.connection.execute(
            """
            SELECT turn_id, ts, src_lang, tgt_lang, source_text, translation
            FROM turns
            WHERE session_id = ? AND outcome = 'ok' AND source_text IS NOT NULL
            ORDER BY ts DESC LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [TurnRecord(**dict(row)) for row in reversed(rows)]

    def last_language(self, session_id: str) -> str | None:
        """§5 short-utterance LID fallback — inherit the previous verdict."""
        row = self.connection.execute(
            "SELECT src_lang FROM turns WHERE session_id = ? AND src_lang IS NOT NULL "
            "ORDER BY ts DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return row["src_lang"] if row else None

    # -- history browsing ---------------------------------------------------
    def _history_where(
        self,
        query: str | None,
        src_lang: str | None,
        tgt_lang: str | None,
        session_id: str | None,
        outcome: str | None,
        since: float | None,
    ) -> tuple[str, list]:
        clauses: list[str] = []
        params: list = []
        if query:
            # Both directions are searched: an operator looking for a turn
            # remembers whichever side they were reading.
            clauses.append(
                "(source_text LIKE ? ESCAPE '\\' OR translation LIKE ? ESCAPE '\\')"
            )
            params += [_like(query), _like(query)]
        if src_lang:
            clauses.append("src_lang = ?")
            params.append(src_lang)
        if tgt_lang:
            # tgt_lang holds a comma-separated list under broadcast routing.
            clauses.append("(tgt_lang = ? OR tgt_lang LIKE ? ESCAPE '\\')")
            params += [tgt_lang, _like(tgt_lang)]
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if outcome:
            clauses.append("outcome = ?")
            params.append(outcome)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params

    def search_turns(
        self,
        query: str | None = None,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        session_id: str | None = None,
        outcome: str | None = None,
        since: float | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[HistoryEntry]:
        """Newest first, for the history browser."""
        where, params = self._history_where(query, src_lang, tgt_lang, session_id, outcome, since)
        rows = self.connection.execute(
            f"SELECT {HISTORY_COLUMNS} FROM turns{where} ORDER BY ts DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return [_entry(row) for row in rows]

    def count_turns(
        self,
        query: str | None = None,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        session_id: str | None = None,
        outcome: str | None = None,
        since: float | None = None,
    ) -> int:
        where, params = self._history_where(query, src_lang, tgt_lang, session_id, outcome, since)
        row = self.connection.execute(
            f"SELECT COUNT(*) AS n FROM turns{where}",
            params,
        ).fetchone()
        return int(row["n"])

    def recent_history(
        self,
        limit: int = 20,
        session_id: str | None = None,
    ) -> list[HistoryEntry]:
        """Oldest first, for the panel inside the interpreter.

        Sessions are not filtered by default: after a restart the operator still
        wants the preceding exchanges on screen.
        """
        rows = self.connection.execute(
            f"SELECT {HISTORY_COLUMNS} FROM turns "
            "WHERE outcome = 'ok' AND translation IS NOT NULL"
            + (" AND session_id = ?" if session_id else "")
            + " ORDER BY ts DESC LIMIT ?",
            ([session_id, limit] if session_id else [limit]),
        ).fetchall()
        return [_entry(row) for row in reversed(rows)]

    def turn(self, turn_id: str) -> HistoryEntry | None:
        row = self.connection.execute(
            f"SELECT {HISTORY_COLUMNS} FROM turns WHERE turn_id = ?", (turn_id,)
        ).fetchone()
        return _entry(row) if row else None

    def history_languages(self) -> list[str]:
        rows = self.connection.execute(
            "SELECT DISTINCT src_lang FROM turns WHERE src_lang IS NOT NULL ORDER BY src_lang"
        ).fetchall()
        return [row["src_lang"] for row in rows]

    def session_summaries(self, limit: int = 50) -> list[SessionSummary]:
        rows = self.connection.execute(
            """
            SELECT s.session_id, s.started_at, s.routing,
                   COUNT(t.id) AS turns, MIN(t.ts) AS first_ts, MAX(t.ts) AS last_ts
            FROM sessions s LEFT JOIN turns t ON t.session_id = s.session_id
            GROUP BY s.session_id
            ORDER BY s.started_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [SessionSummary(**dict(row)) for row in rows]

    # -- Traditional Chinese rules ----------------------------------------
    def zh_rules(self) -> list[tuple[str, str, bool]]:
        rows = self.connection.execute(
            "SELECT pattern, replacement, is_regex FROM zh_rules WHERE enabled = 1 "
            "ORDER BY length(pattern) DESC"
        ).fetchall()
        return [
            (row["pattern"], row["replacement"], bool(row["is_regex"]))
            for row in rows
        ]

    def upsert_zh_rules(self, rules: list[tuple[str, str, bool, str | None]]) -> int:
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO zh_rules (pattern, replacement, is_regex, note)
                VALUES (?,?,?,?)
                ON CONFLICT(pattern) DO UPDATE SET
                    replacement = excluded.replacement,
                    is_regex    = excluded.is_regex,
                    note        = excluded.note,
                    enabled     = 1
                """,
                [
                    (pattern, replacement, 1 if is_regex else 0, note)
                    for pattern, replacement, is_regex, note in rules
                ],
            )
        return len(rules)

    # -- sessions ----------------------------------------------------------
    def start_session(self, session_id: str, routing: str, config: dict) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO sessions (session_id, started_at, routing, config) "
                "VALUES (?,?,?,?)",
                (session_id, time.time(), routing, json.dumps(config, ensure_ascii=False)),
            )
