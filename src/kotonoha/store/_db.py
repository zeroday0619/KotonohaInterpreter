"""SQLite store — glossary, six-turn history, Traditional Chinese rules.

The store exposes synchronous operations. Async callers execute them through
`asyncio.to_thread` so SQLite and filesystem access cannot block the event loop.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Final

from kotonoha._secure_files import reject_symbolic_link
from kotonoha._typing import override

SCHEMA = Path(__file__).with_name("schema.sql")
RETENTION_CHECK_INTERVAL: Final[int] = 100


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    src_lang: str
    src_term: str
    tgt_lang: str
    tgt_term: str
    kind: str = "term"
    note: str | None = None
    priority: int = 100


@dataclass(frozen=True, slots=True)
class TurnRecord:
    turn_id: str
    ts: float
    src_lang: str | None
    tgt_lang: str | None
    source_text: str | None
    translation: str | None


@dataclass(frozen=True, slots=True)
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
    def when(
        self,
        /,
    ) -> datetime:
        return datetime.fromtimestamp(self.ts)

    def as_dict(
        self,
        /,
    ) -> dict:
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


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session_id: str
    started_at: float
    routing: str | None
    turns: int
    first_ts: float | None
    last_ts: float | None


HISTORY_SEARCH_STATEMENT = (
    "SELECT turn_id, ts, session_id, src_lang, tgt_lang, source_text, translation, "
    "lang_source, lid_confidence, asr_avg_logprob, cross_verified, audio_seconds, outcome "
    "FROM turns"
    " WHERE (? IS NULL OR source_text LIKE ? ESCAPE '\\' "
    "OR translation LIKE ? ESCAPE '\\')"
    " AND (? IS NULL OR src_lang = ?)"
    " AND (? IS NULL OR tgt_lang = ? OR tgt_lang LIKE ? ESCAPE '\\')"
    " AND (? IS NULL OR session_id = ?)"
    " AND (? IS NULL OR outcome = ?)"
    " AND (? IS NULL OR ts >= ?)"
    " ORDER BY ts DESC LIMIT ? OFFSET ?"
)
HISTORY_COUNT_STATEMENT = (
    "SELECT COUNT(*) AS n FROM turns"
    " WHERE (? IS NULL OR source_text LIKE ? ESCAPE '\\' "
    "OR translation LIKE ? ESCAPE '\\')"
    " AND (? IS NULL OR src_lang = ?)"
    " AND (? IS NULL OR tgt_lang = ? OR tgt_lang LIKE ? ESCAPE '\\')"
    " AND (? IS NULL OR session_id = ?)"
    " AND (? IS NULL OR outcome = ?)"
    " AND (? IS NULL OR ts >= ?)"
)


def _entry(
    row: sqlite3.Row,
    /,
) -> HistoryEntry:
    data = dict(row)
    data["cross_verified"] = bool(data["cross_verified"])
    return HistoryEntry(**data)


def _like(
    term: str,
    /,
) -> str:
    """Escape the LIKE wildcards so a search for "50%" does not match everything."""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class Store:
    __slots__: ClassVar[tuple[str, ...]] = (
        "_closed",
        "_lock",
        "_maximum_sessions",
        "_maximum_turns",
        "_turns_since_retention",
        "connection",
        "path",
    )
    path: Path
    connection: sqlite3.Connection
    _lock: threading.RLock
    _closed: bool
    _maximum_turns: int
    _maximum_sessions: int
    _turns_since_retention: int

    @override
    def __init__(
        self,
        /,
        path: Path,
        *,
        maximum_turns: int = 10_000,
        maximum_sessions: int = 1_000,
    ) -> None:
        if maximum_turns <= 0 or maximum_sessions <= 0:
            raise ValueError("store retention limits must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        reject_symbolic_link(path)
        self.path = path
        self._lock = threading.RLock()
        self._closed = False
        self._maximum_turns = maximum_turns
        self._maximum_sessions = maximum_sessions
        self._turns_since_retention = 0
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        try:
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA busy_timeout = 5000")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = NORMAL")
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.executescript(SCHEMA.read_text(encoding="utf-8"))
            self._prune_turns()
            self._prune_sessions()
            self.connection.commit()
            self._secure_database_files()
        except Exception:
            self.connection.close()
            self._closed = True
            raise

    def _secure_database_files(
        self,
        /,
    ) -> None:
        """Protect the database and SQLite sidecars that can contain transcripts."""
        for suffix in ("", "-wal", "-shm"):
            candidate = self.path.with_name(f"{self.path.name}{suffix}")
            try:
                candidate.chmod(0o600)
            except FileNotFoundError:
                continue

    def clear_history(
        self,
        /,
        session_id: str | None = None,
    ) -> int:
        """Delete recorded turns and return how many rows were removed.

        The glossary and the Traditional Chinese rules are operator-authored
        configuration rather than history, so they survive a reset. Passing a
        session identifier limits the reset to that conversation.
        """
        with self._lock, self.connection:
            if session_id is None:
                removed = self.connection.execute("DELETE FROM turns").rowcount
                self.connection.execute("DELETE FROM sessions")
            else:
                removed = self.connection.execute(
                    "DELETE FROM turns WHERE session_id = ?",
                    (session_id,),
                ).rowcount
                self.connection.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
        return max(0, removed)

    def _prune_turns(
        self,
        /,
    ) -> None:
        self.connection.execute(
            "DELETE FROM turns WHERE id IN ("
            "SELECT id FROM turns ORDER BY ts DESC, id DESC LIMIT -1 OFFSET ?)",
            (self._maximum_turns,),
        )
        self._turns_since_retention = 0

    def _prune_sessions(
        self,
        /,
    ) -> None:
        self.connection.execute(
            "DELETE FROM sessions WHERE session_id IN ("
            "SELECT session_id FROM sessions "
            "ORDER BY started_at DESC LIMIT -1 OFFSET ?)",
            (self._maximum_sessions,),
        )

    def close(
        self,
        /,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self.connection.close()
            self._closed = True

    def _fetchall(
        self,
        /,
        sql: str,
        parameters: tuple[Any, ...] | list[Any] = (),
    ) -> list[sqlite3.Row]:
        with self._lock:
            return self.connection.execute(sql, parameters).fetchall()

    def _fetchone(
        self,
        /,
        sql: str,
        parameters: tuple[Any, ...] | list[Any] = (),
    ) -> sqlite3.Row | None:
        with self._lock:
            return self.connection.execute(sql, parameters).fetchone()

    # -- glossary --------------------------------------------------------
    def upsert_glossary(
        self,
        /,
        entries: list[GlossaryEntry],
    ) -> int:
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
        with self._lock, self.connection:
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
        self,
        /,
        src_lang: str,
        tgt_lang: str,
        text: str,
        limit: int = 64,
    ) -> list[GlossaryEntry]:
        """Return only the entries that actually appear in the transcript.

        Pasting the whole glossary into the prompt eats the 2048-token context
        fast, and unrelated terms contaminate the translation. Only matches go in.
        """
        rows = self._fetchall(
            """
            SELECT src_lang, src_term, tgt_lang, tgt_term, kind, note, priority
            FROM glossary
            WHERE enabled = 1 AND tgt_lang = ? AND (src_lang = ? OR src_lang = '*')
              AND src_term <> '' AND instr(?, src_term) > 0
            ORDER BY priority ASC, length(src_term) DESC
            LIMIT ?
            """,
            (tgt_lang, src_lang, text, max(0, limit)),
        )
        return [GlossaryEntry(**dict(row)) for row in rows]

    def all_glossary(
        self,
        /,
        limit: int | None = None,
    ) -> list[GlossaryEntry]:
        sql = (
            "SELECT src_lang, src_term, tgt_lang, tgt_term, kind, note, priority "
            "FROM glossary WHERE enabled = 1 ORDER BY priority, src_term"
        )
        parameters: tuple[int, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            parameters = (max(0, limit),)
        rows = self._fetchall(sql, parameters)
        return [GlossaryEntry(**dict(row)) for row in rows]

    # -- history ---------------------------------------------------------
    def add_turn(
        self,
        /,
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
        with self._lock, self.connection:
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
            self._turns_since_retention += 1
            if self._turns_since_retention >= RETENTION_CHECK_INTERVAL:
                self._prune_turns()
        return timestamp

    def recent_turns(
        self,
        /,
        session_id: str,
        limit: int = 6,
    ) -> list[TurnRecord]:
        rows = self._fetchall(
            """
            SELECT turn_id, ts, src_lang, tgt_lang, source_text, translation
            FROM turns
            WHERE session_id = ? AND outcome = 'ok' AND source_text IS NOT NULL
            ORDER BY ts DESC LIMIT ?
            """,
            (session_id, max(0, limit)),
        )
        return [TurnRecord(**dict(row)) for row in reversed(rows)]

    def last_language(
        self,
        /,
        session_id: str,
    ) -> str | None:
        """§5 short-utterance LID fallback — inherit the previous verdict."""
        row = self._fetchone(
            "SELECT src_lang FROM turns WHERE session_id = ? AND src_lang IS NOT NULL "
            "ORDER BY ts DESC LIMIT 1",
            (session_id,),
        )
        return row["src_lang"] if row else None

    # -- history browsing ---------------------------------------------------
    def _history_parameters(
        self,
        /,
        query: str | None,
        src_lang: str | None,
        tgt_lang: str | None,
        session_id: str | None,
        outcome: str | None,
        since: float | None,
    ) -> list[Any]:
        query_value = query or None
        query_pattern = _like(query_value) if query_value is not None else None
        source_value = src_lang or None
        target_value = tgt_lang or None
        target_pattern = _like(target_value) if target_value is not None else None
        session_value = session_id or None
        outcome_value = outcome or None
        # Both text directions are searched. tgt_lang can contain a comma-separated
        # broadcast route, so it retains exact and escaped substring comparisons.
        return [
            query_value,
            query_pattern,
            query_pattern,
            source_value,
            source_value,
            target_value,
            target_value,
            target_pattern,
            session_value,
            session_value,
            outcome_value,
            outcome_value,
            since,
            since,
        ]

    def search_turns(
        self,
        /,
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
        parameters = self._history_parameters(
            query,
            src_lang,
            tgt_lang,
            session_id,
            outcome,
            since,
        )
        rows = self._fetchall(
            HISTORY_SEARCH_STATEMENT,
            [*parameters, max(0, limit), max(0, offset)],
        )
        return [_entry(row) for row in rows]

    def count_turns(
        self,
        /,
        query: str | None = None,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        session_id: str | None = None,
        outcome: str | None = None,
        since: float | None = None,
    ) -> int:
        parameters = self._history_parameters(
            query,
            src_lang,
            tgt_lang,
            session_id,
            outcome,
            since,
        )
        row = self._fetchone(
            HISTORY_COUNT_STATEMENT,
            parameters,
        )
        return int(row["n"])

    def recent_history(
        self,
        /,
        limit: int = 20,
        session_id: str | None = None,
    ) -> list[HistoryEntry]:
        """Oldest first, for the panel inside the interpreter.

        Sessions are not filtered by default: after a restart the operator still
        wants the preceding exchanges on screen.
        """
        if session_id:
            rows = self._fetchall(
                "SELECT turn_id, ts, session_id, src_lang, tgt_lang, source_text, "
                "translation, lang_source, lid_confidence, asr_avg_logprob, "
                "cross_verified, audio_seconds, outcome FROM turns "
                "WHERE outcome = 'ok' AND translation IS NOT NULL AND session_id = ? "
                "ORDER BY ts DESC LIMIT ?",
                (session_id, max(0, limit)),
            )
        else:
            rows = self._fetchall(
                "SELECT turn_id, ts, session_id, src_lang, tgt_lang, source_text, "
                "translation, lang_source, lid_confidence, asr_avg_logprob, "
                "cross_verified, audio_seconds, outcome FROM turns "
                "WHERE outcome = 'ok' AND translation IS NOT NULL "
                "ORDER BY ts DESC LIMIT ?",
                (max(0, limit),),
            )
        return [_entry(row) for row in reversed(rows)]

    def turn(
        self,
        /,
        turn_id: str,
    ) -> HistoryEntry | None:
        row = self._fetchone(
            "SELECT turn_id, ts, session_id, src_lang, tgt_lang, source_text, "
            "translation, lang_source, lid_confidence, asr_avg_logprob, cross_verified, "
            "audio_seconds, outcome FROM turns WHERE turn_id = ?",
            (turn_id,),
        )
        return _entry(row) if row else None

    def history_languages(
        self,
        /,
    ) -> list[str]:
        rows = self._fetchall(
            "SELECT DISTINCT src_lang FROM turns WHERE src_lang IS NOT NULL ORDER BY src_lang"
        )
        return [row["src_lang"] for row in rows]

    def session_summaries(
        self,
        /,
        limit: int = 50,
    ) -> list[SessionSummary]:
        rows = self._fetchall(
            """
            SELECT s.session_id, s.started_at, s.routing,
                   COUNT(t.id) AS turns, MIN(t.ts) AS first_ts, MAX(t.ts) AS last_ts
            FROM sessions s LEFT JOIN turns t ON t.session_id = s.session_id
            GROUP BY s.session_id
            ORDER BY s.started_at DESC LIMIT ?
            """,
            (max(0, limit),),
        )
        return [SessionSummary(**dict(row)) for row in rows]

    # -- Traditional Chinese rules ----------------------------------------
    def zh_rules(
        self,
        /,
    ) -> list[tuple[str, str, bool]]:
        rows = self._fetchall(
            "SELECT pattern, replacement, is_regex FROM zh_rules WHERE enabled = 1 "
            "ORDER BY length(pattern) DESC"
        )
        return [
            (row["pattern"], row["replacement"], bool(row["is_regex"]))
            for row in rows
        ]

    def upsert_zh_rules(
        self,
        /,
        rules: list[tuple[str, str, bool, str | None]],
    ) -> int:
        if any(is_regex for _pattern, _replacement, is_regex, _note in rules):
            raise ValueError(
                "regular-expression Traditional Chinese rules are disabled because "
                "the runtime cannot enforce an execution deadline"
            )
        with self._lock, self.connection:
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
    def start_session(
        self,
        /,
        session_id: str,
        routing: str,
        config: dict,
    ) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO sessions (session_id, started_at, routing, config) "
                "VALUES (?,?,?,?)",
                (session_id, time.time(), routing, json.dumps(config, ensure_ascii=False)),
            )
            self._prune_sessions()
