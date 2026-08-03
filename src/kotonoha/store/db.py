"""SQLite store — glossary, six-turn history, Traditional Chinese rules.

Read only when a translation request is assembled. sqlite3 is synchronous, but
every call here is a few-millisecond indexed lookup, so it does not stall the
asyncio loop. If measurement ever says otherwise, move these to to_thread.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
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


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA.read_text(encoding="utf-8"))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

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
        with self.conn:
            self.conn.executemany(
                sql,
                [
                    (e.src_lang, e.src_term, e.tgt_lang, e.tgt_term, e.kind, e.note, e.priority)
                    for e in entries
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
        rows = self.conn.execute(
            """
            SELECT src_lang, src_term, tgt_lang, tgt_term, kind, note, priority
            FROM glossary
            WHERE enabled = 1 AND tgt_lang = ? AND (src_lang = ? OR src_lang = '*')
            ORDER BY priority ASC, length(src_term) DESC
            """,
            (tgt_lang, src_lang),
        ).fetchall()

        hits: list[GlossaryEntry] = []
        for r in rows:
            if r["src_term"] and r["src_term"] in text:
                hits.append(GlossaryEntry(**dict(r)))
                if len(hits) >= limit:
                    break
        return hits

    def all_glossary(self) -> list[GlossaryEntry]:
        rows = self.conn.execute(
            "SELECT src_lang, src_term, tgt_lang, tgt_term, kind, note, priority "
            "FROM glossary WHERE enabled = 1 ORDER BY priority, src_term"
        ).fetchall()
        return [GlossaryEntry(**dict(r)) for r in rows]

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
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO turns
                    (turn_id, ts, session_id, src_lang, tgt_lang, source_text, translation,
                     lang_source, lid_confidence, asr_avg_logprob, cross_verified,
                     audio_seconds, outcome)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    turn_id,
                    time.time(),
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

    def recent_turns(self, session_id: str, n: int = 6) -> list[TurnRecord]:
        rows = self.conn.execute(
            """
            SELECT turn_id, ts, src_lang, tgt_lang, source_text, translation
            FROM turns
            WHERE session_id = ? AND outcome = 'ok' AND source_text IS NOT NULL
            ORDER BY ts DESC LIMIT ?
            """,
            (session_id, n),
        ).fetchall()
        return [TurnRecord(**dict(r)) for r in reversed(rows)]

    def last_language(self, session_id: str) -> str | None:
        """§5 short-utterance LID fallback — inherit the previous verdict."""
        row = self.conn.execute(
            "SELECT src_lang FROM turns WHERE session_id = ? AND src_lang IS NOT NULL "
            "ORDER BY ts DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return row["src_lang"] if row else None

    # -- Traditional Chinese rules ----------------------------------------
    def zh_rules(self) -> list[tuple[str, str, bool]]:
        rows = self.conn.execute(
            "SELECT pattern, replacement, is_regex FROM zh_rules WHERE enabled = 1 "
            "ORDER BY length(pattern) DESC"
        ).fetchall()
        return [(r["pattern"], r["replacement"], bool(r["is_regex"])) for r in rows]

    def upsert_zh_rules(self, rules: list[tuple[str, str, bool, str | None]]) -> int:
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO zh_rules (pattern, replacement, is_regex, note)
                VALUES (?,?,?,?)
                ON CONFLICT(pattern) DO UPDATE SET
                    replacement = excluded.replacement,
                    is_regex    = excluded.is_regex,
                    note        = excluded.note,
                    enabled     = 1
                """,
                [(p, r, 1 if rx else 0, n) for p, r, rx, n in rules],
            )
        return len(rules)

    # -- sessions ----------------------------------------------------------
    def start_session(self, session_id: str, routing: str, config: dict) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, started_at, routing, config) "
                "VALUES (?,?,?,?)",
                (session_id, time.time(), routing, json.dumps(config, ensure_ascii=False)),
            )
