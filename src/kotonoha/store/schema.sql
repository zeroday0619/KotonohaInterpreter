-- Kotonoha local store. A single SQLite file.
-- No vector DB or embeddings (§12). The glossary is injected as a prompt prefix only.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- Glossary and proper nouns ------------------------------------------------
-- When src_term appears in the transcript, pin the translation to tgt_term.
-- kind: term (general) | name (proper noun) | vocab (regional, e.g. 軟體/影片)
CREATE TABLE IF NOT EXISTS glossary (
    id         INTEGER PRIMARY KEY,
    src_lang   TEXT NOT NULL,
    src_term   TEXT NOT NULL,
    tgt_lang   TEXT NOT NULL,
    tgt_term   TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'term',
    note       TEXT,
    priority   INTEGER NOT NULL DEFAULT 100,
    enabled    INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_glossary
    ON glossary (src_lang, src_term, tgt_lang);
CREATE INDEX IF NOT EXISTS ix_glossary_lookup
    ON glossary (src_lang, tgt_lang, enabled, priority);

-- Conversation history (§3: six turns are injected) ------------------------
CREATE TABLE IF NOT EXISTS turns (
    id               INTEGER PRIMARY KEY,
    turn_id          TEXT NOT NULL UNIQUE,
    ts               REAL NOT NULL DEFAULT (unixepoch('subsec')),
    session_id       TEXT,
    src_lang         TEXT,
    tgt_lang         TEXT,
    source_text      TEXT,
    translation      TEXT,
    lang_source      TEXT,          -- lid | inherited | forced
    lid_confidence   REAL,
    asr_avg_logprob  REAL,
    cross_verified   INTEGER NOT NULL DEFAULT 0,
    audio_seconds    REAL,
    outcome          TEXT NOT NULL DEFAULT 'ok'
);

CREATE INDEX IF NOT EXISTS ix_turns_recent ON turns (session_id, ts DESC);
-- The history browser orders across every session, which the composite
-- index above cannot serve.
CREATE INDEX IF NOT EXISTS ix_turns_ts ON turns (ts DESC);

-- Traditional Chinese post-processing rules --------------------------------
-- Final substitutions for Taiwanese usage that OpenCC s2twp does not catch.
CREATE TABLE IF NOT EXISTS zh_rules (
    id          INTEGER PRIMARY KEY,
    pattern     TEXT NOT NULL UNIQUE,
    replacement TEXT NOT NULL,
    is_regex    INTEGER NOT NULL DEFAULT 0,
    note        TEXT,
    enabled     INTEGER NOT NULL DEFAULT 1
);

-- Snapshot of the session configuration ------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL DEFAULT (unixepoch('subsec')),
    routing    TEXT,
    config     TEXT
);
