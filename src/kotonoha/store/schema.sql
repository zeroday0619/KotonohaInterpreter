-- Kotonoha 로컬 저장소. 단일 SQLite 파일.
-- 벡터DB/임베딩은 도입하지 않는다(§12). 용어집은 프롬프트 프리픽스로만 주입한다.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- 용어집 · 고유명사 -------------------------------------------------------
-- src_term 이 전사에 등장하면 tgt_term 으로 고정한다.
-- kind: term(일반 용어) | name(고유명사) | vocab(지역 어휘, 예: 軟體/影片)
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

-- 대화 히스토리 (§3: 6턴 주입) --------------------------------------------
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

-- 번체 변환 후처리 규칙 ---------------------------------------------------
-- OpenCC s2twp 로도 안 잡히는 대만 관용 표현을 여기서 마지막에 치환한다.
CREATE TABLE IF NOT EXISTS zh_rules (
    id          INTEGER PRIMARY KEY,
    pattern     TEXT NOT NULL UNIQUE,
    replacement TEXT NOT NULL,
    is_regex    INTEGER NOT NULL DEFAULT 0,
    note        TEXT,
    enabled     INTEGER NOT NULL DEFAULT 1
);

-- 세션 설정 스냅샷 --------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL DEFAULT (unixepoch('subsec')),
    routing    TEXT,
    config     TEXT
);
