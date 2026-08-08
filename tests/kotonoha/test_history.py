"""Interpretation history storage, filtering, export, and retention."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from kotonoha._history_support import OUTCOMES, export_jsonl
from kotonoha.store._db import GlossaryEntry, Store


@pytest.fixture
def store(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> Any:
    history_store = Store(tmp_path / "history.db")
    history_store.start_session("session-a", "pair", {})
    history_store.start_session("session-b", "pair", {})
    history_store.add_turn(
        "t1", "session-a", "ko", "en", "회의는 세 시입니다", "The meeting is at three"
    )
    history_store.add_turn(
        "t2", "session-a", "en", "ko", "Send the software list", "목록을 보내주세요"
    )
    history_store.add_turn(
        "t3", "session-b", "ja", "en", "資料を共有します", "I will share the materials"
    )
    history_store.add_turn("t4", "session-b", "ko", "en", None, None, outcome="empty_asr")
    history_store.add_turn(
        "t5", "session-b", "ko", "en,ja", "방송 모드입니다", "This is broadcast mode"
    )
    yield history_store
    history_store.close()


def test_search_filter_and_pagination(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
) -> None:
    assert [entry.turn_id for entry in store.search_turns(limit=3)] == ["t5", "t4", "t3"]
    assert [entry.turn_id for entry in store.search_turns(query="meeting")] == ["t1"]
    assert store.count_turns(src_lang="ko", outcome="ok") == 2
    assert [entry.turn_id for entry in store.search_turns(limit=2, offset=2)] == ["t3", "t2"]


def test_recent_history_skips_failed_turns_and_crosses_sessions(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
) -> None:
    entries = store.recent_history(10)
    assert [entry.turn_id for entry in entries] == ["t1", "t2", "t3", "t5"]
    assert {entry.session_id for entry in entries} == {"session-a", "session-b"}


def test_history_outcomes_and_broadcast_target(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
) -> None:
    assert "asr_failed" in OUTCOMES
    assert [entry.turn_id for entry in store.search_turns(tgt_lang="ja")] == ["t5"]
    assert store.history_languages() == ["en", "ja", "ko"]


def test_export_writes_secure_json_lines(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
    tmp_path: Path,
) -> None:
    target = tmp_path / "out" / "history.jsonl"
    export_jsonl(store.search_turns(limit=10), target)
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    assert json.loads(lines[0])["turn_id"] == "t5"
    assert target.stat().st_mode & 0o777 == 0o600


def test_store_serializes_concurrent_worker_access(
    _positional_only: object | None = None,
    /,
    *,
    store: Store,
) -> None:
    def add_turn(
        sequence: int,
        /,
    ) -> None:
        store.add_turn(
            f"concurrent-{sequence}", "session-a", "ko", "en", str(sequence), str(sequence)
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(add_turn, range(32)))
    assert store.count_turns(session_id="session-a") == 34


def test_clear_history_preserves_glossary(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Path,
) -> None:
    history_store = Store(tmp_path / "history.sqlite3")
    try:
        history_store.add_turn("t1", "session-a", "ko", "en", "안녕", "hello")
        history_store.add_turn("t2", "session-b", "en", "ko", "hello", "안녕")
        history_store.upsert_glossary([GlossaryEntry("en", "Kotonoha", "ko", "코토노하")])
        assert history_store.clear_history("session-a") == 1
        assert history_store.count_turns(session_id="session-b") == 1
        assert history_store.clear_history() == 1
        assert history_store.all_glossary(10)
    finally:
        history_store.close()
