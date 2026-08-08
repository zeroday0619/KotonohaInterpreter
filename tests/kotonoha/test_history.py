"""Interpretation history: store queries, the interpreter panel, and the browser."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from kotonoha._i18n import set_locale, translate_to
from kotonoha.store._db import HistoryEntry, Store
from kotonoha.tui._app import HistoryPane
from kotonoha.tui._history_app import OUTCOMES, HistoryApp, excerpt, export_jsonl


@pytest.fixture(autouse=True)
def _reset_locale() -> Any:
    yield
    set_locale(None)


@pytest.fixture
def store(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> Any:
    st = Store(tmp_path / "history.db")
    st.start_session("session-a", "pair", {})
    st.start_session("session-b", "pair", {})
    st.add_turn("t1", "session-a", "ko", "en", "회의는 세 시입니다", "The meeting is at three")
    st.add_turn(
        "t2", "session-a", "en", "ko", "Send the software list", "소프트웨어 목록을 보내주세요"
    )
    st.add_turn("t3", "session-b", "ja", "en", "資料を共有します", "I will share the materials")
    st.add_turn("t4", "session-b", "ko", "en", None, None, outcome="empty_asr")
    st.add_turn(
        "t5", "session-b", "ko", "en,ja", "방송 모드입니다", "This is broadcast mode"
    )
    yield st
    st.close()


# -- store queries ----------------------------------------------------------
def test_search_returns_newest_first(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
) -> None:
    assert [e.turn_id for e in store.search_turns(limit=3)] == ["t5", "t4", "t3"]


def test_recent_history_is_oldest_first_and_skips_failed_turns(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
) -> None:
    """The panel reads top to bottom, and an empty-ASR turn has nothing to show."""
    entries = store.recent_history(10)
    assert [e.turn_id for e in entries] == ["t1", "t2", "t3", "t5"]


def test_recent_history_crosses_sessions(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
) -> None:
    """After a restart the preceding exchanges are still the working context."""
    assert {e.session_id for e in store.recent_history(10)} == {"session-a", "session-b"}
    only_b = store.recent_history(10, session_id="session-b")
    assert {e.session_id for e in only_b} == {"session-b"}


def test_search_matches_source_and_translation(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
) -> None:
    """An operator recalls whichever side they were reading."""
    assert [e.turn_id for e in store.search_turns(query="회의")] == ["t1"]
    assert [e.turn_id for e in store.search_turns(query="meeting")] == ["t1"]


def test_search_escapes_like_wildcards(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
) -> None:
    store.add_turn("pct", "session-a", "ko", "en", "가격이 100% 올랐다", "Prices rose 100%")
    assert [e.turn_id for e in store.search_turns(query="100%")] == ["pct"]
    assert store.count_turns(query="%") == 1


def test_filters_compose(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
) -> None:
    assert store.count_turns(src_lang="ko", outcome="ok") == 2
    assert store.count_turns(src_lang="ko", outcome="empty_asr") == 1
    assert [e.turn_id for e in store.search_turns(src_lang="ja")] == ["t3"]


def test_history_outcomes_include_asr_service_failures() -> None:
    assert "asr_failed" in OUTCOMES


def test_target_filter_matches_broadcast_lists(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
) -> None:
    """Broadcast routing stores a comma-separated target, which must still match."""
    assert [e.turn_id for e in store.search_turns(tgt_lang="ja")] == ["t5"]


def test_pagination_walks_the_result_set(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
) -> None:
    first = store.search_turns(limit=2, offset=0)
    second = store.search_turns(limit=2, offset=2)
    assert [e.turn_id for e in first] == ["t5", "t4"]
    assert [e.turn_id for e in second] == ["t3", "t2"]


def test_turn_lookup_and_language_list(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
) -> None:
    assert store.turn("t3").translation == "I will share the materials"
    assert store.turn("missing") is None
    assert store.history_languages() == ["en", "ja", "ko"]


def test_session_summaries_count_turns(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
) -> None:
    summaries = {s.session_id: s for s in store.session_summaries()}
    assert summaries["session-a"].turns == 2
    assert summaries["session-b"].turns == 3


def test_add_turn_returns_the_stored_timestamp(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
) -> None:
    """The panel and the database must agree rather than drift by a round trip."""
    ts = store.add_turn("t9", "session-a", "ko", "en", "테스트", "test")
    assert store.turn("t9").ts == ts


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
            f"concurrent-{sequence}",
            "session-a",
            "ko",
            "en",
            str(sequence),
            str(sequence),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(add_turn, range(64)))

    assert store.count_turns(session_id="session-a") == 66


def test_store_file_is_owner_readable_and_writable_only(
    _positional_only: object | None = None,
    /,
    *,
    store: Store,
) -> None:
    assert store.path.stat().st_mode & 0o777 == 0o600
    for suffix in ("-wal", "-shm"):
        sidecar = store.path.with_name(f"{store.path.name}{suffix}")
        if sidecar.exists():
            assert sidecar.stat().st_mode & 0o777 == 0o600


def test_store_prunes_turns_and_sessions_to_bounded_retention(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    limited_store = Store(
        tmp_path / "limited.db",
        maximum_turns=10,
        maximum_sessions=2,
    )
    try:
        for session_index in range(3):
            limited_store.start_session(f"session-{session_index}", "pair", {})
        for turn_index in range(100):
            limited_store.add_turn(
                f"turn-{turn_index}",
                "session-2",
                "ko",
                "en",
                str(turn_index),
                str(turn_index),
            )

        assert limited_store.count_turns() == 10
        assert len(limited_store.session_summaries(limit=10)) == 2
    finally:
        limited_store.close()


def test_store_rejects_a_symbolic_link_database_path(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
) -> None:
    protected_file = tmp_path / "protected.db"
    protected_file.write_text("preserve", encoding="utf-8")
    database_path = tmp_path / "kotonoha.db"
    database_path.symlink_to(protected_file)

    with pytest.raises(OSError):
        Store(database_path)

    assert protected_file.read_text(encoding="utf-8") == "preserve"


# -- export -----------------------------------------------------------------
def test_export_writes_one_json_object_per_turn(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
    tmp_path: Any,
) -> None:
    target = tmp_path / "out" / "history.jsonl"
    export_jsonl(store.search_turns(limit=10), target)
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    first = json.loads(lines[0])
    assert first["turn_id"] == "t5"
    assert "time" in first and first["source_text"] == "방송 모드입니다"
    assert target.stat().st_mode & 0o777 == 0o600


def test_export_replaces_a_symbolic_link_without_overwriting_its_target(
    _positional_only: object | None = None,
    /,
    *,
    store: Any,
    tmp_path: Any,
) -> None:
    protected_file = tmp_path / "protected.txt"
    protected_file.write_text("preserve", encoding="utf-8")
    target = tmp_path / "history.jsonl"
    target.symlink_to(protected_file)

    export_jsonl(store.search_turns(limit=1), target)

    assert protected_file.read_text(encoding="utf-8") == "preserve"
    assert not target.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8"))["turn_id"] == "t5"


def test_excerpt_truncates_and_flattens() -> None:
    assert excerpt("a\nb") == "a b"
    assert excerpt("x" * 100, width=10).endswith("…")
    assert len(excerpt("x" * 100, width=10)) == 10
    assert excerpt(None) == ""


# -- interpreter panel ------------------------------------------------------
def _entry(
    turn_id: str,
    /,
    ts: float,
    source: str,
    translation: str,
) -> HistoryEntry:
    return HistoryEntry(
        turn_id=turn_id,
        ts=ts,
        session_id="s",
        src_lang="ko",
        tgt_lang="en",
        source_text=source,
        translation=translation,
        lang_source="lid",
        lid_confidence=0.9,
        asr_avg_logprob=-0.2,
        cross_verified=False,
        audio_seconds=2.0,
        outcome="ok",
    )


def test_panel_keeps_only_the_configured_number_of_turns() -> None:
    pane = HistoryPane("History", limit=2)
    pane.load([_entry(f"t{i}", 1000.0 + i, f"source {i}", f"translation {i}") for i in range(5)])
    rendered = pane.render().plain
    assert "translation 4" in rendered
    assert "translation 3" in rendered
    assert "translation 2" not in rendered


def test_panel_appends_completed_turns() -> None:
    pane = HistoryPane("History", limit=10)
    pane.load([])
    pane.append(1000.0, "ko", "en", "안녕하세요", "Hello")
    rendered = pane.render().plain
    assert "안녕하세요" in rendered and "Hello" in rendered
    assert "ko→en" in rendered
    assert "╭─ KO" in rendered
    assert "    ╭─ EN" in rendered
    assert rendered.index("안녕하세요") < rendered.index("Hello")


def test_empty_panel_says_so() -> None:
    set_locale("en")
    pane = HistoryPane("History", limit=10)
    assert translate_to("en", "No past turns") in pane.render().plain


# -- browser ----------------------------------------------------------------
async def test_browser_lists_stored_turns(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("KOTONOHA__STORE__PATH", str(tmp_path / "history.db"))
    from kotonoha._config import load_settings

    seed = Store(tmp_path / "history.db")
    seed.start_session("s", "pair", {})
    seed.add_turn("t1", "s", "ko", "en", "회의는 세 시입니다", "The meeting is at three")
    seed.add_turn("t2", "s", "en", "ko", "Send the list", "목록을 보내주세요")
    seed.close()

    app = HistoryApp(settings=load_settings())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.total == 2
        assert app.table.row_count == 2
        # Newest first, and the preview follows the top row.
        assert app.entries[0].turn_id == "t2"
        assert "목록을 보내주세요" in app.detail.render().plain


async def test_browser_search_filters_the_table(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("KOTONOHA__STORE__PATH", str(tmp_path / "history.db"))
    from kotonoha._config import load_settings

    seed = Store(tmp_path / "history.db")
    seed.start_session("s", "pair", {})
    seed.add_turn("t1", "s", "ko", "en", "회의는 세 시입니다", "The meeting is at three")
    seed.add_turn("t2", "s", "en", "ko", "Send the list", "목록을 보내주세요")
    seed.close()

    app = HistoryApp(settings=load_settings())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query.text = "meeting"
        await app.reload()
        assert app.total == 1
        assert app.table.row_count == 1


async def test_browser_export_writes_the_visible_rows(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("KOTONOHA__STORE__PATH", str(tmp_path / "history.db"))
    monkeypatch.setenv("KOTONOHA__ROOT", str(tmp_path))
    from kotonoha._config import load_settings

    seed = Store(tmp_path / "history.db")
    seed.start_session("s", "pair", {})
    seed.add_turn("t1", "s", "ko", "en", "안녕하세요", "Hello")
    seed.close()

    app = HistoryApp(settings=load_settings())
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.action_export()
        exported = sorted((tmp_path / "data" / "exports").glob("history-*.jsonl"))
        assert len(exported) == 1
        assert json.loads(exported[0].read_text(encoding="utf-8"))["turn_id"] == "t1"


async def test_browser_titles_follow_the_locale(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("KOTONOHA__STORE__PATH", str(tmp_path / "history.db"))
    from kotonoha._config import load_settings

    set_locale("zh-TW")
    app = HistoryApp(settings=load_settings())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.title == translate_to("zh-TW", "Interpretation history")
        assert [b.description for b in app._bindings.shown_keys] == [
            translate_to("zh-TW", "Search"),
            translate_to("zh-TW", "Reload"),
            translate_to("zh-TW", "Export"),
            translate_to("zh-TW", "Next page"),
            translate_to("zh-TW", "Previous page"),
            translate_to("zh-TW", "Back"),
            translate_to("zh-TW", "Quit"),
        ]
