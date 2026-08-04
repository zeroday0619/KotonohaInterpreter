"""Interpretation history: store queries, the interpreter panel, and the browser."""

from __future__ import annotations

import json

import pytest

from kotonoha.i18n import set_locale, translate_to
from kotonoha.store.db import HistoryEntry, Store
from kotonoha.tui.app import HistoryPane
from kotonoha.tui.history_app import HistoryApp, excerpt, export_jsonl


@pytest.fixture(autouse=True)
def _reset_locale():
    yield
    set_locale(None)


@pytest.fixture
def store(tmp_path):
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
def test_search_returns_newest_first(store):
    assert [e.turn_id for e in store.search_turns(limit=3)] == ["t5", "t4", "t3"]


def test_recent_history_is_oldest_first_and_skips_failed_turns(store):
    """The panel reads top to bottom, and an empty-ASR turn has nothing to show."""
    entries = store.recent_history(10)
    assert [e.turn_id for e in entries] == ["t1", "t2", "t3", "t5"]


def test_recent_history_crosses_sessions(store):
    """After a restart the preceding exchanges are still the working context."""
    assert {e.session_id for e in store.recent_history(10)} == {"session-a", "session-b"}
    only_b = store.recent_history(10, session_id="session-b")
    assert {e.session_id for e in only_b} == {"session-b"}


def test_search_matches_source_and_translation(store):
    """An operator recalls whichever side they were reading."""
    assert [e.turn_id for e in store.search_turns(query="회의")] == ["t1"]
    assert [e.turn_id for e in store.search_turns(query="meeting")] == ["t1"]


def test_search_escapes_like_wildcards(store):
    store.add_turn("pct", "session-a", "ko", "en", "가격이 100% 올랐다", "Prices rose 100%")
    assert [e.turn_id for e in store.search_turns(query="100%")] == ["pct"]
    assert store.count_turns(query="%") == 1


def test_filters_compose(store):
    assert store.count_turns(src_lang="ko", outcome="ok") == 2
    assert store.count_turns(src_lang="ko", outcome="empty_asr") == 1
    assert [e.turn_id for e in store.search_turns(src_lang="ja")] == ["t3"]


def test_target_filter_matches_broadcast_lists(store):
    """Broadcast routing stores a comma-separated target, which must still match."""
    assert [e.turn_id for e in store.search_turns(tgt_lang="ja")] == ["t5"]


def test_pagination_walks_the_result_set(store):
    first = store.search_turns(limit=2, offset=0)
    second = store.search_turns(limit=2, offset=2)
    assert [e.turn_id for e in first] == ["t5", "t4"]
    assert [e.turn_id for e in second] == ["t3", "t2"]


def test_turn_lookup_and_language_list(store):
    assert store.turn("t3").translation == "I will share the materials"
    assert store.turn("missing") is None
    assert store.history_languages() == ["en", "ja", "ko"]


def test_session_summaries_count_turns(store):
    summaries = {s.session_id: s for s in store.session_summaries()}
    assert summaries["session-a"].turns == 2
    assert summaries["session-b"].turns == 3


def test_add_turn_returns_the_stored_timestamp(store):
    """The panel and the database must agree rather than drift by a round trip."""
    ts = store.add_turn("t9", "session-a", "ko", "en", "테스트", "test")
    assert store.turn("t9").ts == ts


# -- export -----------------------------------------------------------------
def test_export_writes_one_json_object_per_turn(store, tmp_path):
    target = tmp_path / "out" / "history.jsonl"
    export_jsonl(store.search_turns(limit=10), target)
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    first = json.loads(lines[0])
    assert first["turn_id"] == "t5"
    assert "time" in first and first["source_text"] == "방송 모드입니다"


def test_excerpt_truncates_and_flattens():
    assert excerpt("a\nb") == "a b"
    assert excerpt("x" * 100, width=10).endswith("…")
    assert len(excerpt("x" * 100, width=10)) == 10
    assert excerpt(None) == ""


# -- interpreter panel ------------------------------------------------------
def _entry(turn_id: str, ts: float, source: str, translation: str) -> HistoryEntry:
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


def test_panel_keeps_only_the_configured_number_of_turns():
    pane = HistoryPane("History", limit=2)
    pane.load([_entry(f"t{i}", 1000.0 + i, f"source {i}", f"translation {i}") for i in range(5)])
    rendered = pane.render().plain
    assert "translation 4" in rendered
    assert "translation 3" in rendered
    assert "translation 2" not in rendered


def test_panel_appends_completed_turns():
    pane = HistoryPane("History", limit=10)
    pane.load([])
    pane.append(1000.0, "ko", "en", "안녕하세요", "Hello")
    rendered = pane.render().plain
    assert "안녕하세요" in rendered and "Hello" in rendered
    assert "ko→en" in rendered


def test_empty_panel_says_so():
    set_locale("en")
    pane = HistoryPane("History", limit=10)
    assert translate_to("en", "No past turns") in pane.render().plain


# -- browser ----------------------------------------------------------------
async def test_browser_lists_stored_turns(tmp_path, monkeypatch):
    monkeypatch.setenv("KOTONOHA__STORE__PATH", str(tmp_path / "history.db"))
    from kotonoha.config import load_settings

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


async def test_browser_search_filters_the_table(tmp_path, monkeypatch):
    monkeypatch.setenv("KOTONOHA__STORE__PATH", str(tmp_path / "history.db"))
    from kotonoha.config import load_settings

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


async def test_browser_export_writes_the_visible_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("KOTONOHA__STORE__PATH", str(tmp_path / "history.db"))
    monkeypatch.setenv("KOTONOHA__ROOT", str(tmp_path))
    from kotonoha.config import load_settings

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


async def test_browser_titles_follow_the_locale(tmp_path, monkeypatch):
    monkeypatch.setenv("KOTONOHA__STORE__PATH", str(tmp_path / "history.db"))
    from kotonoha.config import load_settings

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
