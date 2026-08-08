from __future__ import annotations

from kotonoha.core._clauses import ClauseStreamer
from kotonoha.prompts._translate import SRC_MARKER, parse_llm_output


def feed(
    text: str,
    /,
    chunk: int = 3,
) -> tuple[list[str], ClauseStreamer]:
    cs = ClauseStreamer()
    out: list[str] = []
    for i in range(0, len(text), chunk):
        out += cs.push(text[i : i + chunk])
    out += cs.flush()
    return out, cs


def test_splits_on_sentence_end() -> None:
    out, _ = feed("First sentence here. Second sentence here. Third one.")
    assert len(out) == 3
    assert out[0].endswith(".")


def test_cjk_terminators() -> None:
    out, _ = feed("今日は会議があります。資料を共有してください。")
    assert out == ["今日は会議があります。", "資料を共有してください。"]


def test_first_clause_is_allowed_short() -> None:
    """Only the first clause may be short, so the first audio goes out sooner."""
    out, _ = feed("네, 알겠습니다. 그럼 다음 주에 뵙겠습니다.")
    assert out[0] == "네, 알겠습니다."


def test_marker_stops_stream_even_when_split_across_deltas() -> None:
    """Nothing after ⟦SRC⟧ (the reconstructed source) may reach TTS."""
    body = "Please send it by Tuesday."
    src = "다음 주 화요일까지 보내주세요."
    out, cs = feed(body + "\n" + SRC_MARKER + "\n" + src, chunk=1)
    joined = " ".join(out)
    assert "화요일까지" not in joined
    assert SRC_MARKER not in joined
    assert cs.stopped
    assert cs.translation.strip() == body


def test_parse_llm_output() -> None:
    t, s = parse_llm_output(f"Hello there.\n{SRC_MARKER}\n안녕하세요.")
    assert t == "Hello there."
    assert s == "안녕하세요."

    t2, s2 = parse_llm_output("No marker at all.")
    assert t2 == "No marker at all." and s2 is None


def test_no_duplicate_emission() -> None:
    text = "One. Two. Three."
    out, _ = feed(text, chunk=1)
    assert "".join(out).replace(" ", "") == text.replace(" ", "")


def test_long_run_without_terminator_is_cut() -> None:
    text = ("word " * 60).strip()
    out, _ = feed(text, chunk=7)
    assert len(out) > 1
    assert all(len(c) <= 120 for c in out)


def test_long_cjk_run_without_spaces_is_hard_bounded() -> None:
    text = "字" * 400
    out, _ = feed(text, chunk=7)

    assert "".join(out) == text
    assert all(len(clause) <= 90 for clause in out)
