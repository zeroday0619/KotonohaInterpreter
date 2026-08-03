"""Qwen3-ASR 컨텍스트 바이어스 프롬프트.

Qwen3-ASR 은 자유 형식 컨텍스트를 받아 전사를 편향시킬 수 있다. 두 가지를 넣는다.

  1. 직전 턴들의 원문 — CJK 동음이의 오인식을 줄인다.
  2. 등장 가능성이 높은 고유명사·용어.

그리고 §5 의 번체 요구사항: **프롬프트 자체를 번체로 쓴다.** 컨텍스트가 간체면
모델이 간체로 끌려간다. 출력 후처리(s2twp)만으로는 어휘 선택까지 되돌릴 수 없다.
"""

from __future__ import annotations

from ..store.db import GlossaryEntry, TurnRecord

# 컨텍스트가 길면 오히려 전사가 컨텍스트를 따라 환각한다. 상한을 둔다.
MAX_CONTEXT_CHARS = 600

_TW_HINT = (
    "本次錄音為臺灣華語。請以繁體中文（臺灣用語）輸出，"
    "例如：軟體、影片、資訊、滑鼠、網路、程式。"
)


def build_asr_context(
    history: list[TurnRecord],
    glossary: list[GlossaryEntry],
    expect_traditional: bool,
) -> str:
    """ASR 에 넘길 컨텍스트 문자열. 비어 있으면 빈 문자열."""
    parts: list[str] = []

    if expect_traditional:
        parts.append(_TW_HINT)

    terms: list[str] = []
    seen: set[str] = set()
    for g in glossary:
        for t in (g.src_term, g.tgt_term):
            if t and t not in seen:
                seen.add(t)
                terms.append(t)
    if terms:
        parts.append("關鍵詞 / keywords: " + ", ".join(terms[:40]))

    if history:
        recent = [h.source_text for h in history[-3:] if h.source_text]
        if recent:
            parts.append("上文 / context: " + " ".join(recent))

    ctx = "\n".join(parts)
    if len(ctx) > MAX_CONTEXT_CHARS:
        ctx = ctx[:MAX_CONTEXT_CHARS]
    return ctx
