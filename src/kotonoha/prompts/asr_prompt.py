"""Context-biasing prompt for Qwen3-ASR.

Qwen3-ASR accepts free-form context to bias transcription. Two things go in:

  1. The source text of recent turns, which cuts down CJK homophone errors.
  2. Proper nouns and terms that are likely to come up.

Plus the Traditional Chinese requirement from §5: **the prompt itself is written
in Traditional characters.** Simplified context pulls the model towards
Simplified output, and post-processing the output with s2twp cannot undo the
vocabulary choices that follow from that.
"""

from __future__ import annotations

from ..store.db import GlossaryEntry, TurnRecord

# Long context makes the model hallucinate towards the context instead of the
# audio. Cap it.
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
    """The context string handed to ASR. Empty string when there is nothing to say."""
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
