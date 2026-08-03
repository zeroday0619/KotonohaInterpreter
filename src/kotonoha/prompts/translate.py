"""Correction and translation in a single LLM pass (§5.3).

Splitting this into two stages compounds the errors: correct the transcript
first, feed the result to a translator, and the translator amplifies whatever
the correction stage got wrong. So the N-best candidates, the cross-verification
hypothesis, the history and the glossary all go in at once, and the model is
asked to reconstruct from context *and* translate in one shot.

The output format matters. The translation has to stream **first** so it can be
handed to TTS clause by clause. The reconstructed source is only needed for the
display and the history, so it goes last, behind a marker. At 5 tok/s, putting
the source first would delay the first audio by exactly its length.
"""

from __future__ import annotations

from .. import LANG_NAMES, LANG_NATIVE
from ..store.db import GlossaryEntry, TurnRecord

SRC_MARKER = "⟦SRC⟧"

SYSTEM = """You are a professional consecutive interpreter. You are given automatic \
speech-recognition (ASR) hypotheses of ONE utterance, plus conversation context.

Do BOTH of these in a single pass:
1. Silently reconstruct what the speaker actually said. The hypotheses contain \
recognition errors, especially homophone confusions in Korean, Japanese and Chinese. \
Use the conversation history and the glossary to decide. Never invent content that is \
not supported by at least one hypothesis.
2. Translate that reconstruction into {target_name} ({target_native}).

Hard rules:
- Translate DIRECTLY from {source_name} to {target_name}. Never pivot through English.
- Output the translation FIRST, as plain text, with no preamble, labels, or quotes.
- Preserve numbers, dates, units and proper nouns exactly.
- Apply every glossary entry verbatim when its source term appears.
- Keep the register of the original (polite stays polite, casual stays casual).
- If the hypotheses are pure noise or empty, output exactly: {marker}
- After the translation, emit the marker {marker} on its own line, then the \
reconstructed {source_name} sentence. Nothing after that."""

_TW_RULE = (
    "- The target is Taiwan Mandarin. Use Traditional characters and Taiwanese "
    "vocabulary (軟體, 影片, 資訊, 滑鼠, 網路, 程式), never Mainland forms."
)


def _fmt_history(history: list[TurnRecord]) -> str:
    lines = []
    for h in history:
        if not h.source_text:
            continue
        src = LANG_NATIVE.get(h.src_lang or "", h.src_lang or "?")
        lines.append(f"[{src}] {h.source_text}")
        if h.translation:
            tgt = LANG_NATIVE.get(h.tgt_lang or "", h.tgt_lang or "?")
            lines.append(f"[{tgt}] {h.translation}")
    return "\n".join(lines)


def _fmt_glossary(glossary: list[GlossaryEntry]) -> str:
    return "\n".join(
        f"- {g.src_term} → {g.tgt_term}" + (f"  ({g.note})" if g.note else "")
        for g in glossary
    )


def build_translate_messages(
    n_best: list[str],
    source_lang: str,
    target_lang: str,
    history: list[TurnRecord] | None = None,
    glossary: list[GlossaryEntry] | None = None,
    verify_hypothesis: str | None = None,
    verify_divergent: bool = False,
) -> list[dict[str, str]]:
    system = SYSTEM.format(
        source_name=LANG_NAMES.get(source_lang, source_lang),
        target_name=LANG_NAMES.get(target_lang, target_lang),
        target_native=LANG_NATIVE.get(target_lang, target_lang),
        marker=SRC_MARKER,
    )
    if target_lang == "zh-TW":
        system += "\n" + _TW_RULE

    blocks: list[str] = []

    if history:
        h = _fmt_history(history)
        if h:
            blocks.append("## Conversation so far\n" + h)

    if glossary:
        blocks.append("## Glossary (apply verbatim)\n" + _fmt_glossary(glossary))

    hyps = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(n_best) if t.strip())
    blocks.append(
        f"## ASR hypotheses ({LANG_NAMES.get(source_lang, source_lang)}), best first\n{hyps}"
    )

    if verify_hypothesis:
        note = (
            "This second engine disagrees substantially with the list above. "
            "Decide which reading is right; do not blend them."
            if verify_divergent
            else "A second ASR engine produced this. Use it to break ties."
        )
        blocks.append(f"## Second engine\n{verify_hypothesis}\n\n{note}")

    blocks.append(f"Now output the {LANG_NAMES.get(target_lang, target_lang)} translation.")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]


def parse_llm_output(text: str) -> tuple[str, str | None]:
    """Returns (translation, reconstructed source or None).

    Call this on the complete string once streaming has finished.
    """
    if SRC_MARKER not in text:
        return text.strip(), None
    head, _, tail = text.partition(SRC_MARKER)
    translation = head.strip()
    corrected = tail.strip() or None
    return translation, corrected
