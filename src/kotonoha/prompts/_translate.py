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

from typing import Any, Final

from kotonoha._languages import LANGUAGE_NAMES, LANGUAGE_NATIVE_NAMES
from kotonoha._logging_setup import get_logger
from kotonoha.store._db import GlossaryEntry, TurnRecord

SRC_MARKER = "⟦SRC⟧"
MAXIMUM_TRANSLATION_PROMPT_CHARACTERS: Final[int] = 32768
log = get_logger(__name__)

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


def _format_history(
    history: list[TurnRecord],
    /,
) -> str:
    lines = []
    for turn in history:
        if not turn.source_text:
            continue
        source_name = LANGUAGE_NATIVE_NAMES.get(turn.src_lang or "", turn.src_lang or "?")
        lines.append(f"[{source_name}] {turn.source_text}")
        if turn.translation:
            target_name = LANGUAGE_NATIVE_NAMES.get(turn.tgt_lang or "", turn.tgt_lang or "?")
            lines.append(f"[{target_name}] {turn.translation}")
    return "\n".join(lines)


def _format_glossary(
    glossary: list[GlossaryEntry],
    /,
) -> str:
    return "\n".join(
        f"- {entry.src_term} → {entry.tgt_term}"
        + (f"  ({entry.note})" if entry.note else "")
        for entry in glossary
    )


def build_translate_messages(
    hypotheses: list[str],
    /,
    source_language: str,
    target_language: str,
    history: list[TurnRecord] | None = None,
    glossary: list[GlossaryEntry] | None = None,
    verify_hypothesis: str | None = None,
    verify_divergent: bool = False,
) -> list[dict[str, Any]]:
    system = SYSTEM.format(
        source_name=LANGUAGE_NAMES.get(source_language, source_language),
        target_name=LANGUAGE_NAMES.get(target_language, target_language),
        target_native=LANGUAGE_NATIVE_NAMES.get(target_language, target_language),
        marker=SRC_MARKER,
    )
    if target_language == "zh-TW":
        system += "\n" + _TW_RULE

    formatted_hypotheses = "\n".join(
        f"{index + 1}. {text}"
        for index, text in enumerate(hypotheses)
        if text.strip()
    )
    required_blocks = [
        "## ASR hypotheses "
        f"({LANGUAGE_NAMES.get(source_language, source_language)}), best first\n"
        f"{formatted_hypotheses}"
    ]

    if verify_hypothesis:
        note = (
            "This second engine disagrees substantially with the list above. "
            "Decide which reading is right; do not blend them."
            if verify_divergent
            else "A second ASR engine produced this. Use it to break ties."
        )
        required_blocks.append(f"## Second engine\n{verify_hypothesis}\n\n{note}")

    required_blocks.append(
        f"Now output the {LANGUAGE_NAMES.get(target_language, target_language)} translation."
    )

    retained_history = list(history or [])
    dropped_history_turns = 0
    while True:
        blocks: list[str] = []
        if retained_history:
            formatted_history = _format_history(retained_history)
            if formatted_history:
                blocks.append("## Conversation so far\n" + formatted_history)
        if glossary:
            blocks.append("## Glossary (apply verbatim)\n" + _format_glossary(glossary))
        blocks.extend(required_blocks)
        prompt = f"{system}\n\n" + "\n\n".join(blocks)
        if len(prompt) <= MAXIMUM_TRANSLATION_PROMPT_CHARACTERS:
            break
        if not retained_history:
            raise ValueError(
                "translation prompt exceeds "
                f"{MAXIMUM_TRANSLATION_PROMPT_CHARACTERS} characters without history"
            )
        retained_history.pop(0)
        dropped_history_turns += 1

    if dropped_history_turns:
        log.warning(
            "translation.history_truncated",
            dropped_turns=dropped_history_turns,
            retained_turns=len(retained_history),
            prompt_characters=len(prompt),
        )
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_language,
                    "target_lang_code": target_language,
                    "text": prompt,
                }
            ],
        }
    ]


def parse_llm_output(
    text: str,
    /,
) -> tuple[str, str | None]:
    """Returns (translation, reconstructed source or None).

    Call this on the complete string once streaming has finished.
    """
    if SRC_MARKER not in text:
        return text.strip(), None
    translation_text, _, source_text = text.partition(SRC_MARKER)
    translation = translation_text.strip()
    corrected = source_text.strip() or None
    return translation, corrected
