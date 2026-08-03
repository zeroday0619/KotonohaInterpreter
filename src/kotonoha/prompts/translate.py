"""정정 + 번역 단일 LLM 패스 (§5.3).

두 단계로 나누면 오류가 누적된다. 전사 정정을 먼저 시키고 그 결과를 다시
번역기에 넣으면, 정정 단계가 만든 오류를 번역 단계가 확대한다. 그래서
N-best 후보 + 교차검증 가설 + 히스토리 + 용어집을 한꺼번에 주고
"문맥으로 정정한 뒤 번역"까지 한 번에 시킨다.

출력 형식이 중요하다. 번역문을 **먼저** 흘려야 절 단위로 TTS 에 넘길 수 있다.
정정된 원문은 화면 표시·히스토리용이라 급하지 않으므로 맨 뒤 마커 뒤에 붙인다.
5 tok/s 환경에서 원문을 앞에 두면 첫 음성이 그만큼 통째로 밀린다.
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
    """(번역문, 정정된 원문 or None). 스트리밍이 끝난 뒤 전체 문자열에 대해 호출."""
    if SRC_MARKER not in text:
        return text.strip(), None
    head, _, tail = text.partition(SRC_MARKER)
    translation = head.strip()
    corrected = tail.strip() or None
    return translation, corrected
