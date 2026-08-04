"""Quality gate (§5.5).

Whisper is only called when the primary ASR's average log-probability falls
below threshold. Calling it every turn adds 0.8 s each time and destroys the
latency budget.

When the two hypotheses disagree badly, both go to the LLM. "Badly" is measured
as character error rate — token-level metrics are meaningless for CJK.
"""

from __future__ import annotations

import unicodedata


def should_cross_verify(
    average_log_probability: float,
    threshold: float,
    hypotheses: list[str] | None = None,
    duration_seconds: float = 0.0,
) -> tuple[bool, str]:
    """Return whether cross-verification is required and the activation reason."""
    if average_log_probability < threshold:
        return True, f"avg_logprob {average_log_probability:.3f} < {threshold}"

    # If the top N-best candidates disagree sharply, the model is wobbling
    # regardless of what its confidence score says.
    if hypotheses and len(hypotheses) >= 2:
        disagreement = character_error_rate(hypotheses[0], hypotheses[1])
        if disagreement > 0.5:
            return True, f"n-best disagreement cer={disagreement:.2f}"

    # Very short utterances tend to have inflated log-probabilities and require
    # explicit verification under the configured quality policy.
    if 0.0 < duration_seconds < 1.0:
        return True, f"short utterance {duration_seconds:.2f}s"

    return False, ""


def normalize_for_comparison(text: str) -> str:
    """Normalise before comparing: NFKC, lowercase, strip whitespace and punctuation."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and unicodedata.category(character)[0] not in ("P", "Z")
    )


def levenshtein(reference: str, hypothesis: str) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous_row = list(range(len(hypothesis) + 1))
    for reference_index, reference_character in enumerate(reference, 1):
        current_row = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(hypothesis, 1):
            current_row.append(
                min(
                    previous_row[hypothesis_index] + 1,
                    current_row[hypothesis_index - 1] + 1,
                    previous_row[hypothesis_index - 1]
                    + (reference_character != hypothesis_character),
                )
            )
        previous_row = current_row
    return previous_row[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Character error rate. 0.0 means identical, >= 1.0 means completely different."""
    normalized_reference = normalize_for_comparison(reference)
    normalized_hypothesis = normalize_for_comparison(hypothesis)
    if not normalized_reference:
        return 0.0 if not normalized_hypothesis else 1.0
    return levenshtein(normalized_reference, normalized_hypothesis) / len(
        normalized_reference
    )


def is_divergent(reference: str, hypothesis: str, threshold: float) -> bool:
    return character_error_rate(reference, hypothesis) >= threshold
