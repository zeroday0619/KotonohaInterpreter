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
    avg_logprob: float,
    threshold: float,
    n_best: list[str] | None = None,
    duration_s: float = 0.0,
) -> tuple[bool, str]:
    """Returns (should_fire, reason)."""
    if avg_logprob < threshold:
        return True, f"avg_logprob {avg_logprob:.3f} < {threshold}"

    # If the top N-best candidates disagree sharply, the model is wobbling
    # regardless of what its confidence score says.
    if n_best and len(n_best) >= 2:
        d = cer(n_best[0], n_best[1])
        if d > 0.5:
            return True, f"n-best disagreement cer={d:.2f}"

    # Very short utterances tend to have inflated log-probabilities, and the
    # check is cheap for them anyway.
    if 0.0 < duration_s < 1.0:
        return True, f"short utterance {duration_s:.2f}s"

    return False, ""


def normalize_for_compare(s: str) -> str:
    """Normalise before comparing: NFKC, lowercase, strip whitespace and punctuation."""
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(
        ch for ch in s if not ch.isspace() and unicodedata.category(ch)[0] not in ("P", "Z")
    )


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate. 0.0 means identical, >= 1.0 means completely different."""
    r = normalize_for_compare(ref)
    h = normalize_for_compare(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def is_divergent(a: str, b: str, threshold: float) -> bool:
    return cer(a, b) >= threshold
