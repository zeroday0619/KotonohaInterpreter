"""품질 게이트 (§5.5).

1차 ASR 평균 로그확률이 임계 미달일 때만 Whisper 를 부른다. 상시 호출하면
매 턴 0.8초가 붙어서 지연 예산이 무너진다.

두 가설이 크게 다르면 LLM 에 둘 다 넘겨 판단시킨다. '크게 다르다'의 기준은
문자 오류율이다 — 토큰 단위로 재면 CJK 에서 의미가 없다.
"""

from __future__ import annotations

import unicodedata


def should_cross_verify(
    avg_logprob: float,
    threshold: float,
    n_best: list[str] | None = None,
    duration_s: float = 0.0,
) -> tuple[bool, str]:
    """(발동 여부, 사유)."""
    if avg_logprob < threshold:
        return True, f"avg_logprob {avg_logprob:.3f} < {threshold}"

    # N-best 상위 후보끼리도 크게 갈리면 신뢰도 점수와 무관하게 흔들리는 것이다.
    if n_best and len(n_best) >= 2:
        d = cer(n_best[0], n_best[1])
        if d > 0.5:
            return True, f"n-best disagreement cer={d:.2f}"

    # 아주 짧은 발화는 로그확률이 과대평가되기 쉽다. 비용도 작으니 확인한다.
    if 0.0 < duration_s < 1.0:
        return True, f"short utterance {duration_s:.2f}s"

    return False, ""


def normalize_for_compare(s: str) -> str:
    """비교 전 정규화: NFKC, 소문자, 공백·문장부호 제거."""
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
    """문자 오류율. 0.0 = 동일, 1.0 이상 = 완전히 다름."""
    r = normalize_for_compare(ref)
    h = normalize_for_compare(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def is_divergent(a: str, b: str, threshold: float) -> bool:
    return cer(a, b) >= threshold
