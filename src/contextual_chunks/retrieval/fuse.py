"""Fusion: manual RRF (control) + cross-check against Qdrant's native RRF.

Qdrant semantics (`qdrant_client/hybrid/fusion.py`):

    score = 1 / ((pos + 1) / w + k - 1)        # pos is zero-based, DEFAULT k = 2
    == textbook 1 / (k_t + rank)  with  k_t = k - 1

That is, "Qdrant k=2" and "textbook k=1" are the same number. So in the config we pass
`rrf_k` explicitly and pin `convention` in the report.
"""

from __future__ import annotations


def manual_rrf(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None,
               convention: str = "qdrant") -> dict[str, float]:
    """rankings = lists of ids in descending relevance (one per channel).

    convention="qdrant"   -> 1/(pos + k)          (matches Qdrant at the same k)
    convention="textbook" -> 1/(k + pos + 1)      (RRF from the Cormack 2009 paper, k=60)
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    assert len(weights) == len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=False):
        if w <= 0:
            continue
        for pos, _id in enumerate(ranking):
            inc = 1.0 / ((pos + 1.0) / w + k - 1.0) if convention == "qdrant" \
                else w / (k + pos + 1.0)
            scores[_id] = scores.get(_id, 0.0) + inc
    return scores


def sorted_ids(scores: dict[str, float], limit: int | None = None) -> list[str]:
    out = [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
    return out[:limit] if limit else out


def overlap_at_k(a: list[str], b: list[str], k: int = 10) -> float:
    """Fraction of shared ids in the top-k, a sanity check of native RRF vs manual."""
    sa, sb = set(a[:k]), set(b[:k])
    return len(sa & sb) / max(1, k)
