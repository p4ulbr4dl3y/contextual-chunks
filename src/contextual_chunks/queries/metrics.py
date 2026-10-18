"""Retrieval experiment metrics that account for overlapping chunks.

Corpus chunks are overlapping windows of one document (median Jaccard of adjacent
pairs 0.33, 31% of pairs > 0.5), so "hit the wrong chunk" often means "hit the same
text". Metrics are computed over the gold group: any chunk of the document whose
Jaccard with the gold is >= the threshold counts as a hit. Document-level hit is
computed separately, as a diagnostic of whether the right document was found at all.

    groups = gold_groups(chunks, [(qid, gold_idx), ...])
    metrics(rankings, groups, docs)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from contextual_chunks.queries.build import jaccard, stem_ws


@dataclass
class Ranking:
    qid: str
    ids: list[int]          # ordered corpus idx (top-k)


def gold_groups(chunks, pairs: list[tuple[str, int]], threshold: float = 0.9,
                radius: int = 2) -> dict[str, set[int]]:
    """qid -> idx equivalent to the gold.

    Chunks are overlapping windows, so a hit in a neighbouring chunk (+-radius over
    doc_chunk_index) means the same text. `threshold` catches verbatim duplicates
    elsewhere in the document (repeated tables), but it must be high: at 0.5 the
    group grows to ~130 chunks on long documents.
    """
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    out: dict[str, set[int]] = {}
    for qid, gold_idx in pairs:
        gold = next((c for c in chunks if c.idx == gold_idx), None)
        if gold is None:
            out[qid] = set()
            continue
        g_st = set(stem_ws(gold.chunk))
        group = {gold_idx}
        for c in by_doc[gold.doc_id]:
            if c.idx == gold_idx:
                continue
            if abs(c.doc_chunk_index - gold.doc_chunk_index) <= radius:
                group.add(c.idx)
            elif jaccard(g_st, set(stem_ws(c.chunk))) >= threshold:
                group.add(c.idx)
        out[qid] = group
    return out


def doc_of(chunks) -> dict[int, str]:
    return {c.idx: c.doc_id for c in chunks}


def first_hit(ids: list[int], group: set[int]) -> int:
    for i, idx in enumerate(ids, start=1):
        if idx in group:
            return i
    return 0


def metrics(rankings: list[Ranking], groups: dict[str, set[int]],
            idx_to_doc: dict[int, str] | None = None, gold_doc: dict[str, str] | None = None,
            k_max: int = 50) -> dict[str, float]:
    n = max(1, len(rankings))
    out: dict[str, float] = {}
    for k in (1, 5, 10, 50):
        if k > k_max:
            continue
        out[f"recall@{k}"] = round(
            sum(1 for r in rankings if groups.get(r.qid) and first_hit(r.ids[:k], groups[r.qid])) / n, 4)
    pos = [first_hit(r.ids, groups.get(r.qid, set())) for r in rankings]
    out["mrr@10"] = round(sum(1 / p for p in pos if 0 < p <= 10) / n, 4)
    out["ndcg@10"] = round(sum(1 / math.log2(p + 1) for p in pos if 0 < p <= 10) / n, 4)
    out["miss@50"] = round(sum(1 for p in pos if p == 0) / n, 4)
    if idx_to_doc and gold_doc:
        out["doc_recall@10"] = round(sum(
            1 for r in rankings
            if any(idx_to_doc.get(i) == gold_doc.get(r.qid) for i in r.ids[:10])) / n, 4)
    out["avg_group"] = round(sum(len(groups.get(r.qid, ())) for r in rankings) / n, 1)
    return out
