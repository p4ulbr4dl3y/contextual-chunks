"""Audit of the query set: invariants + calibration against a baseline.

Invariants (exit 1 on violation): unique qid, the gold chunk exists, question length
8-25 words, no unique identifiers from the chunk (leakage), no shared n-grams with the
chunk, the judge confirmed the gold chunk, no duplicates, stratum coverage.

Calibration is the main acceptance criterion: BM25-only (no dense, no rerank) recall@10
must land in 0.3-0.6. Above 0.9 means the set is lexical (BM25 solves everything and the
difference between arms disappears); below 0.1 means the questions are unsolvable.

    .venv/bin/python -m contextual_chunks.queries.audit \
        --in artifacts/queries.jsonl --baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from contextual_chunks import paths
from contextual_chunks.queries.build import (
    df_table,
    is_identifier,
    jaccard,
    max_ngram,
    stem_ws,
    words,
)
from contextual_chunks.queries.metrics import Ranking, doc_of, gold_groups, metrics
from contextual_chunks.retrieval.corpus import load
from contextual_chunks.retrieval.sparse import RankBM25

DEFAULT_IN = Path(paths.QUERIES)
REPORT = Path(paths.RESULTS) / "queries_audit.md"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    ap.add_argument("--baseline", action="store_true", help="compute BM25-only metrics")
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--group-threshold", type=float, default=0.9,
                    help="Jaccard for verbatim duplicates of the gold chunk")
    ap.add_argument("--group-radius", type=int, default=2,
                    help="radius over doc_chunk_index: neighbouring windows = same text")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.inp.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    chunks = load()
    by_idx = {c.idx: c for c in chunks}
    df = df_table(chunks)
    good = [r for r in rows if not r.get("reject") and r.get("q")]
    problems: list[str] = []

    def fail(msg: str) -> None:
        problems.append(msg)

    seen_stems: list[tuple[str, set[str]]] = []
    for r in good:
        q_st = stem_ws(r["q"])
        if r["qid"] in [x["qid"] for x in good[:good.index(r)]]:
            fail(f"{r['qid']}: duplicate qid")
        gold = by_idx.get(r["gold_idx"])
        if gold is None:
            fail(f"{r['qid']}: no chunk with idx={r['gold_idx']}")
            continue
        if not (8 <= len(words(r["q"])) <= 25):
            fail(f"{r['qid']}: length {len(words(r['q']))} words")
        leak = [w for w in q_st if df[w] <= 2 and is_identifier(w)]
        if leak:
            fail(f"{r['qid']}: identifier leakage {leak[:3]}")
        ng = max_ngram(q_st, stem_ws(gold.chunk))
        if ng >= 5:
            fail(f"{r['qid']}: copy of the chunk, n-gram {ng}")
        if r.get("judge") != r.get("judge_letter"):
            fail(f"{r['qid']}: judge said {r.get('judge')}, gold {r.get('judge_letter')}")
        if any(jaccard(set(q_st), prev) > 0.6 for _, prev in seen_stems):
            fail(f"{r['qid']}: near-duplicate question")
        seen_stems.append((r["qid"], set(q_st)))

    # --- strata
    def dist(key: str) -> dict[str, int]:
        return dict(sorted(Counter(r[key] for r in good).items()))

    cats = dist("category")
    types = dist("type")
    gens = dist("generator")
    rejected = Counter((r.get("reject") or "OK").split(":")[0] for r in rows)

    # --- baseline: BM25-only over the raw chunk (arm A without dense/RRF)
    base: dict[str, float] | None = None
    if args.baseline and good:
        groups = gold_groups(chunks, [(r["qid"], r["gold_idx"]) for r in good],
                             threshold=args.group_threshold, radius=args.group_radius)
        bm = RankBM25([c.chunk for c in chunks], stem=True)
        rankings = [Ranking(r["qid"], [i for i, _ in bm.search(r["q"], limit=50)]) for r in good]
        base = metrics(rankings, groups, doc_of(chunks), {r["qid"]: by_idx[r["gold_idx"]].doc_id
                                                          for r in good})
        # lexical set: BM25 finds both the document and the fragment almost always
        if base["recall@10"] > 0.95 and base["mrr@10"] > 0.5:
            problems.append(f"CALIBRATION: BM25 recall@10={base['recall@10']}, "
                            f"mrr@10={base['mrr@10']} - the set is too lexical")
        elif base["mrr@10"] < 0.02 and base["recall@50"] < 0.3:
            problems.append(f"CALIBRATION: BM25 recall@50={base['recall@50']} - questions "
                            f"are unsolvable without context")

    lines = [f"# Query set audit - `{args.inp.name}`", "",
             f"candidates {len(rows)}, accepted {len(good)}, "
             f"rejected {len(rows) - len(good)}", "",
             f"rejection reasons: {dict(rejected)}", "",
             f"categories: {cats}", "", f"types: {types}", "", f"generators: {gens}", ""]
    if base:
        lines += [f"## Calibration: BM25-only (raw chunk, no dense, no rerank), "
                  f"gold group: neighbours ±{args.group_radius} / Jaccard>={args.group_threshold}", "",
                  "| metric | value |", "|---|---|"] + \
                 [f"| {k} | {v} |" for k, v in base.items()] + [""]
    lines += ["## Invariants", ""]
    lines += [f"* {p}" for p in problems] if problems else ["* no violations"]
    lines += [""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")

    if not args.quiet:
        print(f"candidates {len(rows)}, accepted {len(good)}, rejected {dict(rejected)}")
        print(f"categories {cats}")
        print(f"types      {types}")
        print(f"generators {gens}")
        if base:
            print(f"BM25-only: {base}")
        if problems:
            print("\nVIOLATIONS:")
            for p in problems:
                print(f"  * {p}")
        else:
            print("\ninvariants: no violations")
        print(f"-> {args.report}")
    return 1 if problems else 0

if __name__ == "__main__":
    sys.exit(main())
