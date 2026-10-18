"""Baseline check of the query set: can a dense model solve it (no context, no BM25).

Rationale: the set must separate the arms. BM25-only usually yields a low MRR (82
near-identical contracts -> lexis does not discriminate). If dense-only is also close to
zero, the questions are unsolvable and the set is useless; if dense is clearly better than
BM25, the set is about semantics.

    .venv/bin/python -m contextual_chunks.queries.baseline_dense \
        --in artifacts/queries.jsonl --backend mlx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


from contextual_chunks import paths
from contextual_chunks.queries.metrics import Ranking, doc_of, gold_groups, metrics
from contextual_chunks.retrieval.config import BASE
from contextual_chunks.retrieval.corpus import load
from contextual_chunks.retrieval.dense import embed_texts, make_embedder
from contextual_chunks.retrieval.models import EMB_QUERY_PROMPT

DEFAULT_IN = Path(paths.QUERIES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    ap.add_argument("--backend", default="mlx", choices=["mlx", "torch"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--group-threshold", type=float, default=0.9)
    ap.add_argument("--group-radius", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="first N accepted queries (0 = all)")
    ap.add_argument("--cache", type=Path, default=None,
                    help="defaults to the shared harness cache (one corpus pass for all runs)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.inp.read_text(encoding="utf-8").splitlines() if l.strip()]
    good = [r for r in rows if not r.get("reject") and r.get("q")]
    if args.limit:
        good = good[: args.limit]
    if not good:
        print("no accepted queries")
        return 1
    chunks = load()
    by_idx = {c.idx: c for c in chunks}

    cache = args.cache or BASE.dense_cache_path(args.backend)
    emb = make_embedder(args.backend)
    print(f"embedder {args.backend}, corpus {len(chunks)} chunks")
    # corpus tag, same as in index.build, otherwise the corpus is computed twice
    tag = f"{args.backend}-d{BASE.embed_dim or 0}-ml{BASE.embed_max_length}"
    print(f"embedding cache: {cache} (corpus tag {tag})")
    mat, info = embed_texts(emb, [c.chunk for c in chunks], cache, tag=tag,
                            batch_size=args.batch_size)
    qvec, qinfo = embed_texts(emb, [r["q"] for r in good], cache, tag=f"{tag}-q",
                              prompt=EMB_QUERY_PROMPT, batch_size=args.batch_size)
    sims = qvec @ mat.T
    rankings = [Ranking(r["qid"], [int(i) for i in np.argsort(-sims[k])[:50]])
                for k, r in enumerate(good)]
    groups = gold_groups(chunks, [(r["qid"], r["gold_idx"]) for r in good],
                         threshold=args.group_threshold, radius=args.group_radius)
    m = metrics(rankings, groups, doc_of(chunks),
                {r["qid"]: by_idx[r["gold_idx"]].doc_id for r in good})
    print(f"dense-only ({args.backend}, raw chunk, no context and no BM25):")
    for k, v in m.items():
        print(f"  {k:<14} {v}")
    out = Path(args.inp).with_name(Path(args.inp).stem + f"_dense_{args.backend}.json")
    out.write_text(json.dumps({"metrics": m, "embed": info, "queries": qinfo,
                               "n_queries": len(good)}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
