"""Run the query set over the article arms: quality (recall/MRR/nDCG) + latency.

Indexes must be built in advance (`build_index.py --arms A B --tag <tag>`).
Metrics are computed over the gold group (neighboring windows = the same text), so the
index must cover the whole corpus: if the gold chunk is not in the subsample, the query
is excluded and this is printed separately.

    .venv/bin/python -m contextual_chunks.retrieval.eval_queries \
        --queries artifacts/queries.jsonl --arms A B C D --tag full
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from contextual_chunks import paths
from contextual_chunks.queries.metrics import Ranking, doc_of, gold_groups, metrics
from contextual_chunks.retrieval.config import BM25_BACKENDS, arm
from contextual_chunks.retrieval.corpus import load
from contextual_chunks.retrieval.search import Searcher

RESULTS = Path(paths.RESULTS)
DEFAULT_Q = Path(paths.QUERIES)


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, default=DEFAULT_Q)
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--backend", default="mlx", choices=["mlx", "torch"])
    ap.add_argument("--tag", default="full")
    ap.add_argument("--context-source", default="teacher", choices=["teacher", "student"])
    ap.add_argument("--bm25", default=None, choices=list(BM25_BACKENDS),
                    help="override the BM25 backend (must match the index)")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="limit the number of queries (debug)")
    ap.add_argument("--stratified", type=int, default=0,
                    help="take N queries balanced by type (lexical/multihop/semantic/table)")
    ap.add_argument("--seed", type=int, default=17, help="seed for --stratified")
    ap.add_argument("--rerank-pool", type=int, default=0,
                    help="override rerank_pool (the reranker is expensive: 0.5 s/pair)")
    ap.add_argument("--group-radius", type=int, default=2)
    ap.add_argument("--group-threshold", type=float, default=0.9)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    good = [r for r in rows if not r.get("reject") and r.get("q")]
    if args.stratified:
        import random
        rng = random.Random(args.seed)
        by_type: dict[str, list] = {}
        for r in good:
            by_type.setdefault(r.get("type") or "other", []).append(r)
        per = args.stratified // max(1, len(by_type))
        picked = []
        for t in sorted(by_type):
            bucket = by_type[t][:]
            rng.shuffle(bucket)
            picked.extend(bucket[:per])
        good = sorted(picked, key=lambda r: r["qid"])
    elif args.limit:
        good = good[: args.limit]
    chunks = load(context_source=args.context_source)
    by_idx = {c.idx: c for c in chunks}
    have = set(by_idx)
    usable = [r for r in good if r["gold_idx"] in have]
    dropped = len(good) - len(usable)
    groups = gold_groups(chunks, [(r["qid"], r["gold_idx"]) for r in usable],
                         threshold=args.group_threshold, radius=args.group_radius)
    idx_to_doc = doc_of(chunks)
    gold_doc = {r["qid"]: by_idx[r["gold_idx"]].doc_id for r in usable}
    print(f"queries {len(good)}, in index {len(usable)} (dropped {dropped}), "
          f"corpus {len(chunks)} chunks, backend {args.backend}, tag {args.tag}")

    report: dict = {"queries": len(usable), "corpus": len(chunks), "backend": args.backend,
                    "tag": args.tag, "dropped": dropped, "arms": {}}
    lines = [f"# Retrieval run over arms - tag `{args.tag}`", "",
             f"queries {len(usable)} (of {len(good)}, dropped {dropped}: gold chunk outside the index), "
             f"corpus {len(chunks)} chunks, backend {args.backend}, top-k {args.top_k}", "",
             "Metrics over the gold group (neighboring windows +/-2 = the same text).", "",
             f"rerank pool: {args.rerank_pool or 'default'}", "",
             "| arm | recall@1 | recall@5 | recall@10 | recall@50 | MRR@10 | nDCG@10 | "
             "doc_recall@10 | miss@50 | p50 total, ms | p95 total, ms |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    for name in args.arms:
        cfg = arm(name)
        if args.bm25:
            cfg = cfg.with_(bm25=args.bm25)
        if args.rerank_pool:
            cfg = cfg.with_(rerank_pool=args.rerank_pool)
        t0 = time.time()
        try:
            searcher = Searcher(cfg, backend=args.backend, chunks=chunks, tag=args.tag)
        except Exception as e:  # index not built
            print(f"arm {name}: index unavailable ({e})")
            continue
        rankings, timings = [], []
        for r in usable:
            T: dict = {}
            hits = searcher.search(r["q"], top_k=args.top_k, timings=T)
            rankings.append(Ranking(r["qid"], [h.idx for h in hits]))
            timings.append(T)
        searcher.close()
        m = metrics(rankings, groups, idx_to_doc, gold_doc, k_max=args.top_k)
        lat = {"total_p50_ms": pct([t["total_ms"] for t in timings], 50),
               "total_p95_ms": pct([t["total_ms"] for t in timings], 95),
               "embed_p50_ms": pct([t.get("embed_ms", 0) for t in timings], 50),
               "fuse_p50_ms": pct([t.get("fuse_ms", 0) for t in timings], 50),
               "rerank_p50_ms": pct([t.get("rerank_ms", 0) for t in timings], 50),
               "total_mean_ms": round(statistics.fmean(t["total_ms"] for t in timings), 1)}
        per_query = []
        for r, rk in zip(usable, rankings, strict=False):
            qid = r["qid"]
            mm = metrics([rk], {qid: groups[qid]}, idx_to_doc, {qid: gold_doc[qid]},
                         k_max=args.top_k)
            per_query.append({"qid": qid, "type": r.get("type"), "category": r.get("category"),
                              "recall@1": mm.get("recall@1"), "recall@10": mm.get("recall@10"),
                              "mrr@10": mm.get("mrr@10"), "ndcg@10": mm.get("ndcg@10")})
        report["arms"][name] = {"name": cfg.name, "metrics": m, "latency": lat,
                                "per_query": per_query, "sec": round(time.time() - t0, 1)}
        print(f"arm {name} ({cfg.name}): {m}")
        print(f"  latency: {lat} ({time.time() - t0:.0f}s)")
        lines.append(f"| {cfg.name} | {m.get('recall@1')} | {m.get('recall@5')} | "
                     f"{m.get('recall@10')} | {m.get('recall@50')} | {m.get('mrr@10')} | "
                     f"{m.get('ndcg@10')} | {m.get('doc_recall@10')} | {m.get('miss@50')} | "
                     f"{lat['total_p50_ms']} | {lat['total_p95_ms']} |")
    lines.append("")

    out = args.out or RESULTS / f"retrieval_eval_{args.tag}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
