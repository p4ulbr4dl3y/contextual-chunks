"""Smoke: indexes on a subsample + real queries over all arms.

    .venv/bin/python -m contextual_chunks.retrieval.smoke_search --limit 300
    .venv/bin/python -m contextual_chunks.retrieval.smoke_search --limit 300 --arms B D

Checks: indexing (dense+BM25), native and manual RRF, the reranker, timings,
two BM25 backends. Without gold chunks this is a qualitative check, not metrics.
"""

from __future__ import annotations

import argparse
import sys
import time

from contextual_chunks.retrieval import corpus as corpus_mod
from contextual_chunks.retrieval import index as index_mod
from contextual_chunks.retrieval.config import ARMS, BASE
from contextual_chunks.retrieval.dense import make_embedder, make_reranker
from contextual_chunks.retrieval.fuse import overlap_at_k
from contextual_chunks.retrieval.search import Searcher

QUERIES = [
    "какой срок поставки товара по контракту",
    "размер обеспечения исполнения контракта",
    "штраф за просрочку исполнения обязательств поставщиком",
    "требования к упаковке и маркировке продукции",
    "порядок оплаты и размер аванса",
    "гарантийный срок на поставляемое оборудование",
    "условия расторжения контракта в одностороннем порядке",
    "адрес и реквизиты заказчика",
]


def short(s: str, n: int = 88) -> str:
    return " ".join(s.split())[:n]


def show(hits, n: int = 5) -> None:
    for h in hits[:n]:
        tag = f"d{h.dense_rank or '-'}/b{h.bm25_rank or '-'}"
        rr = f" rr{h.rerank_rank} {h.rerank_score:.3f}" if h.rerank_rank else ""
        print(f"    [{h.score:.4f} {tag}{rr}] {h.doc_id.split('_')[0]:12s} {short(h.text)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300, help="corpus subsample size")
    ap.add_argument("--arms", nargs="*", default=["A", "B", "C", "D"])
    ap.add_argument("--backend", default="mlx", choices=["mlx", "torch"])
    ap.add_argument("--queries", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--check-fusion", action="store_true", help="native RRF vs manual")
    args = ap.parse_args()

    queries = args.queries or list(QUERIES)
    chunks = corpus_mod.load(limit=args.limit)
    tag = f"smoke{args.limit}"
    print(f"corpus: {corpus_mod.stats(chunks)} | index tag={tag}")

    cfgs = [ARMS[a.upper()] for a in args.arms]
    for cfg in {c.index_fingerprint(): c for c in cfgs}.values():
        if not index_mod.exists(cfg, args.backend, tag):
            index_mod.build(cfg, chunks, backend=args.backend, recreate=True, tag=tag)

    print(f"\n=== model load timings (backend={args.backend}) ===")
    t0 = time.perf_counter()
    emb = make_embedder(args.backend)
    print(f"  embedder: {time.perf_counter()-t0:.2f}s")
    t0 = time.perf_counter()
    rrk = make_reranker(args.backend)
    print(f"  reranker: {time.perf_counter()-t0:.2f}s")
    del emb, rrk

    timings_by_arm: dict[str, list[float]] = {}
    for cfg in cfgs:
        searcher = Searcher(cfg, backend=args.backend, chunks=chunks, tag=tag)
        idx_meta = index_mod.read_meta(cfg, args.backend, tag) or {}
        print(f"\n=== arm {cfg.name} | index_text={cfg.index_text} | rerank={cfg.rerank} "
              f"| fusion={searcher.effective_fusion}"
              f"{' (native)' if cfg.fusion == 'native' and searcher.effective_fusion == 'manual' else ''}"
              f" bm25={cfg.bm25} k={cfg.rrf_k}"
              f"{' weights=' + str(cfg.weights) if cfg.weights else ''} "
              f"| index {idx_meta.get('points_count')} points "
              f"{idx_meta.get('total_sec')}s ===")
        tq: list[float] = []
        for q in queries:
            T: dict = {}
            hits = searcher.search(q, top_k=args.top, timings=T)
            tq.append(T["total_ms"])
            print(f"  Q: {q}")
            show(hits, args.top)
            rerank = f" | rerank {T['rerank_ms']:.1f}ms" if "rerank_ms" in T else ""
            print(f"     embed {T['embed_ms']:.1f}ms | fuse {T['fuse_ms']:.1f}ms{rerank}"
                  f" | total {T['total_ms']:.1f}ms")
        timings_by_arm[cfg.name] = tq
        print(f"  -- query latency: p50 {sorted(tq)[len(tq)//2]:.1f}ms "
              f"min {min(tq):.1f}ms max {max(tq):.1f}ms (n={len(tq)}, without warmup)")

        if args.check_fusion and searcher.effective_fusion == "native":
            manual = searcher.cfg.with_(fusion="manual")
            s2 = Searcher(manual, backend=args.backend, chunks=chunks, tag=tag)
            ov, ov5 = [], []
            for q in queries:
                a = [h.idx for h in searcher.search(q, top_k=20)]
                b = [h.idx for h in s2.search(q, top_k=20)]
                ov.append(overlap_at_k([str(i) for i in a], [str(i) for i in b], 10))
                ov5.append(overlap_at_k([str(i) for i in a], [str(i) for i in b], 5))
            s2.close()
            print(f"  -- native vs manual RRF: overlap@5 {sum(ov5)/len(ov5):.2f}, "
                  f"overlap@10 {sum(ov)/len(ov):.2f}")
        searcher.close()

    # two BM25 backends: comparison of the purely lexical channel
    from contextual_chunks.retrieval.sparse import FastEmbedBM25
    if not FastEmbedBM25.available():
        print("\n=== BM25 backends: skipped (fastembed Qdrant/bm25 not in the cache, "
              "network without a proxy required) ===")
        print(f"\n=== query latency summary (n={len(queries)}, without warmup) ===")
        for name, ts in timings_by_arm.items():
            print(f"  {name:28s} p50 {sorted(ts)[len(ts)//2]:7.1f}ms  max {max(ts):7.1f}ms")
        return 0
    print("\n=== BM25 backends (BM25 channel only, top-20) ===")
    rankings: dict[str, dict[str, list[str]]] = {}
    for be in ("fastembed", "rank_bm25"):
        cfg = BASE.with_(index_text="chunk", bm25=be)
        if not index_mod.exists(cfg, args.backend, tag):
            index_mod.build(cfg, chunks, backend=args.backend, recreate=True, tag=tag)
        s = Searcher(cfg, backend=args.backend, chunks=chunks, tag=tag)
        rankings[be] = {q: [str(i) for i in (s._sparse_top_qdrant(q, 20) if be == "fastembed"
                                            else s._sparse_top_memory(q, 20))] for q in queries}
        s.close()
    o5 = [overlap_at_k(rankings["fastembed"][q], rankings["rank_bm25"][q], 5) for q in queries]
    o20 = [overlap_at_k(rankings["fastembed"][q], rankings["rank_bm25"][q], 20) for q in queries]
    print(f"  rank_bm25+snowball vs fastembed Qdrant/bm25: overlap@5 {sum(o5)/len(o5):.2f}, "
          f"overlap@20 {sum(o20)/len(o20):.2f}")

    print(f"\n=== query latency summary (n={len(queries)}, without warmup) ===")
    for name, ts in timings_by_arm.items():
        print(f"  {name:28s} p50 {sorted(ts)[len(ts)//2]:7.1f}ms  max {max(ts):7.1f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
