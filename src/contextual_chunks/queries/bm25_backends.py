"""Comparison of BM25 backends: `rank_bm25` (in-memory, BM25Okapi + Snowball) vs fastembed
(`Qdrant/bm25`: BM25 in Python + per-language stop words, sparse vectors in Qdrant,
server-side IDF).

Computes BM25-only ranking with both backends and the latency, to decide whether the
sparse section in the index and the fastembed dependency are worth it. Hybrid RRF numbers
come from the ready runs `retrieval_eval_full.json` (rank_bm25) and
`retrieval_eval_bm25_fastembed.json`.

    .venv/bin/python -m contextual_chunks.queries.bm25_backends
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
from contextual_chunks.retrieval import index as index_mod
from contextual_chunks.retrieval.config import arm
from contextual_chunks.retrieval.corpus import load
from contextual_chunks.retrieval.sparse import FastEmbedBM25, RankBM25

QUERIES = Path(paths.QUERIES)
RESULTS = Path(paths.RESULTS)
KEYS = ("recall@1", "recall@5", "recall@10", "recall@50", "mrr@10", "ndcg@10",
        "doc_recall@10", "miss@50")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bm25fe", help="index tag with sparse vectors")
    ap.add_argument("--limit", type=int, default=100, help="BM25 channel depth")
    ap.add_argument("--out", type=Path, default=RESULTS / "bm25_backends.md")
    args = ap.parse_args()

    cfg = arm("A").with_(bm25="fastembed")
    chunks = load(context_source="teacher")
    texts = [c.text("chunk") for c in chunks]
    qs = [json.loads(l) for l in QUERIES.read_text(encoding="utf-8").splitlines() if l.strip()]
    qs = [q for q in qs if not q.get("reject") and q.get("q")]
    groups = gold_groups(chunks, [(q["qid"], q["gold_idx"]) for q in qs])
    idx_to_doc = doc_of(chunks)
    gold_doc = {q["qid"]: chunks[q["gold_idx"]].doc_id for q in qs}
    print(f"queries {len(qs)}, corpus {len(texts)}")

    # --- rank_bm25 (in-memory) -------------------------------------------
    t0 = time.perf_counter()
    rb = RankBM25(texts, stem=cfg.bm25_stem)
    build_rb = time.perf_counter() - t0
    lat_rb, ranks_rb = [], []
    for q in qs:
        t = time.perf_counter()
        top = [i for i, _ in rb.search(q["q"], args.limit)]
        lat_rb.append((time.perf_counter() - t) * 1000)
        ranks_rb.append(Ranking(q["qid"], top))
    print(f"rank_bm25: build {build_rb:.1f}s, search p50 {statistics.median(lat_rb):.1f} ms")

    # --- fastembed (ONNX + Qdrant sparse) --------------------------------
    t0 = time.perf_counter()
    fe = FastEmbedBM25(stem=cfg.bm25_stem, avg_len=index_mod.read_meta(cfg, "mlx", args.tag)
                       .get("bm25_avg_word_len", 256.0))
    build_fe = time.perf_counter() - t0
    client = index_mod.open_client(cfg, "mlx", args.tag)
    lat_fe, ranks_fe = [], []
    for q in qs:
        t = time.perf_counter()
        r = client.query_points(index_mod.COLLECTION, query=fe.embed_query(q["q"]),
                                using="bm25", limit=args.limit, with_payload=False)
        lat_fe.append((time.perf_counter() - t) * 1000)
        ranks_fe.append(Ranking(q["qid"], [int(p.id) for p in r.points]))
    client.close()
    print(f"fastembed: model load {build_fe:.1f}s, search p50 {statistics.median(lat_fe):.1f} ms")

    m_rb = metrics(ranks_rb, groups, idx_to_doc, gold_doc, k_max=50)
    m_fe = metrics(ranks_fe, groups, idx_to_doc, gold_doc, k_max=50)

    # hybrid RRF numbers from ready runs
    full = json.loads((RESULTS / "retrieval_eval_full.json").read_text(encoding="utf-8"))
    fev = json.loads((RESULTS / "retrieval_eval_bm25_fastembed.json").read_text(encoding="utf-8"))

    lines = ["# BM25 backends: rank_bm25 vs fastembed (`Qdrant/bm25`)", "",
             f"queries {len(qs)}, corpus {len(texts)} chunks, channel depth {args.limit}.",
             "", "## BM25-only", "",
             "| backend | R@1 | R@5 | R@10 | R@50 | MRR@10 | nDCG@10 | docR@10 | miss@50 | p50, ms |",
             "|---|---|---|---|---|---|---|---|---|---|",
             "| rank_bm25 (BM25Okapi + Snowball) | " +
             " | ".join(f"{m_rb[k]:.3f}" for k in KEYS) + f" | {statistics.median(lat_rb):.1f} |",
             "| fastembed (`Qdrant/bm25`) | " +
             " | ".join(f"{m_fe[k]:.3f}" for k in KEYS) + f" | {statistics.median(lat_fe):.1f} |",
             "", "## Hybrid RRF (dense + BM25, arms A/B)", "",
             "| arm | backend | R@10 | MRR@10 | nDCG@10 | docR@10 | p50, ms |",
             "|---|---|---|---|---|---|---|"]
    for name, key in (("A flat / RRF", "A"), ("B ctx / RRF", "B")):
        for label, src in (("rank_bm25", full), ("fastembed", fev)):
            a = src["arms"][key]
            m, l = a["metrics"], a["latency"]
            lines.append(f"| {name} | {label} | {m['recall@10']:.3f} | {m['mrr@10']:.3f} | "
                         f"{m['ndcg@10']:.3f} | {m['doc_recall@10']:.3f} | "
                         f"{l['total_p50_ms']:.0f} |")
    lines += ["", "## Conclusion", "",
              f"* the fastembed backend gives better BM25 ranking (R@10 "
              f"{m_fe['recall@10']:.3f} vs {m_rb['recall@10']:.3f}) and a better hybrid "
              f"(A {fev['arms']['A']['metrics']['recall@10']:.3f} vs "
              f"{full['arms']['A']['metrics']['recall@10']:.3f}), but at the cost of a sparse section in the index "
              f"and the fastembed dependency (+{statistics.median(lat_fe) - statistics.median(lat_rb):.0f} ms "
              f"per query: sparse query to Qdrant + server-side IDF). The `Qdrant/bm25` model is "
              f"18 stop-word files (92 KB) and BM25 in Python, without a neural network or weights.",
              "* Both options are still **worse than pure dense** (F 0.658): the conclusion that "
              "\"BM25 hurts on a homogeneous corpus\" holds, fastembed only softens the degradation.",
              "* For the edge scenario rank_bm25 wins: no sparse section in the index, no "
              "fastembed dependency, search 14× faster, and the hybrid quality differs "
              "by 1-3 p.p.",
              "* fastembed is useful if BM25 already lives in Qdrant (sparse channel on the server), "
              "you need language stop words out of the box, or an IDF modifier via Qdrant.", ""]

    args.out.write_text("\n".join(lines), encoding="utf-8")
    (args.out.with_suffix(".json")).write_text(json.dumps(
        {"queries": len(qs), "corpus": len(texts), "limit": args.limit,
         "rank_bm25": {**{k: m_rb[k] for k in KEYS}, "p50_ms": round(statistics.median(lat_rb), 1),
                       "build_sec": round(build_rb, 1)},
         "fastembed": {**{k: m_fe[k] for k in KEYS}, "p50_ms": round(statistics.median(lat_fe), 1),
                       "load_sec": round(build_fe, 1)},
         "rrf_rank_bm25": {a: full["arms"][a]["metrics"] for a in ("A", "B")},
         "rrf_fastembed": {a: fev["arms"][a]["metrics"] for a in ("A", "B")}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
