"""Retrieval ablations without rebuilding the index: MRL dimension, instruction
language, RRF k/weights, top-k BM25, dense-only vs hybrid.

Corpus vectors are taken directly from the Qdrant index (the arm is set by `--arm`),
query vectors are computed by the embedder on the fly. Everything else is numpy +
rank_bm25 in memory, so a run takes minutes rather than hours (the index is not rebuilt).

    .venv/bin/python -m contextual_chunks.queries.ablate --arm F
    .venv/bin/python -m contextual_chunks.queries.ablate --arm G --out-suffix student
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from contextual_chunks import paths
from contextual_chunks.queries.metrics import Ranking, doc_of, gold_groups, metrics
from contextual_chunks.retrieval import index as index_mod
from contextual_chunks.retrieval.config import arm
from contextual_chunks.retrieval.corpus import load
from contextual_chunks.retrieval.dense import make_embedder
from contextual_chunks.retrieval.models import EMB_QUERY_PROMPT
from contextual_chunks.retrieval.sparse import RankBM25

PROMPT_RU = ("Инструкция: по поисковому запросу найди фрагменты, "
             "которые отвечают на этот запрос\nЗапрос:")
PROMPTS = {"en": EMB_QUERY_PROMPT, "ru": PROMPT_RU}
QUERIES = Path(paths.QUERIES)
RESULTS = Path(paths.RESULTS)


def norm_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def load_arm_vectors(cfg, backend: str = "mlx", tag: str = "full"):
    """Corpus vectors + payload from a ready index (order = point id)."""
    client = index_mod.open_client(cfg, backend, tag)
    pts, _ = client.scroll(index_mod.COLLECTION, limit=10_000, with_vectors=True,
                           with_payload=True)
    pts.sort(key=lambda p: int(p.id))
    ids = np.array([int(p.id) for p in pts])
    vecs = norm_rows(np.stack([np.array(p.vector["dense"], dtype=np.float32) for p in pts]))
    payload = [p.payload for p in pts]
    client.close()
    return ids, vecs, payload


def rrf(rank_lists: list[list[int]], k: int, weights: list[float] | None = None) -> list[int]:
    score: dict[int, float] = {}
    w = weights or [1.0] * len(rank_lists)
    for rl, wi in zip(rank_lists, w, strict=False):
        for pos, idx in enumerate(rl, start=1):
            score[idx] = score.get(idx, 0.0) + wi / (k + pos)
    return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def top_k_ids(scores: np.ndarray, ids: np.ndarray, k: int) -> list[int]:
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    part = part[np.argsort(-scores[part])]
    return [int(ids[i]) for i in part]


def evaluate(name: str, rankings: list[Ranking], groups, idx_to_doc, gold_doc,
             lines: list[str], out: dict, extra: dict | None = None) -> None:
    m = metrics(rankings, groups, idx_to_doc, gold_doc, k_max=50)
    out[name] = {**m, **(extra or {})}
    lines.append(f"| {name} | " + " | ".join(f"{m[k]:.3f}" for k in
                 ("recall@1", "recall@5", "recall@10", "recall@50", "mrr@10",
                  "ndcg@10", "doc_recall@10", "miss@50")) + " |")
    print(f"  {name:38s} R@1={m['recall@1']:.3f} R@10={m['recall@10']:.3f} "
          f"MRR@10={m['mrr@10']:.3f} nDCG@10={m['ndcg@10']:.3f} miss@50={m['miss@50']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="F", help="arm that provides the vectors (F/G are dense-only)")
    ap.add_argument("--backend", default="mlx")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--dims", nargs="*", type=int, default=[1024, 512, 256, 128])
    ap.add_argument("--prompts", nargs="*", default=["en", "ru"])
    ap.add_argument("--rrf-k", nargs="*", type=int, default=[1, 2, 10, 60])
    ap.add_argument("--pool", type=int, default=100, help="channel depth for RRF")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    cfg = arm(args.arm)
    print(f"arm {args.arm} ({cfg.name}), context={cfg.context_source}, fp={cfg.index_fingerprint()}")
    ids, vecs, payload = load_arm_vectors(cfg, args.backend, args.tag)
    texts = [p["text"] for p in payload]
    print(f"vectors: {vecs.shape}, texts {len(texts)}")

    qs = [json.loads(line) for line in QUERIES.read_text(encoding="utf-8").splitlines()
          if line.strip()]
    qs = [q for q in qs if not q.get("reject") and q.get("q")]
    print(f"queries: {len(qs)}")

    chunks = load(context_source=cfg.context_source)
    groups = gold_groups(chunks, [(q["qid"], q["gold_idx"]) for q in qs])
    idx_to_doc = doc_of(chunks)
    gold_doc = {q["qid"]: chunks[q["gold_idx"]].doc_id for q in qs}

    embedder = make_embedder(args.backend, dim=cfg.embed_dim, max_length=cfg.embed_max_length,
                             token_budget=cfg.token_budget)
    qvecs: dict[str, np.ndarray] = {}
    for pname in args.prompts:
        t0 = time.perf_counter()
        v = embedder.encode([q["q"] for q in qs], prompt=PROMPTS[pname], sort_by_length=False)
        qvecs[pname] = norm_rows(np.asarray(v, dtype=np.float32))
        print(f"query embeddings ({pname}): {qvecs[pname].shape} in {time.perf_counter()-t0:.1f}s")

    print("BM25 ranking...")
    bm25 = RankBM25(texts, stem=cfg.bm25_stem)
    bm25_ranks = [[i for i, _ in bm25.search(q["q"], args.pool)] for q in qs]

    out: dict = {"arm": args.arm, "name": cfg.name, "context_source": cfg.context_source,
                 "queries": len(qs), "corpus": len(ids), "ablations": {}}
    lines = [f"# Retrieval ablations - arm {args.arm} ({cfg.name})", "",
             f"queries {len(qs)}, corpus {len(ids)} chunks, backend {args.backend}, tag {args.tag}",
             "", "Metrics over the gold group (±2 windows). Channels: dense = cosine, BM25 = rank_bm25.",
             "", "| variant | R@1 | R@5 | R@10 | R@50 | MRR@10 | nDCG@10 | docR@10 | miss@50 |",
             "|---|---|---|---|---|---|---|---|---|"]

    # --- 1. MRL: embedding dimension ------------------------------------
    print("\n[1] MRL dimension (dense-only)")
    lines += ["", "## 1. MRL: embedding dimension (dense-only)", "",
              "| variant | R@1 | R@5 | R@10 | R@50 | MRR@10 | nDCG@10 | docR@10 | miss@50 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for d in args.dims:
        if d > vecs.shape[1]:
            continue
        vd, qd = norm_rows(vecs[:, :d].copy()), norm_rows(qvecs["en"][:, :d].copy())
        rankings = []
        for i, q in enumerate(qs):
            scores = vd @ qd[i]
            rankings.append(Ranking(q["qid"], top_k_ids(scores, ids, 50)))
        evaluate(f"dense d={d}", rankings, groups, idx_to_doc, gold_doc, lines, out["ablations"],
                 {"dim": d})

    # --- 2. instruction language ----------------------------------------
    print("\n[2] query instruction language (dense-only, d=1024)")
    lines += ["", "## 2. Query instruction language (dense-only, d=1024)", "",
              "| variant | R@1 | R@5 | R@10 | R@50 | MRR@10 | nDCG@10 | docR@10 | miss@50 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for pname in args.prompts:
        rankings = []
        for i, q in enumerate(qs):
            scores = vecs @ qvecs[pname][i]
            rankings.append(Ranking(q["qid"], top_k_ids(scores, ids, 50)))
        evaluate(f"инструкция {pname.upper()}", rankings, groups, idx_to_doc, gold_doc,
                 lines, out["ablations"], {"prompt": pname})

    # --- 3. RRF: k, weights, channel composition -------------------------
    print("\n[3] RRF: k, weights, channel composition")
    lines += ["", "## 3. Hybrid RRF: k, weights, channel composition", "",
              "| variant | R@1 | R@5 | R@10 | R@50 | MRR@10 | nDCG@10 | docR@10 | miss@50 |",
              "|---|---|---|---|---|---|---|---|---|"]
    dense_ranks = []
    for i, _q in enumerate(qs):
        scores = vecs @ qvecs["en"][i]
        dense_ranks.append(top_k_ids(scores, ids, args.pool))
    combos = [("dense-only", [dense_ranks]), ("BM25-only", [bm25_ranks])]
    for k in args.rrf_k:
        combos.append((f"RRF k={k} 1:1", None))
        out.setdefault("_rrf", {})
    for name, ranks in combos:
        if ranks is not None:
            rankings = [Ranking(q["qid"], ranks[0][i]) for i, q in enumerate(qs)]
            evaluate(name, rankings, groups, idx_to_doc, gold_doc, lines, out["ablations"])
            continue
        k = int(name.split("k=")[1].split()[0])
        rankings = [Ranking(q["qid"], rrf([dense_ranks[i], bm25_ranks[i]], k))
                    for i, q in enumerate(qs)]
        evaluate(name, rankings, groups, idx_to_doc, gold_doc, lines, out["ablations"], {"rrf_k": k})
    # dense:bm25 weights at k=60
    for wd, wb in ((1.0, 0.5), (1.0, 0.25), (1.0, 2.0), (1.0, 0.0)):
        if wb == 0.0:
            continue
        rankings = [Ranking(q["qid"], rrf([dense_ranks[i], bm25_ranks[i]], 60, [wd, wb]))
                    for i, q in enumerate(qs)]
        evaluate(f"RRF k=60 w={wd:g}:{wb:g}", rankings, groups, idx_to_doc, gold_doc,
                 lines, out["ablations"], {"weights": [wd, wb]})

    # --- 4. BM25 channel depth ------------------------------------------
    print("\n[4] BM25 channel depth in RRF (k=60)")
    lines += ["", "## 4. BM25 channel depth (RRF k=60, 1:1)", "",
              "| variant | R@1 | R@5 | R@10 | R@50 | MRR@10 | nDCG@10 | docR@10 | miss@50 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for pool in (20, 50, 100, 200):
        b_ranks = [[i for i, _ in bm25.search(q["q"], pool)] for q in qs]
        rankings = [Ranking(q["qid"], rrf([dense_ranks[i], b_ranks[i]], 60))
                    for i, q in enumerate(qs)]
        evaluate(f"BM25 pool={pool}", rankings, groups, idx_to_doc, gold_doc,
                 lines, out["ablations"], {"bm25_pool": pool})

    # --- 5. combinations: MRL x instruction language --------------------
    print("\n[5] MRL x instruction language combinations (dense-only)")
    lines += ["", "## 5. Combinations: MRL × instruction language (dense-only)", "",
              "| variant | R@1 | R@5 | R@10 | R@50 | MRR@10 | nDCG@10 | docR@10 | miss@50 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for d in (1024, 512, 256):
        for pname in args.prompts:
            vd = norm_rows(vecs[:, :d].copy())
            qd = norm_rows(qvecs[pname][:, :d].copy())
            rankings = []
            for i, q in enumerate(qs):
                rankings.append(Ranking(q["qid"], top_k_ids(vd @ qd[i], ids, 50)))
            evaluate(f"d={d} + {pname.upper()}", rankings, groups, idx_to_doc, gold_doc,
                     lines, out["ablations"], {"dim": d, "prompt": pname})

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    md = RESULTS / f"ablation_retrieval{suffix}.md"
    js = RESULTS / f"ablation_retrieval{suffix}.json"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    js.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {md}\n-> {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
