"""Breakdown of retrieval metrics by stratum: query type, category, table-ness.

Computes dense-only ranks for the teacher arm (F) and the student arm (G) in memory
(vectors from ready Qdrant indexes, query embeddings on the fly) and groups the
metrics by:
  * query `type`: lexical / multihop / semantic / table;
  * `category`: construction / food / it / pharma / security;
  * `is_table` of the gold chunk.

    .venv/bin/python -m contextual_chunks.queries.breakdown
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from contextual_chunks.queries.ablate import (
    PROMPTS,
    QUERIES,
    RESULTS,
    load_arm_vectors,
    norm_rows,
    top_k_ids,
)
from contextual_chunks.queries.metrics import Ranking, doc_of, gold_groups, metrics
from contextual_chunks.retrieval.config import arm
from contextual_chunks.retrieval.corpus import load
from contextual_chunks.retrieval.dense import make_embedder

KEYS = ("recall@1", "recall@10", "mrr@10", "ndcg@10", "doc_recall@10", "miss@50")

def rankings_for(cfg, backend: str, tag: str, qs, prompt: str, dim: int | None):
    ids, vecs, _ = load_arm_vectors(cfg, backend, tag)
    chunks = load(context_source=cfg.context_source)
    idx_to_doc = doc_of(chunks)
    groups = gold_groups(chunks, [(q["qid"], q["gold_idx"]) for q in qs])
    gold_doc = {q["qid"]: chunks[q["gold_idx"]].doc_id for q in qs}

    embedder = make_embedder(backend, dim=cfg.embed_dim, max_length=cfg.embed_max_length,
                             token_budget=cfg.token_budget)
    t0 = time.perf_counter()
    qv = norm_rows(np.asarray(embedder.encode([q["q"] for q in qs], prompt=PROMPTS[prompt],
                                              sort_by_length=False), dtype=np.float32))
    if dim and dim < vecs.shape[1]:
        vecs, qv = norm_rows(vecs[:, :dim].copy()), norm_rows(qv[:, :dim].copy())
    ranks = [Ranking(q["qid"], top_k_ids(vecs @ qv[i], ids, 50)) for i, q in enumerate(qs)]
    print(f"  {cfg.name}: {vecs.shape[1]}d, {len(ids)} chunks, {time.perf_counter() - t0:.1f}s")
    return ranks, groups, idx_to_doc, gold_doc

def table(lines: list[str], title: str, keys: list[str], rows: list[tuple[str, dict, dict]],
          out: dict) -> None:
    lines += [f"## {title}", "",
              "| strata | n | F R@10 | G R@10 | Δ R@10 | F MRR@10 | G MRR@10 | Δ MRR@10 |",
              "|---|---|---|---|---|---|---|---|"]
    for label, mf, mg in rows:
        n = mf["_n"]
        lines.append(f"| {label} | {n} | {mf['recall@10']:.3f} | {mg['recall@10']:.3f} | "
                     f"{mg['recall@10'] - mf['recall@10']:+.3f} | {mf['mrr@10']:.3f} | "
                     f"{mg['mrr@10']:.3f} | {mg['mrr@10'] - mf['mrr@10']:+.3f} |")
        out[label] = {"n": n, "F": {k: mf[k] for k in keys}, "G": {k: mg[k] for k in keys}}
    lines.append("")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mlx")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--dim", type=int, default=None, help="MRL truncation (e.g. 512)")
    ap.add_argument("--prompt", default="en", choices=sorted(PROMPTS))
    ap.add_argument("--out", type=Path, default=RESULTS / "breakdown_by_stratum.md")
    args = ap.parse_args()

    qs = [json.loads(line) for line in QUERIES.read_text(encoding="utf-8").splitlines()
          if line.strip()]
    qs = [q for q in qs if not q.get("reject") and q.get("q")]
    print(f"queries: {len(qs)}")

    print("teacher arm F:")
    rf, gf, idf, gdf = rankings_for(arm("F"), args.backend, args.tag, qs, args.prompt, args.dim)
    print("student arm G:")
    rg, gg, idg, gdg = rankings_for(arm("G"), args.backend, args.tag, qs, args.prompt, args.dim)

    by_qid_f = {r.qid: r for r in rf}
    by_qid_g = {r.qid: r for r in rg}
    chunks = load(context_source="teacher")

    def sub(qids: list[str]):
        r_f = [by_qid_f[q] for q in qids]
        r_g = [by_qid_g[q] for q in qids]
        mf = metrics(r_f, gf, idf, gdf, k_max=50)
        mg = metrics(r_g, gg, idg, gdg, k_max=50)
        mf["_n"] = mg["_n"] = len(qids)
        return mf, mg

    all_q = [q["qid"] for q in qs]
    mf, mg = sub(all_q)

    lines = ["# Retrieval breakdown by stratum (F teacher vs G student)", "",
             f"queries {len(qs)}, dense-only, d={args.dim or 1024}, instruction {args.prompt.upper()}",
             "", "| strata | n | F R@10 | G R@10 | Δ | F MRR@10 | G MRR@10 | Δ |",
             "|---|---|---|---|---|---|---|---|",
             f"| **all** | {len(qs)} | {mf['recall@10']:.3f} | {mg['recall@10']:.3f} | "
             f"{mg['recall@10'] - mf['recall@10']:+.3f} | {mf['mrr@10']:.3f} | "
             f"{mg['mrr@10']:.3f} | {mg['mrr@10'] - mf['mrr@10']:+.3f} |", ""]
    out: dict = {"queries": len(qs), "dim": args.dim or 1024, "prompt": args.prompt,
                 "all": {"F": {k: mf[k] for k in KEYS}, "G": {k: mg[k] for k in KEYS}},
                 "strata": {}}

    for field, title in (("type", "By query type"), ("category", "By category")):
        vals = sorted({q[field] for q in qs if q.get(field)})
        rows = []
        for v in vals:
            qids = [q["qid"] for q in qs if q.get(field) == v]
            rows.append((v, *sub(qids)))
        table(lines, title, list(KEYS), rows, out["strata"].setdefault(field, {}))

    rows = []
    for label, pred in (("gold-таблица", lambda q: chunks[q["gold_idx"]].is_table),
                        ("gold-текст", lambda q: not chunks[q["gold_idx"]].is_table)):
        qids = [q["qid"] for q in qs if pred(q)]
        rows.append((label, *sub(qids)))
    table(lines, "By table-ness of the gold chunk", list(KEYS), rows, out["strata"].setdefault("is_table", {}))

    args.out.write_text("\n".join(lines), encoding="utf-8")
    (args.out.with_suffix(".json")).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
