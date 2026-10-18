"""Wrapper correctness against the model READMEs (regression gate).

References from the READMEs:
* Qwen3-Embedding-0.6B (transformers path): queries with the instruct prompt, documents
  without a prompt, last-token pooling, L2 norm:
  scores = [[0.7645568251609802, 0.14142508804798126], [0.13549736142158508, 0.5999549627304077]]
* Qwen3-Reranker-0.6B: prefix/suffix + `<Instruct>/<Query>/<Document>`,
  score = exp(log_softmax([no, yes])[yes]), true/false = 9693/2152.

Plus checks without which the benchmarks are invalid:
* batch/bucket-invariance: batched == one by one (catches padding/pooling bugs);
* (torch) `logits_to_keep=1` == full logits;
* MRL truncation yields normalized vectors of the requested dimension;
* both backends (torch and mlx) produce consistent scores.

Run:  .venv/bin/python -m contextual_chunks.retrieval.check_models [--backend torch|mlx|both]
"""

import argparse
import sys
from pathlib import Path


import numpy as np

from contextual_chunks.retrieval.models import EMB_QUERY_PROMPT

QUERIES = ["What is the capital of China?", "Explain gravity"]
DOCS = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to "
    "physical objects and is responsible for the movement of planets around the sun.",
]
EMB_REF = np.array([[0.7645568251609802, 0.14142508804798126],
                    [0.13549736142158508, 0.5999549627304077]])


def check(backend: str) -> tuple[list[str], dict]:
    fails: list[str] = []
    out: dict = {}
    if backend == "torch":
        from contextual_chunks.retrieval.models import Embedder as EmbCls
        from contextual_chunks.retrieval.models import Reranker as RrkCls
        from contextual_chunks.retrieval.models import pick_device, pick_dtype
        emb, rrk = EmbCls(), RrkCls()
        tol = 2e-3
        out["device"] = f"{pick_device()} {pick_dtype(pick_device())}"
    else:
        from contextual_chunks.retrieval.mlx_models import MlxEmbedder as EmbCls
        from contextual_chunks.retrieval.mlx_models import MlxReranker as RrkCls
        emb, rrk = EmbCls(), RrkCls()
        tol = 5e-3
        out["device"] = "mlx-gpu float16 (mlx_convert)"
    print(f"--- backend={backend} ({out['device']}) tolerance={tol}")

    eq = emb.encode(QUERIES, prompt=EMB_QUERY_PROMPT)
    ed = emb.encode(DOCS, prompt="")
    sim = eq @ ed.T
    d = float(np.abs(sim - EMB_REF).max())
    out["embed_sim"] = sim.tolist()
    print("embedding cosine:\n", np.round(sim, 4), f"\nmax |ours - README| = {d:.2e}")
    if d > tol:
        fails.append(f"embedding scores diverge from README by {d:.2e}")

    ed_one = np.stack([emb.encode([t], prompt="")[0] for t in DOCS])
    db = float(np.abs(ed - ed_one).max())
    out["batch_invariance"] = db
    print(f"batch-invariance (docs batched vs single) = {db:.2e}")
    if db > tol:
        fails.append(f"embedding batch-invariance broken: {db:.2e}")

    small = emb.encode(QUERIES, prompt=EMB_QUERY_PROMPT, dim=256)
    ok_mrl = small.shape[1] == 256 and abs(np.linalg.norm(small, axis=1) - 1).max() < 1e-3
    print("MRL dim=256:", small.shape, "norms", np.round(np.linalg.norm(small, axis=1), 4))
    if not ok_mrl:
        fails.append("MRL truncation/renormalization broken")

    out["reranker"] = {"yes": rrk.true_id, "no": rrk.false_id, "prefix": len(rrk.prefix),
                       "suffix": len(rrk.suffix), "body_max": rrk.body_max}
    print(f"reranker: {out['reranker']}")
    if (rrk.true_id, rrk.false_id) != (9693, 2152):
        fails.append(f"token ids changed: {rrk.true_id}/{rrk.false_id}")

    pairs = [(QUERIES[0], DOCS[0]), (QUERIES[0], DOCS[1]), (QUERIES[1], DOCS[1])]
    scores = rrk.score(pairs)
    out["rerank_scores"] = scores
    print("reranker P(yes) batched:", [round(s, 5) for s in scores])
    if not (scores[0] > 0.9 and scores[1] < 0.1 and scores[2] > 0.9):
        fails.append(f"reranker ranking wrong: {scores}")

    one = [rrk.score([p], sort_by_length=False)[0] for p in pairs]
    ds = float(np.abs(np.array(scores) - np.array(one)).max())
    out["rerank_batch_invariance"] = ds
    print(f"batch-invariance (reranker) = {ds:.2e}")
    if ds > tol:
        fails.append(f"reranker batch-invariance broken: {ds:.2e}")

    if backend == "torch" and rrk._keep:
        keep = rrk._keep
        rrk._keep = None
        full = rrk.score(pairs, sort_by_length=False)
        rrk._keep = keep
        dk = float(np.abs(np.array(scores) - np.array(full)).max())
        out["logits_to_keep_delta"] = dk
        print(f"logits_to_keep=1 vs full logits = {dk:.2e}")
        if dk > 1e-3:
            fails.append(f"logits_to_keep changes scores: {dk:.2e}")
    return fails, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "mlx", "both"], default="both")
    args = ap.parse_args()
    backends = ["torch", "mlx"] if args.backend == "both" else [args.backend]

    fails: list[str] = []
    results = {}
    for b in backends:
        f, out = check(b)
        fails += [f"{b}: {x}" for x in f]
        results[b] = out

    if len(results) == 2:
        a, c = results["torch"], results["mlx"]
        de = float(np.abs(np.array(a["embed_sim"]) - np.array(c["embed_sim"])).max())
        dr = float(np.abs(np.array(a["rerank_scores"]) - np.array(c["rerank_scores"])).max())
        print(f"--- cross-backend: embedding {de:.2e}, reranker {dr:.2e}")
        if de > 5e-3 or dr > 5e-3:
            fails.append(f"torch vs mlx disagree: embed {de:.2e}, rerank {dr:.2e}")

    if fails:
        for f in fails:
            print("FAIL:", f)
        return 1
    print(f"RESULT: OK -- {', '.join(backends)} wrappers match the model cards "
          "(README refs + batch invariance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
