"""Paired bootstrap for comparing arms on per-query metrics.

Reads the JSON reports of eval_queries.py (field `arms.<ARM>.per_query`) and computes
a paired bootstrap: mean difference, 95% confidence interval and standard error for
recall@10, MRR@10 and nDCG@10.

Usage:
    .venv/bin/python -m contextual_chunks.queries.paired_stats \
        --a artifacts/results/retrieval_eval_strat60.json \
        --pairs F:C F:G E:B --out artifacts/results/paired_stats.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

from contextual_chunks import paths

ROOT = Path(paths.ROOT)

METRICS = ("recall@10", "mrr@10", "ndcg@10")


def paired(a: dict, b: dict, metric: str, n_boot: int = 20000, seed: int = 3407) -> dict:
    """Paired bootstrap over shared qids: mean difference a - b."""
    by_qid_b = {r["qid"]: r for r in b}
    diffs = [r[metric] - by_qid_b[r["qid"]][metric] for r in a if r["qid"] in by_qid_b]
    if not diffs:
        return {}
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(statistics.fmean(diffs[rng.randrange(n)] for _ in range(n)))
    means.sort()
    return {
        "n": n,
        "mean_diff": round(statistics.fmean(diffs), 4),
        "se": round(statistics.stdev(means), 4),
        "ci95_low": round(means[int(0.025 * (n_boot - 1))], 4),
        "ci95_high": round(means[int(0.975 * (n_boot - 1))], 4),
        "p_a_gt_b": round(sum(1 for m in means if m > 0) / n_boot, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="JSON report with per_query")
    ap.add_argument("--b", type=Path, default=None, help="second report (defaults to --a)")
    ap.add_argument("--pairs", nargs="+", required=True, help="pairs of the form ARM:ARM")
    ap.add_argument("--out", type=Path, default=Path(paths.RESULTS) / "paired_stats.json")
    args = ap.parse_args()

    rep_a = json.loads(args.a.read_text(encoding="utf-8"))
    rep_b = json.loads(args.b.read_text(encoding="utf-8")) if args.b else rep_a
    report: dict = {"source_a": args.a.name, "source_b": (args.b or args.a).name, "pairs": {}}
    for pair in args.pairs:
        arm_a, arm_b = pair.split(":")
        pa = rep_a["arms"].get(arm_a, {}).get("per_query")
        pb = rep_b["arms"].get(arm_b, {}).get("per_query")
        if not pa or not pb:
            print(f"no per_query for {pair}")
            continue
        report["pairs"][pair] = {m: paired(pa, pb, m) for m in METRICS}
        row = report["pairs"][pair]["recall@10"]
        print(f"{pair} recall@10: {row.get('mean_diff')} "
              f"[{row.get('ci95_low')}, {row.get('ci95_high')}] se={row.get('se')} n={row.get('n')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
