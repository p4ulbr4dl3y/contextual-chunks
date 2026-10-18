"""Merge two eval_queries.py reports into one (different arm sets, the same query set).

Usage:
    .venv/bin/python -m contextual_chunks.queries.merge_reports \
        --parts artifacts/results/retrieval_eval_strat60_sub.json \
                artifacts/results/retrieval_eval_strat60_rr.json \
        --out artifacts/results/retrieval_eval_strat60.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextual_chunks import paths

ROOT = Path(paths.ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    merged: dict | None = None
    for part in args.parts:
        rep = json.loads(part.read_text(encoding="utf-8"))
        if merged is None:
            merged = {k: v for k, v in rep.items() if k != "arms"}
            merged["arms"] = {}
        for arm, data in rep["arms"].items():
            if arm in merged["arms"]:
                raise SystemExit(f"arm {arm} appears twice")
            merged["arms"][arm] = data

    assert merged is not None
    counts = {len(v.get("per_query", [])) for v in merged["arms"].values()}
    if len(counts) != 1:
        raise SystemExit(f"arms were computed on a different number of queries: {counts}")
    merged["queries"] = counts.pop()
    merged["sources"] = [p.name for p in args.parts]

    args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"arms {len(merged['arms'])}: {', '.join(sorted(merged['arms']))} "
          f"on {merged['queries']} queries -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
