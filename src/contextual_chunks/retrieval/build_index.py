"""Build the indexes for the article arms.

    .venv/bin/python -m contextual_chunks.retrieval.build_index --arms A B
    .venv/bin/python -m contextual_chunks.retrieval.build_index --arms A B D --fresh

There are physically two indexes (so the embedding is not run twice):
    A/D -> index_text=chunk,  B/C -> index_text=context_chunk
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from contextual_chunks import paths
from contextual_chunks.retrieval import corpus as corpus_mod
from contextual_chunks.retrieval import index as index_mod
from contextual_chunks.retrieval.config import (BM25_BACKENDS, INDEX_ROOT, ARMS,
                                                 RetrievalConfig)

RESULTS = Path(paths.RESULTS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["A", "B"], help="which arms are needed (A/B/C/D)")
    ap.add_argument("--backend", default="mlx", choices=["mlx", "torch"])
    ap.add_argument("--limit", type=int, default=None, help="truncate the corpus (debug)")
    ap.add_argument("--tag", default="", help="index directory suffix (debug: smoke300)")
    ap.add_argument("--fresh", action="store_true", help="rebuild even if the index exists")
    ap.add_argument("--context-source", default=None, choices=["teacher", "student"],
                    help="override the context source (default: from the arm)")
    ap.add_argument("--bm25", default=None, choices=list(BM25_BACKENDS),
                    help="override the BM25 backend (rank_bm25 | fastembed | none)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    reports = []
    seen: set[str] = set()
    cache: dict[str, list] = {}
    for a in args.arms:
        cfg: RetrievalConfig = ARMS[a.upper()]
        if args.bm25:
            cfg = cfg.with_(bm25=args.bm25)
        src = args.context_source or cfg.context_source
        if src not in cache:
            cache[src] = corpus_mod.load(limit=args.limit, context_source=src)
            print(f"corpus ({src}): {corpus_mod.stats(cache[src])}")
        chunks = cache[src]
        if cfg.index_fingerprint() in seen:
            print(f"arm {a}: same index (index_text={cfg.index_text}), skipping")
            continue
        seen.add(cfg.index_fingerprint())
        if index_mod.exists(cfg, args.backend, args.tag) and not args.fresh:
            meta = index_mod.read_meta(cfg, args.backend, args.tag)
            print(f"arm {a}: index already exists ({meta['points_count']} points), skipping "
                  f"(rebuild needed: --fresh)")
            reports.append(meta)
            continue
        print(f"arm {a} ({cfg.name}): index_text={cfg.index_text}, bm25={cfg.bm25}, "
              f"fp={cfg.index_fingerprint()}")
        t0 = time.perf_counter()
        meta = index_mod.build(cfg, chunks, backend=args.backend,
                               batch_size=args.batch_size, recreate=True, tag=args.tag)
        meta["arm"] = a
        meta["build_wall_sec"] = round(time.perf_counter() - t0, 1)
        reports.append(meta)

    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"index_report{('_' + args.tag) if args.tag else ''}.json"
    (RESULTS / name).write_text(json.dumps(reports, ensure_ascii=False, indent=2))
    print(f"\nreport: artifacts/results/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
