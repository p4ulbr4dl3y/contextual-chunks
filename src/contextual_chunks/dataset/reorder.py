"""Reorder `dataset_ready_for_labeling.jsonl` -> `dataset_optimized_by_doc.jsonl`.

Goal: 100% prompt-cache locality for the teacher.  Prefix caching only pays off
if consecutive requests share the same `<document>` prefix, so every row of a
document must be emitted back-to-back, windows in order.

The reordering is verified, not assumed: the script asserts that each document
forms exactly one contiguous block and that `window_index` is non-decreasing
inside it (`--check` re-verifies an already reordered file without rewriting).

Usage:
    .venv/bin/python -m contextual_chunks.dataset.reorder
    .venv/bin/python -m contextual_chunks.dataset.reorder --check
"""

import sys
import json
import argparse
import collections

from contextual_chunks import paths


def order_key(rec):
    return (rec["doc_id"], rec.get("window_index", 0), rec.get("chunk_index", 0))


def check(rows, label):
    """Assert one contiguous, window-ascending block per document."""
    blocks = []
    for rec in rows:
        if not blocks or blocks[-1][0] != rec["doc_id"]:
            blocks.append([rec["doc_id"], []])
        blocks[-1][1].append(rec.get("window_index", 0))

    seen = collections.Counter(b[0] for b in blocks)
    for doc, windows in blocks:
        if windows != sorted(windows):
            raise AssertionError(f"{label}: window_index not ascending inside {doc}")
    for doc, n in seen.items():
        if n > 1:
            raise AssertionError(f"{label}: document {doc} split into {n} blocks")
    return len(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=paths.READY)
    ap.add_argument("--out", dest="out", default=paths.OPTIMIZED)
    ap.add_argument("--check", action="store_true",
                    help="only verify an existing dataset_optimized_by_doc.jsonl")
    a = ap.parse_args()

    if a.check:
        rows = [json.loads(l) for l in open(a.out, encoding="utf-8")]
        blocks = check(rows, a.out)
        print(f"OK: {len(rows)} rows, {blocks} contiguous document blocks in {a.out}")
        return

    docs = collections.defaultdict(list)
    for line in open(a.inp, encoding="utf-8"):
        rec = json.loads(line)
        docs[rec["doc_id"]].append(rec)

    total = 0
    windows = set()
    with open(a.out, "w", encoding="utf-8") as out:
        for doc_id in sorted(docs):
            chunks = sorted(docs[doc_id], key=order_key)
            for idx, rec in enumerate(chunks):
                rec = dict(rec)
                rec["doc_chunk_index"] = idx
                rec["doc_total_chunks"] = len(chunks)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
                windows.add((rec["doc_id"], rec.get("window_index", 0)))

    rows = [json.loads(l) for l in open(a.out, encoding="utf-8")]
    blocks = check(rows, a.out)

    print("=== REORDER FOR CACHE ===")
    print(f"rows:                   {total}")
    print(f"documents:              {len(docs)}")
    print(f"contiguous doc blocks:  {blocks} (== documents)")
    print(f"distinct window prefixes: {len(windows)}")
    print(f"  -> expected cache misses in labelling: {len(windows)} / {total} "
          f"({len(windows) / total * 100:.1f}%)")
    print(f"output:                 {a.out}")


if __name__ == "__main__":
    main()
