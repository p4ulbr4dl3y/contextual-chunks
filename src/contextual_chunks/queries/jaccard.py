"""Statistics of adjacent-chunk overlap (for section 3 of the paper).

Computes the median token Jaccard between adjacent chunks of the same document
and writes the result to artifacts/results/corpus_jaccard.json.

Usage:
    .venv/bin/python -m contextual_chunks.queries.jaccard
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from contextual_chunks import paths
from contextual_chunks.retrieval.corpus import load

OUT = Path(paths.RESULTS) / "corpus_jaccard.json"

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def main() -> int:
    chunks = load(context_source="teacher")
    by_doc: dict[str, list] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)

    values: list[float] = []
    for doc_chunks in by_doc.values():
        doc_chunks.sort(key=lambda c: c.idx)
        for left, right in zip(doc_chunks, doc_chunks[1:], strict=False):
            values.append(jaccard(set(left.text("chunk").split()), set(right.text("chunk").split())))

    report = {
        "chunks": len(chunks),
        "documents": len(by_doc),
        "adjacent_pairs": len(values),
        "jaccard_median": round(statistics.median(values), 4),
        "jaccard_mean": round(statistics.fmean(values), 4),
        "jaccard_p90": round(sorted(values)[int(0.9 * (len(values) - 1))], 4),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    print(f"-> {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
