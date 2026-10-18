"""Drop duplicate prompts from the labelled dataset (keep the first occurrence).

Two rows can carry the *same prompt* (same document window + same chunk) when a
document has a near-duplicate twin that is not byte-identical as a whole (e.g. a
contract and its scan: same body, different requisites page).  Such rows are
exact duplicate training samples and would also leak across a train/val split,
so they are removed here and a `window_hash` is written for group splitting.

Inputs are rewritten in place (atomically, with `.bak` backups unless --no-backup):

    labeled_dataset_metadata.jsonl   <- source of truth for the dedup keys
    train_sft_dataset.jsonl          <- same ids, pruned to match

    .venv/bin/python -m contextual_chunks.dataset.dedup            # report + prune
    .venv/bin/python -m contextual_chunks.dataset.dedup --dry-run
"""

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import sys

from contextual_chunks import paths

_WS = re.compile(r"\s+")
P_HEAD = "<document>\n"


def norm(x):
    return _WS.sub(" ", x or "").strip()


def window_hash(text):
    return hashlib.sha1(norm(text).encode("utf-8")).hexdigest()[:16]


def parse_prompt(prompt):
    body = prompt[len(P_HEAD):]
    doc = body.split("\n</document>\n", 1)[0]
    chunk = body.split("<chunk>\n", 1)[1].split("\n</chunk>", 1)[0]
    return doc, chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=paths.LABELED)
    ap.add_argument("--sft", default=paths.SFT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.meta):
        sys.exit(f"missing {a.meta}")

    kept, dropped = [], collections.Counter()
    seen, seen_doc = set(), set()
    for line in open(a.meta, encoding="utf-8"):
        rec = json.loads(line)
        doc, chunk = parse_prompt(rec["prompt"])
        wh = window_hash(doc)
        n_chunk = norm(chunk)
        key, doc_key = (wh, n_chunk), (rec.get("doc_hash") or wh, n_chunk)
        if key in seen or doc_key in seen_doc:
            dropped["rows"] += 1
            dropped[rec["doc_id"]] += 1
            continue
        seen.add(key)
        seen_doc.add(doc_key)
        rec["window_hash"] = wh
        kept.append(rec)

    print("=== DEDUP LABELLED DATASET ===")
    print(f"rows in:   {len(kept) + dropped['rows']}")
    print(f"duplicates:{dropped['rows']}")
    print(f"rows out:  {len(kept)}")
    print(f"distinct window_hash groups: {len({r['window_hash'] for r in kept})}")
    if dropped["rows"]:
        print("docs affected:")
        for doc, n in dropped.most_common(12):
            if doc != "rows":
                print(f"  {doc[:56]:<56} {n}")
    if a.dry_run:
        return

    for path in (a.meta, a.sft):
        if not a.no_backup and os.path.exists(path):
            shutil.copy2(path, path + ".bak")
    with open(a.meta, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    wh_by_id = {r["id"]: r["window_hash"] for r in kept}
    meta_by_id = {r["id"]: r for r in kept}
    n_sft = 0
    with open(a.sft + ".tmp", "w", encoding="utf-8") as out, open(a.sft, encoding="utf-8") as src:
        for line in src:
            rec = json.loads(line)
            if rec["id"] not in wh_by_id:
                continue
            rec["window_hash"] = wh_by_id[rec["id"]]
            rec.setdefault("doc_hash", meta_by_id[rec["id"]].get("doc_hash"))
            rec.setdefault("category", meta_by_id[rec["id"]].get("category"))
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_sft += 1
    os.replace(a.sft + ".tmp", a.sft)
    print(f"sft rows out: {n_sft}")


if __name__ == "__main__":
    main()
