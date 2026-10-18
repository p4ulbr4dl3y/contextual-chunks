"""Validate `chunks_unlabeled.jsonl` -> `dataset_ready_for_labeling.jsonl`.

Guards (hard failures, exit code 1):
  * chunk is an exact slice of the document window shown in the prompt;
  * no collapsed tables (a one-line chunk with many `|` means the markdown
    structure was destroyed);
  * the window/chunk fit the token budget;
  * no `</document>` / `<chunk>` markers inside the payload (they are the
    prompt delimiters);
  * no duplicate (doc_hash, chunk) pairs.

The prompt template is the Anthropic-style Russian template the teacher was
validated with -- do not change it without re-labelling.

Usage:
    .venv/bin/python -m contextual_chunks.dataset.validate
    .venv/bin/python -m contextual_chunks.dataset.validate --in chunks_unlabeled.jsonl --out dataset_ready_for_labeling.jsonl
"""

import argparse
import collections
import json
import re
import sys

from contextual_chunks import paths
from contextual_chunks.dataset.tokenizer import count_tokens

_WS = re.compile(r"\s+")
DEFAULT_MAX_PROMPT_TOKENS = 12288  # trainer max_length 16384 -> 37% headroom

ANTHROPIC_PROMPT_TEMPLATE = """<document>
{document}
</document>
Вот фрагмент, который нужно соотнести со всем документом:
<chunk>
{chunk}
</chunk>
Сформулируй краткий контекст (1-2 предложения), чтобы привязать этот фрагмент к документу в целом для улучшения поисковой выдачи. Ответь только кратким контекстом и ничем больше."""


def _norm(x):
    return _WS.sub(" ", x).strip()


def prompt_tokens(prompt):
    return count_tokens(prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=paths.CHUNKS)
    ap.add_argument("--out", dest="out", default=paths.READY)
    ap.add_argument("--min-chunk-tokens", type=int, default=30)
    ap.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    a = ap.parse_args()

    stats = collections.Counter()
    categories = collections.Counter()
    seen_pairs = set()
    seen_doc_chunks = set()
    prompt_tokens_list = []
    missing_prompt = []

    with open(a.inp, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as out:
        for line in f:
            stats["read"] += 1
            rec = json.loads(line)
            doc = (rec.get("full_document") or "").strip()
            chunk = (rec.get("chunk") or "").strip()

            if not doc or not chunk:
                stats["bad_empty"] += 1
                continue

            markers = ("</document>", "<document>", "<chunk>", "</chunk>")
            if any(m in doc or m in chunk for m in markers):
                stats["bad_prompt_marker"] += 1
                continue

            # 1. containment: exact slice of the window (whitespace-insensitive)
            if _norm(chunk) not in _norm(doc):
                stats["bad_chunk_not_in_doc"] += 1
                missing_prompt.append(rec.get("chunk_id"))
                continue

            # 2. table structure intact: a one-line chunk stuffed with pipes is
            #    only acceptable if it is a verbatim line of the document (a
            #    genuinely huge table row); the legacy bug word-wrapped rows and
            #    produced a reflow that matches no single line.
            if ("\n" not in chunk and chunk.count("|") > 10
                    and not any(chunk in ln for ln in doc.split("\n"))):
                stats["bad_collapsed_table"] += 1
                continue

            # 3. budget
            if count_tokens(chunk) < a.min_chunk_tokens:
                stats["bad_too_short"] += 1
                continue
            pt = prompt_tokens(
                ANTHROPIC_PROMPT_TEMPLATE.format(document=doc, chunk=chunk)
            )
            if pt > a.max_prompt_tokens:
                stats["bad_too_long"] += 1
                continue

            # 4. dedupe: identical prompt (window+chunk) and same chunk twice
            #    inside one document are both useless training samples
            n_chunk = _norm(chunk)
            pair = (rec.get("window_hash") or _norm(doc), n_chunk)
            doc_pair = (rec.get("doc_hash") or _norm(doc), n_chunk)
            if pair in seen_pairs or doc_pair in seen_doc_chunks:
                stats["bad_duplicate"] += 1
                continue
            seen_pairs.add(pair)
            seen_doc_chunks.add(doc_pair)

            cat = rec["doc_id"].split("_")[0]
            categories[cat] += 1
            stats["valid"] += 1
            prompt_tokens_list.append(pt)

            out.write(json.dumps({
                "id": rec["chunk_id"],
                "category": cat,
                "doc_id": rec["doc_id"],
                "doc_hash": rec.get("doc_hash"),
                "window_index": rec.get("window_index"),
                "window_total": rec.get("window_total"),
                "chunk_index": rec.get("chunk_index"),
                "is_table": bool(rec.get("is_table")),
                "is_requisites": bool(re.search(r"(ИНН|КПП|ОГРН|Р/с|БИК|р/сч)", chunk)),
                "chunk_tokens": count_tokens(chunk),
                "window_tokens": count_tokens(doc),
                "prompt_tokens": pt,
                "prompt": ANTHROPIC_PROMPT_TEMPLATE.format(document=doc, chunk=chunk),
                "chunk": chunk,
            }, ensure_ascii=False) + "\n")

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(int(p * len(xs)), len(xs) - 1)] if xs else 0

    print("=== VALIDATE DATASET ===")
    print(f"input:                  {a.inp}")
    print(f"rows read:              {stats['read']}")
    print(f"  empty:                {stats['bad_empty']}")
    print(f"  chunk not in document:{stats['bad_chunk_not_in_doc']}")
    print(f"  collapsed table:      {stats['bad_collapsed_table']}")
    print(f"  too short (<{a.min_chunk_tokens} tok):{stats['bad_too_short']}")
    print(f"  prompt > {a.max_prompt_tokens} tok:   {stats['bad_too_long']}")
    print(f"  prompt marker inside: {stats['bad_prompt_marker']}")
    print(f"  duplicate content:    {stats['bad_duplicate']}")
    print(f"valid rows:             {stats['valid']}")
    if prompt_tokens_list:
        print(f"prompt tokens med/p95/max: {pct(prompt_tokens_list, .5)}/{pct(prompt_tokens_list, .95)}/{max(prompt_tokens_list)}")
    print("categories:")
    for cat, n in categories.most_common():
        print(f"  {cat:<22} {n}")
    print(f"output:                 {a.out}")

    bad = sum(v for k, v in stats.items() if k.startswith("bad_"))
    if bad:
        print(f"\nFAILED: {bad} rejected rows")
        for cid in missing_prompt[:10]:
            print(f"  e.g. {cid}")
        sys.exit(1)
    print("\nOK: every row satisfies all invariants.")


if __name__ == "__main__":
    main()
