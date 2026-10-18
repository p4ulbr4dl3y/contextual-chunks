"""Build `chunks_unlabeled.jsonl` -- the teacher-facing dataset.

Invariants enforced here (and re-checked by `audit_dataset.py`):
  * every chunk is a contiguous substring of the document window shown in the
    prompt (``chunk <= full_document``);
  * windows are capped in **tokens** (Qwen3 tokenizer), not characters;
  * markdown tables keep their line structure and are split BY ROWS with the
    header block repeated in every chunk (never word-wrapped);
  * duplicate documents (by `doc_hash`) and duplicate (window, chunk) pairs --
    identical prompts, possibly from near-duplicate documents -- are dropped and
    hashed (`doc_hash`, `window_hash`), so a group split cannot leak.

Usage:
    .venv/bin/python -m contextual_chunks.dataset.chunks                          # cap 8000 tokens
    .venv/bin/python -m contextual_chunks.dataset.chunks --cap-tokens 6000
    .venv/bin/python -m contextual_chunks.dataset.chunks --out chunks_unlabeled.jsonl
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import re

from contextual_chunks import paths
from contextual_chunks.dataset.prompts import (
    chunk_text,
    extract_document_text,
    is_junk_table,
    looks_like_table,
    normalize_text,
    table_parts,
)
from contextual_chunks.dataset.tokenizer import count_tokens, split_by_tokens

_WS = re.compile(r"\s+")

# Model budget: student ctx 32k, trainer max_length 16384.  A 8000-token window
# + ~1200-token chunk + 149-token template = ~10.3k tokens worst case, i.e. 37%
# headroom, and 56/87 documents still fit whole (the point of the task).
DEFAULT_CAP_TOKENS = 8000
DEFAULT_MAX_CHUNK_TOKENS = 1200
DEFAULT_MAX_TABLE_ROWS = 8
DEFAULT_MAX_TABLE_TOKENS = 1500
DEFAULT_MIN_CHUNK_TOKENS = 30
DEFAULT_MIN_WINDOW_TOKENS = 30

def _norm(x):
    return _WS.sub(" ", x).strip()

def content_hash(text):
    """Whitespace-insensitive content hash (documents and windows)."""
    return hashlib.sha1(_WS.sub(" ", text).strip().encode("utf-8")).hexdigest()[:16]

doc_hash = content_hash

def window_blocks(text, cap_tokens):
    """Blocks (paragraphs / table row-groups) each fitting `cap_tokens`."""
    blocks = []
    for para in (p.strip() for p in text.split("\n\n") if p.strip()):
        if looks_like_table(para):
            if is_junk_table(para):
                continue
            if count_tokens(para) <= cap_tokens:
                blocks.append(para)
                continue
            header, body = table_parts(para)
            header_text = "\n".join(header)
            header_tokens = count_tokens(header_text)
            cur, cur_tokens = [], header_tokens
            for row in body:
                row_tokens = count_tokens(row)
                if row_tokens > max(cap_tokens - header_tokens, 1):
                    if cur:
                        blocks.append("\n".join(header + cur))
                        cur, cur_tokens = [], header_tokens
                    for piece in split_by_tokens(row, max(cap_tokens - header_tokens, 1)):
                        blocks.append("\n".join(header + [piece]))
                    continue
                if cur and cur_tokens + row_tokens > cap_tokens:
                    blocks.append("\n".join(header + cur))
                    cur, cur_tokens = [], header_tokens
                cur.append(row)
                cur_tokens += row_tokens
            if cur:
                blocks.append("\n".join(header + cur))
        else:
            blocks.extend(split_by_tokens(para, cap_tokens))
    # exactness pass: the additive per-row estimate ignores the newline tokens
    # between rows, so a piece may end up a few tokens over the cap
    return [p for piece in blocks
            for p in (split_by_tokens(piece, cap_tokens)
                      if count_tokens(piece) > cap_tokens else [piece])]

def split_windows(text, cap_tokens, min_window_tokens):
    """Pack blocks into windows of <= cap_tokens (paragraph boundaries).

    The candidate window is measured *after* joining, so the `\n\n` separators
    count against the cap as well (no silent overshoot).
    """
    windows, cur = [], []
    for block in window_blocks(text, cap_tokens):
        candidate = "\n\n".join(cur + [block])
        if cur and count_tokens(candidate) > cap_tokens:
            windows.append("\n\n".join(cur))
            cur = [block]
        else:
            cur.append(block)
    if cur:
        windows.append("\n\n".join(cur))
    return [w for w in windows if count_tokens(w) >= min_window_tokens]

def build(raw_dir, out_file, cap_tokens, max_chunk_tokens, max_table_rows,
          max_table_tokens, min_chunk_tokens):
    files = sorted(glob.glob(os.path.join(raw_dir, "*")))
    stats = collections.Counter()
    seen_docs = {}
    seen_pairs = set()          # (window_hash, chunk) -> identical prompt
    seen_doc_chunks = set()     # (doc_hash, chunk)    -> same chunk in 2 windows
    rows_written = 0
    windows_tokens = []
    chunks_tokens = []
    dropped_docs = []

    with open(out_file, "w", encoding="utf-8") as out:
        for path in files:
            name = os.path.basename(path)
            text = normalize_text(extract_document_text(path) or "")
            if not text:
                stats["docs_empty"] += 1
                continue
            stats["docs_with_text"] += 1

            dh = doc_hash(text)
            if dh in seen_docs:
                stats["docs_duplicate"] += 1
                dropped_docs.append((name, seen_docs[dh]))
                continue
            seen_docs[dh] = name
            stats["docs_kept"] += 1

            windows = split_windows(text, cap_tokens, DEFAULT_MIN_WINDOW_TOKENS)
            if not windows:
                stats["docs_no_windows"] += 1
                continue

            n_windows = len(windows)
            if n_windows == 1:
                stats["docs_single_window"] += 1

            for w_idx, window in enumerate(windows):
                windows_tokens.append(count_tokens(window))
                chunks = chunk_text(
                    window,
                    max_tokens=max_chunk_tokens,
                    overlap_tokens=0,
                    max_table_rows=max_table_rows,
                    max_table_tokens=max_table_tokens,
                    min_tokens=min_chunk_tokens,
                )
                # dedupe inside the window (identical text => identical label)
                seen_chunks, unique = set(), []
                for c in chunks:
                    key = _norm(c)
                    if key and key not in seen_chunks:
                        seen_chunks.add(key)
                        unique.append(c)
                stats["chunks_before_global_dedup"] += len(unique)

                w_hash = content_hash(window)
                for c_idx, chunk in enumerate(unique):
                    # global dedup: identical (window, chunk) => identical prompt,
                    # even when it comes from a near-duplicate document
                    norm_chunk = _norm(chunk)
                    pair = (w_hash, norm_chunk)
                    doc_pair = (dh, norm_chunk)
                    if pair in seen_pairs or doc_pair in seen_doc_chunks:
                        stats["chunks_duplicate_dropped"] += 1
                        continue
                    seen_pairs.add(pair)
                    seen_doc_chunks.add(doc_pair)

                    assert _norm(chunk) in _norm(window), (
                        "chunk must be contained in the document window "
                        f"({name} win{w_idx} chunk{c_idx})"
                    )
                    is_table = looks_like_table(chunk) or (
                        chunk.lstrip().startswith("|") and chunk.count("|") > 1)
                    rec = {
                        "doc_id": name,
                        "doc_hash": dh,
                        "window_hash": w_hash,
                        "chunk_id": f"{name}__win{w_idx}_chunk_{c_idx}",
                        "window_index": w_idx,
                        "window_total": n_windows,
                        "chunk_index": c_idx,
                        "is_table": is_table,
                        "window_tokens": count_tokens(window),
                        "chunk_tokens": count_tokens(chunk),
                        "full_document": window,
                        "chunk": chunk,
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    rows_written += 1
                    chunks_tokens.append(rec["chunk_tokens"])
                    stats["chunks_table" if is_table else "chunks_plain"] += 1

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(int(p * len(xs)), len(xs) - 1)] if xs else 0

    print("=== BUILD CHUNKS ===")
    print(f"raw files:              {len(files)}")
    print(f"docs with text:         {stats['docs_with_text']}")
    print(f"docs kept (unique):     {stats['docs_kept']}")
    print(f"docs dropped (dup):     {stats['docs_duplicate']}")
    print(f"docs empty/skipped:     {stats['docs_empty']}")
    print(f"docs = 1 window:        {stats['docs_single_window']}/{stats['docs_kept']}")
    print(f"chunks written:         {rows_written}")
    print(f"  table chunks:         {stats['chunks_table']}")
    print(f"  plain chunks:         {stats['chunks_plain']}")
    print(f"  dup chunks dropped:   {stats['chunks_duplicate_dropped']}")
    print(f"window tokens  med/p95/max: {pct(windows_tokens, .5)}/{pct(windows_tokens, .95)}/{max(windows_tokens, default=0)}")
    print(f"chunk  tokens  med/p95/max: {pct(chunks_tokens, .5)}/{pct(chunks_tokens, .95)}/{max(chunks_tokens, default=0)}")
    print(f"cap:                    {cap_tokens} tokens")
    print(f"output:                 {out_file}")
    if dropped_docs:
        print("\ndropped duplicate documents:")
        for dup, kept in dropped_docs:
            print(f"  {dup[:60]}  ==  {kept[:60]}")
    return rows_written

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=paths.RAW_DIR)
    ap.add_argument("--out", default=paths.CHUNKS)
    ap.add_argument("--cap-tokens", type=int, default=DEFAULT_CAP_TOKENS)
    ap.add_argument("--max-chunk-tokens", type=int, default=DEFAULT_MAX_CHUNK_TOKENS)
    ap.add_argument("--max-table-rows", type=int, default=DEFAULT_MAX_TABLE_ROWS)
    ap.add_argument("--max-table-tokens", type=int, default=DEFAULT_MAX_TABLE_TOKENS)
    ap.add_argument("--min-chunk-tokens", type=int, default=DEFAULT_MIN_CHUNK_TOKENS)
    a = ap.parse_args()
    build(a.raw_dir, a.out, a.cap_tokens, a.max_chunk_tokens, a.max_table_rows,
          a.max_table_tokens, a.min_chunk_tokens)

if __name__ == "__main__":
    main()
