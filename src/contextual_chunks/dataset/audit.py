"""End-to-end audit of every pipeline stage.

Detects the stage from the record fields and runs every applicable invariant.
Exit code 1 if anything fails, so it can gate the pipeline in CI.

    .venv/bin/python -m contextual_chunks.dataset.audit chunks_unlabeled.jsonl
    .venv/bin/python -m contextual_chunks.dataset.audit dataset_ready_for_labeling.jsonl
    .venv/bin/python -m contextual_chunks.dataset.audit dataset_optimized_by_doc.jsonl
    .venv/bin/python -m contextual_chunks.dataset.audit labeled_dataset_metadata.jsonl
    .venv/bin/python -m contextual_chunks.dataset.audit train_sft_dataset.jsonl
    .venv/bin/python -m contextual_chunks.dataset.audit --all          # auto-discovers the stages
"""

import os
import re
import sys
import json
import argparse
import collections

from contextual_chunks import paths
from contextual_chunks.dataset.tokenizer import count_tokens

_WS = re.compile(r"\s+")
_SENTINEL = re.compile(r"^\|[\s\-:|]+\|$")
PROMPT_HEAD = "<document>\n"
PROMPT_MID = "\n</document>\nВот фрагмент, который нужно соотнести со всем документом:\n<chunk>\n"
PROMPT_TAIL = "\n</chunk>\nСформулируй краткий контекст (1-2 предложения), чтобы привязать этот фрагмент к документу в целом для улучшения поисковой выдачи. Ответь только кратким контекстом и ничем больше."

MAX_PROMPT_TOKENS = 12288
MAX_WINDOW_TOKENS = 8000


def norm(x):
    return _WS.sub(" ", x or "").strip()


def parse_prompt(prompt):
    """Recover (document, chunk) from a prompt built with the pipeline template."""
    if not prompt.startswith(PROMPT_HEAD) or PROMPT_TAIL not in prompt:
        return None, None
    body = prompt[len(PROMPT_HEAD):]
    doc, rest = body.split("\n</document>\n", 1)
    if PROMPT_MID not in prompt:
        return None, None
    chunk = rest.split("<chunk>\n", 1)[1].split("\n</chunk>", 1)[0]
    return doc, chunk


def collapsed_table(text, doc):
    """True when a many-pipe chunk is NOT a fragment of a document line.

    The legacy bug word-wrapped table rows and destroyed the newlines; such a
    chunk is a reflow of several rows and therefore appears inside no single
    document line.  A genuinely huge table row (or a token-split piece of one)
    is fine.
    """
    if "\n" in text or text.count("|") <= 10:
        return False
    t = text.strip()
    return not any(t in ln for ln in (doc or "").split("\n"))


def looks_like_table(text):
    return sum(1 for ln in text.split("\n") if ln.lstrip().startswith("|")) >= 2


class Report:
    def __init__(self, path):
        self.path = path
        self.rows = []
        self.failures = []

    def check(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        if not ok:
            self.failures.append(f"{name}: {detail}" if detail else name)

    def print(self):
        print(f"\n=== AUDIT: {self.path} ===")
        for name, ok, detail in self.rows:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name:<38} {detail}")


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(int(p * len(xs)), len(xs) - 1)] if xs else 0


def audit(path, verbose=True):
    if not os.path.exists(path):
        print(f"!! missing file: {path}")
        return None
    recs = []
    bad_json = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            bad_json += 1

    rep = Report(path)
    n = len(recs)
    rep.check("json parses", bad_json == 0, f"{bad_json} bad lines")
    if not n:
        rep.check("non-empty", False, "0 rows")
        rep.print()
        return rep

    fields = set(recs[0])

    # ---- stage detection -------------------------------------------------
    is_chunks = "full_document" in fields
    is_ready = "prompt" in fields and "full_document" not in fields
    is_labeled = is_ready and "generated_context" in fields
    is_sft = fields == {"id", "messages"} or ({"id", "messages"} <= fields)
    stage = ("sft" if is_sft else "labeled" if is_labeled else
             "ready" if is_ready else "chunks" if is_chunks else "unknown")
    rep.check("stage detected", stage != "unknown", stage)

    # ---- documents / chunks per stage ------------------------------------
    docs, chunks, doc_chunks, prompts = {}, [], collections.Counter(), {}
    if stage == "sft":
        for rec in recs:
            msgs = rec.get("messages") or []
            user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
            doc, chunk = parse_prompt(user) if user else (None, None)
            if doc is None:
                rep.check("prompt template parses", False, f"id={rec.get('id')}")
                break
            prompts[rec["id"]] = (user, asst, doc, chunk)
        else:
            rep.check("prompt template parses", True, f"{n} rows")
        rep.check("all rows have user+assistant",
                  all(len(p) == 4 and p[1] for p in prompts.values()),
                  f"{sum(1 for p in prompts.values() if not p[1])} empty assistant")
        chunks = [p[3] for p in prompts.values() if p[3]]
        docs = {i: p[2] for i, p in prompts.items() if p[2]}
    else:
        for rec in recs:
            doc = rec.get("full_document")
            chunk = rec.get("chunk")
            if doc is None and rec.get("prompt"):
                doc, chunk = parse_prompt(rec["prompt"])
                if doc is None and stage != "sft":
                    rep.check("prompt template parses", False, f"id={rec.get('id')}")
                    break
            docs[rec.get("id") or rec.get("chunk_id")] = doc
            chunks.append(chunk)
            doc_chunks[(rec.get("doc_id"), rec.get("window_index"))] += 1
        else:
            if stage != "chunks":
                rep.check("prompt template parses", True, f"{n} rows")

    # ---- invariants ------------------------------------------------------
    if stage != "sft":
        inside = sum(1 for rec in recs
                     if norm(parse_prompt(rec["prompt"])[1] if stage != "chunks" else rec["chunk"])
                     in norm(parse_prompt(rec["prompt"])[0] if stage != "chunks" else rec["full_document"]))
    else:
        inside = sum(1 for i, (u, a, d, c) in prompts.items() if norm(c) in norm(d))
    rep.check("chunk inside document", inside == n, f"{inside}/{n}")

    bad_tables = 0
    for rec in recs:
        if stage == "sft":
            break
        if stage == "chunks":
            chunk, doc = rec.get("chunk"), rec.get("full_document")
        else:
            doc, chunk = parse_prompt(rec.get("prompt", ""))
        if chunk and collapsed_table(chunk, doc):
            bad_tables += 1
    if stage != "sft":
        rep.check("no collapsed tables", bad_tables == 0, f"{bad_tables} rows")

    for marker in ("</document>", "<document>", "<chunk>", "</chunk>"):
        found = 0
        for rec in recs:
            strings = [rec.get(k) for k in ("chunk", "full_document", "generated_context")]
            if stage == "sft":
                # the user turn IS the prompt -- only the target is free text
                strings += [m.get("content") for m in rec.get("messages", [])
                            if m.get("role") != "user"]
            if any(s and marker in s for s in strings if isinstance(s, str)):
                found += 1
        rep.check(f"prompt marker {marker!r} absent", found == 0, f"{found} rows")

    pairs = collections.Counter()
    for rec in recs:
        if stage == "sft":
            u, a, d, c = prompts[rec["id"]]
            pairs[(norm(d), norm(c))] += 1
        elif stage == "chunks":
            pairs[(rec.get("window_hash") or norm(rec["full_document"]), norm(rec["chunk"]))] += 1
        else:
            d, c = parse_prompt(rec["prompt"])
            pairs[(rec.get("window_hash") or norm(d), norm(c))] += 1
    dups = sum(v - 1 for v in pairs.values() if v > 1)
    rep.check("no duplicate (doc,chunk)", dups == 0, f"{dups} duplicates")

    # ---- budget ----------------------------------------------------------
    if stage == "sft":
        pts = [count_tokens(u) for u, a, d, c in prompts.values()]
        ws = [count_tokens(d) for u, a, d, c in prompts.values()]
    elif stage == "chunks":
        pts = [count_tokens(rec["prompt"]) if rec.get("prompt") else 0 for rec in recs]
        ws = [count_tokens(rec["full_document"]) for rec in recs]
    else:
        pts = [count_tokens(rec["prompt"]) for rec in recs]
        ws = [count_tokens(parse_prompt(rec["prompt"])[0]) for rec in recs]
    if stage == "chunks":
        overs = 0
        pt_stats = None
    else:
        overs = sum(1 for p in pts if p > MAX_PROMPT_TOKENS)
        pt_stats = f"med {pctl(pts,.5)} p95 {pctl(pts,.95)} max {max(pts)}"
    rep.check("sample fits token budget", overs == 0,
              f"{overs} rows > {MAX_PROMPT_TOKENS} tok" if overs else pt_stats or "n/a")
    wover = sum(1 for w in ws if w > MAX_WINDOW_TOKENS)
    rep.check("window within cap", wover == 0,
              f"{wover} windows > {MAX_WINDOW_TOKENS} tok, max {max(ws)}")

    if "full_document" in fields:
        drift = sum(1 for rec in recs if rec.get("window_tokens") not in (None, count_tokens(rec["full_document"])))
        rep.check("recorded window_tokens correct", drift == 0, f"{drift} rows")
        drift = sum(1 for rec in recs if rec.get("chunk_tokens") not in (None, count_tokens(rec["chunk"])))
        rep.check("recorded chunk_tokens correct", drift == 0, f"{drift} rows")
    else:
        rec = recs[0]
        if "window_tokens" in fields:
            drift = sum(1 for r in recs if r.get("window_tokens") not in (None, count_tokens(parse_prompt(r["prompt"])[0])))
            rep.check("recorded window_tokens correct", drift == 0, f"{drift} rows")
        if "chunk_tokens" in fields:
            drift = sum(1 for r in recs if r.get("chunk_tokens") not in (None, count_tokens(r["chunk"])))
            rep.check("recorded chunk_tokens correct", drift == 0, f"{drift} rows")
    if "prompt_tokens" in fields:
        drift = sum(1 for rec in recs if rec.get("prompt_tokens") not in (None, count_tokens(rec["prompt"])))
        rep.check("recorded prompt_tokens correct", drift == 0, f"{drift} rows")

    # ---- document structure / cache locality -----------------------------
    # NB: the labelled file is written concurrently, so its row order is
    # interleaved by design; only the pre-labelling stages must be contiguous.
    if "doc_id" in fields and stage != "labeled":
        blocks, cur = [], None
        for rec in recs:
            if rec["doc_id"] != cur:
                blocks.append([rec["doc_id"], []])
                cur = rec["doc_id"]
            blocks[-1][1].append(rec.get("window_index"))
        per_doc = collections.Counter(b[0] for b in blocks)
        split = [d for d, c in per_doc.items() if c > 1]
        rep.check("documents contiguous (cache)", not split,
                  f"{len(split)} docs split" if split else f"{len(blocks)} blocks")
        if "window_index" in fields:
            asc = all(w == sorted(w) for _, w in blocks)
            rep.check("windows ascending inside doc", asc)
        windows = {(r["doc_id"], r.get("window_index")) for r in recs}
        rep.check("cache prefix report", True,
                  f"{len(windows)} window prefixes, {len(recs) - len(windows)} potential cache hits "
                  f"({(1 - len(windows)/len(recs))*100:.1f}%)")
        if "window_index" in fields:
            tot_bad = sum(1 for rec in recs
                          if rec.get("window_total") is not None
                          and not (0 <= rec["window_index"] < rec["window_total"]))
            rep.check("window_index in range", tot_bad == 0, f"{tot_bad} rows")
    if "doc_hash" in fields and "doc_id" in fields:
        mismatch = 0
        by_doc = {}
        for rec in recs:
            if rec.get("doc_hash"):
                by_doc.setdefault(rec["doc_id"], set()).add(rec["doc_hash"])
        mismatch = sum(1 for v in by_doc.values() if len(v) > 1)
        rep.check("doc_hash stable per doc_id", mismatch == 0, f"{mismatch} docs")
        hashes = collections.Counter(rec.get("doc_hash") for rec in recs if rec.get("doc_hash"))
        rep.check("no cross-doc content duplication", len(hashes) == len(
            {rec["doc_id"] for rec in recs}), f"{len(hashes)} hashes / {len({r['doc_id'] for r in recs})} docs")

    # ---- labelling-specific ---------------------------------------------
    if stage == "labeled":
        empty = sum(1 for rec in recs if not (rec.get("generated_context") or "").strip())
        rep.check("contexts non-empty", empty == 0, f"{empty} rows")
        truncated = sum(1 for rec in recs if rec.get("finish_reason") == "length")
        rep.check("no truncated contexts", truncated == 0, f"{truncated} rows finish_reason=length")
        reason = sum(1 for rec in recs if rec.get("usage_reasoning_tokens"))
        rep.check("thinking disabled (usage_reasoning_tokens=0)", reason == 0, f"{reason} rows")
        if "usage_cached_tokens" in fields:
            tot = sum(rec.get("usage_prompt_tokens") or 0 for rec in recs)
            cac = sum(rec.get("usage_cached_tokens") or 0 for rec in recs)
            rep.check("cache telemetry present", tot > 0, f"{cac}/{tot} cached ({cac/tot*100:.1f}%)")
        costs = [rec.get("cost_usd") or 0 for rec in recs]
        rep.check("cost telemetry present", sum(1 for c in costs if c) > 0,
                  f"total ${sum(costs):.4f}")
        lens = [len(rec["generated_context"]) for rec in recs]
        rep.check("context length sane", max(lens) < 2000 if lens else False,
                  f"chars med {pctl(lens,.5)} p95 {pctl(lens,.95)} max {max(lens)}")
    if stage == "sft":
        bad = sum(1 for u, a, d, c in prompts.values() if not (a or "").strip())
        rep.check("assistant targets non-empty", bad == 0, f"{bad} rows")

    if verbose:
        rep.print()
    return rep


def discover():
    return [p for p in paths.STAGES if os.path.exists(p)]


def stage_ids(path):
    ids = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        ids.append(rec.get("id") or rec.get("chunk_id"))
    return ids


def cross_check(paths):
    """Every stage must describe exactly the same rows (same ids)."""
    if len(paths) < 2:
        return []
    base_path = paths[0]
    base = set(stage_ids(base_path))
    failures = []
    print(f"\n=== CROSS-STAGE CONSISTENCY (base: {os.path.basename(base_path)}) ===")
    for path in paths[1:]:
        other = set(stage_ids(path))
        missing, extra = base - other, other - base
        ok = not missing and not extra
        print(f"  [{'PASS' if ok else 'FAIL'}] {os.path.basename(path):<38} "
              f"{len(other)} ids, missing {len(missing)}, extra {len(extra)}")
        if not ok:
            failures.append(f"{os.path.basename(path)}: {len(missing)} missing / {len(extra)} extra ids")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="dataset file to audit")
    ap.add_argument("--all", action="store_true", help="audit every stage present")
    a = ap.parse_args()

    paths = discover() if a.all or not a.path else [a.path]
    if not paths:
        sys.exit("no dataset files found")
    reports = [audit(p) for p in paths]
    failed = [r for r in reports if r and r.failures]
    cross = cross_check(paths)
    print("\n" + "=" * 60)
    if failed or cross:
        print(f"RESULT: FAIL -- {sum(len(r.failures) for r in failed) + len(cross)} failing checks")
        for r in failed:
            for f in r.failures:
                print(f"  {os.path.basename(r.path)}: {f}")
        for f in cross:
            print(f"  {f}")
        sys.exit(1)
    print(f"RESULT: OK -- {len(reports)} files, all invariants hold")


if __name__ == "__main__":
    main()
