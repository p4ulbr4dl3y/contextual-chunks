"""Context generation with the student model (MLX bf16) reusing the KV cache.

Chunks of one window share a prompt prefix (98% of characters, median 7 chunks per
window), so prefill runs once per window, and for the other chunks only the tail is
computed. Without the cache, ingest of the whole corpus is ~8.5 h (prefill of 24.3M
tokens); with the cache it is ~2 h.

    .venv/bin/python -m contextual_chunks.latency.gen_student_context \
        --verify --out artifacts/results/student_context.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from contextual_chunks import paths

DATA = Path(paths.LABELED)
MODEL = Path(paths.MLX_CACHE) / "qwen3-0.6b-contextual-chunks-bfloat16"
OUT = Path(paths.STUDENT_CONTEXT)


def common_prefix_len(seqs: list[list[int]]) -> int:
    if not seqs:
        return 0
    n = min(len(s) for s in seqs)
    i = 0
    while i < n and all(s[i] == seqs[0][i] for s in seqs):
        i += 1
    return i


CLOSE_THINK = "<" + "/" + "think" + ">"   # assemble it so we do not catch an invisible space in the literal


def strip_think(text: str) -> str:
    """Qwen3 generates ` thinking...` before the answer; the teacher does not have it."""
    if CLOSE_THINK in text:
        text = text.split(CLOSE_THINK, 1)[1]
    return text.strip()


def build_windows(rows: list[dict]) -> list[list[dict]]:
    by = defaultdict(list)
    for r in rows:
        by[r["window_hash"]].append(r)
    out = []
    for _wh, group in by.items():
        group.sort(key=lambda r: r["chunk_index"])
        out.append(group)
    out.sort(key=lambda g: (g[0]["doc_id"], g[0]["window_hash"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--windows", type=int, default=0, help="limit the number of windows (debug)")
    ap.add_argument("--limit", type=int, default=0, help="limit the number of chunks (debug)")
    ap.add_argument("--verify", action="store_true", help="compare against generation without cache")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already written to --out (append)")
    args = ap.parse_args()

    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    data_text = args.data.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in data_text.splitlines() if line.strip()]
    done_ids: set[str] = set()
    if args.resume and args.out.exists():
        out_text = args.out.read_text(encoding="utf-8")
        done_ids = {json.loads(line)["id"] for line in out_text.splitlines() if line.strip()}
        print(f"resume: {len(done_ids)} chunks already done")
        rows = [r for r in rows if r["id"] not in done_ids]
    windows = build_windows(rows)
    if args.windows:
        windows = windows[: args.windows]
    if args.limit:
        windows = [g[: args.limit] for g in windows]
    n_chunks = sum(len(g) for g in windows)
    print(f"windows {len(windows)}, chunks {n_chunks}, model {args.model}")

    t0 = time.perf_counter()
    model, tok = load(str(args.model))
    print(f"load {time.perf_counter() - t0:.2f}s, chat_template: "
          f"{hasattr(tok, 'apply_chat_template')}")

    def ids_of(prompt: str) -> list[int]:
        msgs = [{"role": "user", "content": prompt}]
        if hasattr(tok, "apply_chat_template"):
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        return tok.encode(prompt)

    def gen(ids: list[int], cache=None, max_tokens: int | None = None) -> tuple[str, int, float, float]:
        t = time.perf_counter()
        ttft = None
        parts: list[str] = []
        n = 0
        for resp in stream_generate(model, tok, ids, max_tokens=max_tokens or args.max_tokens,
                                    prompt_cache=cache):
            if ttft is None:
                ttft = time.perf_counter() - t
            parts.append(resp.text)          # resp.text is the step delta, not accumulated text
            n = resp.generation_tokens
        return "".join(parts), n, (ttft or 0.0), time.perf_counter() - t

    def run_window(group: list[dict], all_ids: list[list[int]], cache) -> list[tuple]:
        """One prefill per window, then only tails. Returns (text, n, ttft, sec, fed)."""
        pref = common_prefix_len(all_ids)
        cached_len = 0
        out = []
        for ids in all_ids:
            if cached_len > pref:
                trim_prompt_cache(cache, cached_len - pref)
                cached_len = pref
            feed = ids[cached_len:]
            text, n, ttft, sec = gen(feed, cache=cache)
            cached_len += len(feed) + n
            out.append((strip_think(text), n, ttft, sec, len(feed)))
        return out

    if args.verify:
        print("\n=== verification: with cache vs without cache (greedy, texts must match) ===")
        group = windows[0][:3]
        all_ids = [ids_of(r["prompt"]) for r in group]
        pref = common_prefix_len(all_ids)
        print(f"window of {len(group)} chunks, common prefix {pref} tokens")
        cached = run_window(group, all_ids, make_prompt_cache(model))
        ok = 0
        for r, ids, (text, _n, ttft, _sec, fed) in zip(group, all_ids, cached, strict=False):
            plain, pn, _, _ = gen(ids)
            same = text.strip() == plain.strip()
            ok += same
            print(f"  chunk {r['chunk_index']:>4}: prefill {ttft:>5.2f}s, tail {fed:>5} tok "
                  f"(of {len(ids)}), matched plain: {same}")
            print(f"      cache: {text[:90]!r}")
            if not same:
                print(f"      plain: {plain[:90]!r}")
        print(f"  verification result: {ok}/{len(group)} matched")

    out_f = args.out.open("a" if args.resume else "w", encoding="utf-8")
    done = 0
    t_start = time.perf_counter()
    total_out = 0
    for _wi, group in enumerate(windows):
        all_ids = [ids_of(r["prompt"]) for r in group]
        pref = common_prefix_len(all_ids)
        results = run_window(group, all_ids, make_prompt_cache(model))
        for r, ids, (text, n, ttft, sec, _fed) in zip(group, all_ids, results, strict=False):
            rec = {"id": r["id"], "doc_id": r["doc_id"], "chunk_index": r["chunk_index"],
                   "window_hash": r["window_hash"], "category": r["category"],
                   "is_table": r.get("is_table"), "chunk_tokens": r.get("chunk_tokens"),
                   "prompt_tokens": len(ids), "prefix_tokens": pref,
                   "context": text, "out_tokens": n,
                   "prefill_sec": round(ttft, 3), "total_sec": round(sec, 3)}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            done += 1
            total_out += n
            if args.log_every and done % args.log_every == 0:
                el = time.perf_counter() - t_start
                print(f"  {done}/{n_chunks} chunks, {el:.0f}s, "
                      f"{el / done:.2f} s/chunk, remaining estimate {el / done * (n_chunks - done) / 60:.1f} min",
                      flush=True)
    out_f.close()
    el = time.perf_counter() - t_start
    print(f"\ndone: {done} chunks in {el:.0f}s ({el / max(1, done):.2f} s/chunk), "
          f"mean output {total_out / max(1, done):.1f} tokens")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
