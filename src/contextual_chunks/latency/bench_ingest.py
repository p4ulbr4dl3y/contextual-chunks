"""Ingest latency of the student model (MLX bf16) on real dataset prompts.

The student prompt is `<document>window</document> + <chunk>` (median 8.5k tokens,
24.3M over the corpus). So the ingest cost is prefill(prompt) + decode(~50-100
tokens), and prefill dominates. Here we measure both and extrapolate to the corpus.

    .venv/bin/python -m contextual_chunks.latency.bench_ingest --chunks 6
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from contextual_chunks import paths

DATA = Path(paths.LABELED)
MODEL = Path(paths.MLX_CACHE) / "qwen3-0.6b-contextual-chunks-bfloat16"


def load_rows() -> list[dict]:
    text = DATA.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def stratified(rows: list[dict], n: int) -> list[dict]:
    """n chunks, evenly spread by prompt length (note: prefill is quadratic in length)."""
    s = sorted(rows, key=lambda r: r["prompt_tokens"])
    if n >= len(s):
        return s
    step = len(s) / n
    return [s[int((i + 0.5) * step)] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--corpus", type=int, default=0, help="chunks in the corpus for extrapolation")
    ap.add_argument("--out", type=Path, default=Path(paths.RESULTS)
                    / "ingest_student.json")
    args = ap.parse_args()

    from mlx_lm import load, stream_generate

    rows = load_rows()
    corpus_n = args.corpus or len(rows)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in rows)
    print(f"model {args.model}\nchunks in corpus {len(rows)}, total prompt_tokens "
          f"{total_prompt_tokens / 1e6:.2f}M")
    t0 = time.perf_counter()
    model, tok = load(str(args.model))
    load_sec = time.perf_counter() - t0

    out_rows = []
    for r in stratified(rows, args.chunks):
        prompt = r["prompt"]
        ids = tok.encode(prompt)
        t0 = time.perf_counter()
        n_out = 0
        ttft = None
        for resp in stream_generate(model, tok, ids, max_tokens=args.max_tokens):
            if ttft is None:
                ttft = time.perf_counter() - t0
            n_out = resp.generation_tokens
        total = time.perf_counter() - t0
        decode_sec = total - (ttft or 0)
        rec = {"chunk_index": r["chunk_index"], "doc_id": r["doc_id"],
               "prompt_tokens": len(ids), "window_tokens": r["window_tokens"],
               "out_tokens": n_out, "prefill_sec": round(ttft or 0, 3),
               "decode_sec": round(decode_sec, 3), "total_sec": round(total, 3),
               "prefill_tok_per_sec": round(len(ids) / (ttft or 1), 1),
               "decode_tok_per_sec": round(n_out / decode_sec, 1) if decode_sec > 0 else None,
               "context_chars": len(r.get("generated_context") or "")}
        out_rows.append(rec)
        print(f"  chunk {r['chunk_index']:>5} prompt {len(ids):>5} tok: prefill {rec['prefill_sec']:>6.2f}s "
              f"({rec['prefill_tok_per_sec']:>6.1f} tok/s), decode {rec['decode_sec']:.2f}s "
              f"({rec['decode_tok_per_sec']} tok/s), out {n_out} tok")

    prefill = [r["prefill_sec"] for r in out_rows]
    prompt_toks = [r["prompt_tokens"] for r in out_rows]
    # extrapolation: mean time per 1k prompt tokens * total corpus tokens
    sec_per_1k = statistics.fmean(p / (t / 1000)
                                  for p, t in zip(prefill, prompt_toks, strict=False))
    decode_per_chunk = statistics.fmean(r["decode_sec"] for r in out_rows)
    est_prefill = sec_per_1k * total_prompt_tokens / 1000
    est_total = est_prefill + decode_per_chunk * corpus_n
    summary = {
        "model": str(args.model), "load_sec": round(load_sec, 2), "chunks_sampled": len(out_rows),
        "corpus_chunks": corpus_n, "corpus_prompt_tokens": total_prompt_tokens,
        "prefill_sec_per_1k_tokens": round(sec_per_1k, 3),
        "decode_sec_per_chunk": round(decode_per_chunk, 3),
        "est_prefill_hours": round(est_prefill / 3600, 2),
        "est_total_hours": round(est_total / 3600, 2),
        "samples": out_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nload {load_sec:.2f}s | prefill {sec_per_1k:.3f} s / 1k tokens "
          f"| decode {decode_per_chunk:.2f} s/chunk")
    print(f"extrapolated to {corpus_n} chunks ({total_prompt_tokens / 1e6:.2f}M tokens): "
          f"prefill {est_prefill / 3600:.2f} h + decode {decode_per_chunk * corpus_n / 3600:.2f} h "
          f"= {est_total / 3600:.2f} h")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
