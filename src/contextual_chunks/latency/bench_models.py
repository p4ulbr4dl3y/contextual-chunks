"""Inference bench for Qwen3-Embedding / Qwen3-Reranker: batch tuning and gain from optimizations.

We measure what the RAG pipeline needs:
* ingest: texts/s when embedding chunks -> seconds for the whole corpus of 3134 chunks;
* query: pairs/s when reranking -> milliseconds for top-50 candidates;
* ablations: sorted-by-length padding, fp16 vs fp32, MPS vs CPU, logits_to_keep.

Run:      .venv/bin/python -m contextual_chunks.latency.bench_models [--quick|--full]
Result:   artifacts/results/latency_models.md (+ .json)
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path


import numpy as np
import torch

from contextual_chunks import paths
from contextual_chunks.retrieval.models import Embedder, Reranker, pick_device, pick_dtype

try:
    import mlx.core as MLX
except Exception:  # not macOS
    MLX = None
CORPUS = Path(paths.CHUNKS)
OUT_MD = Path(paths.RESULTS) / "latency_models.md"
OUT_JSON = Path(paths.RESULTS) / "latency_models.json"
CORPUS_SIZE = 3134   # chunks in the v2 dataset
TOP_K = 50           # candidates to rerank in the RAG pipeline

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def peak_mem_mb(device: str) -> float:
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**20
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS: bytes; Linux: kilobytes
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def load_chunks(n: int) -> list[str]:
    if not CORPUS.exists():
        return [f"Тестовый фрагмент документа номер {i}. " * 60 for i in range(n)]
    texts = []
    with CORPUS.open() as f:
        for line in f:
            texts.append(json.loads(line)["chunk"])
            if len(texts) >= n:
                break
    return texts


def truncate_to(texts: list[str], tok, n_tokens: int) -> list[str]:
    out = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"][:n_tokens]
        out.append(tok.decode(ids))
    return out


def fmt_time(sec: float) -> str:
    if sec < 90:
        return f"{sec:.1f} s"
    return f"{sec/60:.1f} min"


def fmt_size(mb: float) -> str:
    return f"{mb/1024:.2f} GB" if mb > 1024 else f"{mb:.0f} MB"


def sync() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
    if MLX is not None:
        MLX.synchronize()


def measure(fn, warm_fn=None) -> float:
    """One warmup (on a small sample) + one measurement -> seconds."""
    (warm_fn or fn)()
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="minimal run")
    ap.add_argument("--full", action="store_true", help="wide grid of configs")
    ap.add_argument("--backend", choices=["torch", "mlx"], default="torch",
                    help="mlx = Apple GPU, bf16 (dev path; torch = production path)")
    args = ap.parse_args()
    quick, full = args.quick, args.full

    EmbCls, RrkCls = Embedder, Reranker
    if args.backend == "mlx":
        from contextual_chunks.retrieval.mlx_models import MlxEmbedder, MlxReranker
        EmbCls, RrkCls = MlxEmbedder, MlxReranker
        dev, dtype_name = "mlx-gpu", "float16 (converted MLX cache)"
    else:
        dev = pick_device()
        dtype_name = str(pick_dtype(dev)).replace("torch.", "")

    out = {"device": dev, "backend": args.backend, "dtype": dtype_name,
           "torch": torch.__version__, "threads": torch.get_num_threads(),
           "corpus_size": CORPUS_SIZE, "top_k": TOP_K, "rows": []}
    log(f"backend={args.backend} device={dev} dtype={dtype_name} torch={torch.__version__}")

    embedder = EmbCls()
    n_emb = 32 if quick else (128 if not full else 256)
    n_rrk = 16 if quick else (32 if not full else 64)
    corpus = load_chunks(n_emb if n_emb > n_rrk else n_rrk)
    emb_texts = truncate_to(corpus[:n_emb], embedder.tok, 600)
    rrk600 = truncate_to(corpus[:n_rrk], embedder.tok, 600)
    rrk1200 = truncate_to(corpus[:n_rrk], embedder.tok, 1200)
    queries = ["какой срок поставки товара по контракту", "требования к упаковке и маркировке",
               "порядок оплаты и штрафные санкции", "кто отвечает за приёмку работ"]
    pairs600 = [(queries[i % 4], d) for i, d in enumerate(rrk600)]
    pairs1200 = [(queries[i % 4], d) for i, d in enumerate(rrk1200)]
    log(f"bench: embed {n_emb}x600tok, rerank {n_rrk} pairs x600/1200tok")

    md = [f"# Inference bench Qwen3-0.6B (embedding + reranker), backend `{args.backend}`", "",
          f"- Device: `{dev}`, dtype `{dtype_name}`, torch {torch.__version__}, "
          f"threads {torch.get_num_threads()}.",
          f"- Reference case: corpus of {CORPUS_SIZE} chunks, top-{TOP_K} candidates to rerank.",
          f"- Embedding: {n_emb} real chunks of 600 tokens; rerank: {n_rrk} pairs; "
          "warmup is not counted.",
          "- Profile without KV cache (pure prefill), which is how ingest runs; for rerank this "
          "is an upper bound on cost.", ""]
    rows = out["rows"]

    def record(**kw) -> None:
        rows.append(kw)

    # ---------- 1. Embedding: batch x max_length ----------
    bs_grid = [16, 32] if quick else ([8, 16, 32, 64] if full else [8, 16, 32])
    ml_grid = [1024] if quick else [512, 1024]
    md += ["## 1. Chunk embedding (ingest)", "",
           "| batch | max_len | texts/s | ms/text | " + f"{CORPUS_SIZE} chunks | peak memory |",
           "|---:|---:|---:|---:|---:|---:|"]
    for bs in bs_grid:
        for ml in ml_grid:
            fn = lambda: embedder.encode(emb_texts, batch_size=bs, prompt="", max_length=ml)
            warm = lambda: embedder.encode(emb_texts[:8], batch_size=bs, prompt="", max_length=ml)
            sec = measure(fn, warm)
            rate = n_emb / sec
            record(kind="embed", batch=bs, max_len=ml, sec=sec, rate=rate, unit="texts/s")
            md.append(f"| {bs} | {ml} | {rate:.1f} | {1000*sec/n_emb:.1f} | "
                      f"{fmt_time(sec*CORPUS_SIZE/n_emb)} | {fmt_size(peak_mem_mb(dev))} |")
            log(f"embed bs={bs} ml={ml}: {rate:.1f} texts/s ({1000*sec/n_emb:.1f} ms/text)")

    best_emb = max([r for r in rows if r["kind"] == "embed"], key=lambda r: r["rate"])
    log(f"best embed: bs={best_emb['batch']} ml={best_emb['max_len']} "
        f"{best_emb['rate']:.1f} texts/s")

    # ---------- 2. Embedder ablations (real mix of lengths) ----------
    # On equal 600-token texts padding does not bite; we take real dataset
    # chunks (85..1500 tokens), where the batching policy shows up.
    var_texts = corpus[:n_emb]
    var_lens = embedder.count_tokens(var_texts)
    md += ["", "### Embedder ablations, real mix of lengths", "",
           f"Texts: {n_emb} real chunks, length {min(var_lens)}..{max(var_lens)} tokens "
           f"(total {sum(var_lens)}).", "",
           "| batching policy | padding tokens / useful | texts/s | vs baseline |",
           "|---|---:|---:|---:|"]

    def padding_waste(lengths_: list[int], buckets: list[list[int]]) -> float:
        padded = sum(len(b) * max(lengths_[i] for i in b) for b in buckets)
        return padded / max(1, sum(lengths_))

    base = None
    policies = [
        ("sorted + token_budget 32768 (default)", 32768, 32, True),
        ("sorted + token_budget 8192", 8192, 32, True),
        ("sorted, no budget (bs=32)", None, 32, True),
        ("no sorting (bs=32)", None, 32, False),
        ("one by one (bs=1)", None, 1, True),
    ]
    from contextual_chunks.retrieval.models import bucket_indices as _buckets

    for name, tb, bs, sort in policies:
        def batches_for(bs=bs, tb=tb, sort=sort):
            if not sort:
                return [list(range(i, min(i + bs, len(var_texts))))
                        for i in range(0, len(var_texts), bs)]
            return _buckets(var_lens, tb or 10**12, bs)

        waste = padding_waste(var_lens, batches_for())
        fn = lambda: embedder.encode(var_texts, batch_size=bs, prompt="", max_length=1024,
                                     sort_by_length=sort, token_budget=tb)
        warm = lambda: embedder.encode(var_texts[:8], batch_size=bs, prompt="", max_length=1024,
                                      sort_by_length=sort, token_budget=tb)
        sec = measure(fn, warm)
        rate = n_emb / sec
        base = base or rate
        record(kind="embed_ablation", name=name, sec=sec, rate=rate, padding_waste=waste)
        md.append(f"| {name} | {waste:.2f}x | {rate:.1f} | {rate/base:.2f}x |")
        log(f"ablation embed {name}: {rate:.1f} texts/s, padding {waste:.2f}x ({rate/base:.2f}x)")

    if args.backend == "torch" and dev != "cpu":
        e32 = Embedder(device=dev, dtype=torch.float32)
        sec = measure(lambda: e32.encode(emb_texts, batch_size=32, prompt="", max_length=1024),
                      lambda: e32.encode(emb_texts[:8], batch_size=32, prompt="", max_length=1024))
        record(kind="embed_ablation", name="fp32 (vs fp16)", sec=sec, rate=n_emb/sec)
        md.append(f"| fp32 instead of fp16 | - | {n_emb/sec:.1f} | {(n_emb/sec)/base:.2f}x |")
        log(f"ablation embed fp32: {n_emb/sec:.1f} texts/s ({n_emb/sec/base:.2f}x)")
        del e32

    if args.backend == "torch" and dev == "mps":
        n_c = max(16, n_emb // 4)
        ecpu = Embedder(device="cpu")
        sec = measure(lambda: ecpu.encode(emb_texts[:n_c], batch_size=32, prompt="", max_length=1024),
                      lambda: ecpu.encode(emb_texts[:8], batch_size=32, prompt="", max_length=1024))
        rate = n_c / sec
        record(kind="embed_ablation", name=f"CPU fp32 ({n_c} texts)", sec=sec, rate=rate)
        md.append(f"| CPU fp32 instead of MPS ({n_c} texts) | - | {rate:.1f} | {rate/base:.2f}x |")
        log(f"ablation embed cpu: {rate:.1f} texts/s ({rate/base:.2f}x)")
        del ecpu

    if args.backend == "mlx":
        # the default in the MLX wrappers is fp16 (converted cache), we compare against bf16
        e16 = EmbCls(dtype="bfloat16")
        sec = measure(lambda: e16.encode(emb_texts, batch_size=32, prompt="", max_length=1024),
                      lambda: e16.encode(emb_texts[:8], batch_size=32, prompt="", max_length=1024))
        record(kind="embed_ablation", name="bf16 (vs fp16 default)", sec=sec, rate=n_emb/sec)
        md.append(f"| bf16 instead of fp16 (default) | - | {n_emb/sec:.1f} | {(n_emb/sec)/base:.2f}x |")
        log(f"ablation embed bf16: {n_emb/sec:.1f} texts/s ({(n_emb/sec)/base:.2f}x)")
        del e16

    # ---------- 3. Reranker ----------
    rrk = RrkCls()
    rrk_bs = [8, 16] if quick else ([4, 8, 16, 32] if full else [4, 8, 16])
    docs_grid = [600] if quick else [600, 1200]
    md += ["", "## 2. Reranker (query latency)", "",
           f"| batch | doc tokens | pairs/s | ms/pair | top-{TOP_K} candidates |",
           "|---:|---:|---:|---:|---:|"]
    for bs in rrk_bs:
        for doc_tok, pairs in ((600, pairs600), (1200, pairs1200)):
            if doc_tok not in docs_grid:
                continue
            fn = lambda: rrk.score(pairs, batch_size=bs)
            warm = lambda: rrk.score(pairs[:4], batch_size=bs)
            sec = measure(fn, warm)
            rate = n_rrk / sec
            record(kind="rerank", batch=bs, doc_tokens=doc_tok, sec=sec, rate=rate)
            md.append(f"| {bs} | {doc_tok} | {rate:.1f} | {1000*sec/n_rrk:.1f} | "
                      f"{fmt_time(sec*TOP_K/n_rrk)} |")
            log(f"rerank bs={bs} doc={doc_tok}: {rate:.1f} pairs/s ({1000*sec/n_rrk:.1f} ms/pair)")

    best_rrk = max([r for r in rows if r["kind"] == "rerank"], key=lambda r: r["rate"])
    log(f"best rerank: bs={best_rrk['batch']} doc={best_rrk['doc_tokens']} {best_rrk['rate']:.1f} pairs/s")

    md += ["", "### Reranker ablations (bs=16, doc=600)", "",
           "| variant | pairs/s | vs baseline |", "|---|---:|---:|"]
    r0 = measure(lambda: rrk.score(pairs600, batch_size=16), lambda: rrk.score(pairs600[:4], batch_size=16))
    base_r = n_rrk / r0
    md.append(f"| sorted-by-length ON (default) | {base_r:.1f} | 1.00x |")
    r1 = measure(lambda: rrk.score(pairs600, batch_size=16, sort_by_length=False),
                 lambda: rrk.score(pairs600[:4], batch_size=16))
    md.append(f"| sorted-by-length OFF | {n_rrk/r1:.1f} | {(n_rrk/r1)/base_r:.2f}x |")
    log(f"ablation rerank sorted OFF: {n_rrk/r1:.1f} pairs/s ({(n_rrk/r1)/base_r:.2f}x)")
    record(kind="rerank_ablation", name="sorted-off", sec=r1, rate=n_rrk/r1)
    if args.backend == "torch":
        keep = rrk._keep
        rrk._keep = None
        r2 = measure(lambda: rrk.score(pairs600, batch_size=16),
                     lambda: rrk.score(pairs600[:4], batch_size=16))
        rrk._keep = keep
        md.append(f"| without `logits_to_keep=1` | {n_rrk/r2:.1f} | {(n_rrk/r2)/base_r:.2f}x |")
        log(f"ablation rerank full logits: {n_rrk/r2:.1f} pairs/s ({(n_rrk/r2)/base_r:.2f}x)")
        record(kind="rerank_ablation", name="full-logits", sec=r2, rate=n_rrk/r2)
    else:
        md.append("| (MLX: logits are always only for the last position) | - | - |")

    # ---------- 4. Summary ----------
    ingest_sec = best_emb["sec"] * CORPUS_SIZE / n_emb
    query_sec = best_rrk["sec"] * TOP_K / n_rrk
    out |= {"best_embed": best_emb, "best_rerank": best_rrk,
            "ingest_sec": ingest_sec, "query_topk_sec": query_sec,
            "peak_mem_mb": peak_mem_mb(dev)}
    md += ["", "## 3. Result for the pipeline", "",
           f"- **Ingest** ({CORPUS_SIZE} chunks, embed bs={best_emb['batch']}, "
           f"max_len={best_emb['max_len']}): **{fmt_time(ingest_sec)}** "
           f"({CORPUS_SIZE/ingest_sec:.1f} chunks/s).",
           f"- **Query**: rerank top-{TOP_K} (bs={best_rrk['batch']}, "
           f"doc={best_rrk['doc_tokens']} tokens): **{1000*query_sec:.0f} ms**.",
           f"- Embedding of a single query (~15 tokens): ~{1000/best_emb['rate']*3:.0f} ms "
           "(lower bound, the small-batch overhead is not counted).",
           f"- Process peak memory: {fmt_size(peak_mem_mb(dev))}.", ""]

    suffix = "_mlx" if args.backend == "mlx" else ""
    out_md = OUT_MD.with_name(OUT_MD.stem + suffix + OUT_MD.suffix)
    out_json = OUT_JSON.with_name(OUT_JSON.stem + suffix + OUT_JSON.suffix)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n")
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    log(f"wrote {out_md.name} and {out_json.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
