"""Dense side: a wrapper over the MLX/torch embedder + an on-disk vector cache.

The cache is mandatory: ingesting 3134 chunks on MLX fp16 takes ~13 min, and there are
many arms and ablations. The cache key is the text hash + a tag (backend, MRL size,
max_length), so different settings are not mixed.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from contextual_chunks.retrieval.models import EMB_QUERY_PROMPT, EmbedCache, Embedder, Reranker


def make_embedder(backend: str = "mlx", **kw):
    """`mlx` = Apple GPU (dev path and edge device), `torch` = CPU/MPS control."""
    if backend == "mlx":
        from contextual_chunks.retrieval.mlx_models import MlxEmbedder
        return MlxEmbedder(**kw)
    return Embedder(**kw)


def make_reranker(backend: str = "mlx", **kw):
    if backend == "mlx":
        from contextual_chunks.retrieval.mlx_models import MlxReranker
        return MlxReranker(**kw)
    return Reranker(**kw)


def embed_texts(embedder, texts: list[str], cache_path: Path | str | None = None,
                tag: str = "", batch_size: int | None = None, prompt: str = "",
                log=print) -> tuple[np.ndarray, dict]:
    """-> (vectors [n, dim] float32, stats). Computes only the cache misses."""
    cache = EmbedCache(str(cache_path) if cache_path else "", tag=tag)
    out = np.zeros((len(texts), embedder.out_dim), dtype=np.float32)
    todo, idx = [], []
    for i, t in enumerate(texts):
        v = cache.get(t)
        if v is None:
            todo.append(t)
            idx.append(i)
        else:
            out[i] = v
    t0 = time.perf_counter()
    if todo:
        vecs = embedder.encode(todo, batch_size=batch_size, prompt=prompt, sort_by_length=True)
        for k, i in enumerate(idx):
            out[i] = vecs[k]
            cache.put(texts[i], vecs[k])
        cache.save()
    sec = time.perf_counter() - t0
    info = {"n": len(texts), "cached": len(texts) - len(todo), "embedded": len(todo),
            "sec": round(sec, 1),
            "texts_per_sec": round(len(todo) / sec, 2) if sec > 0 and todo else None,
            "dim": int(out.shape[1])}
    log(f"  embedding: {info['embedded']} new + {info['cached']} from cache in {sec:.1f}s "
        f"({info['texts_per_sec'] or 0} texts/s)")
    return out, info


def query_vector(embedder, query: str, prompt: str | None = None) -> np.ndarray:
    """Query vector: the instruction is mandatory (the model is trained on instructions)."""
    v = embedder.encode([query], prompt=EMB_QUERY_PROMPT if prompt is None else prompt,
                        sort_by_length=False)
    return v[0]
