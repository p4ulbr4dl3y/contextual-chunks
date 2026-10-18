"""Index: local Qdrant (no docker) with named dense+sparse + meta.json.

One index = one config fingerprint (index_text + model + BM25 settings).
Local Qdrant mode (`path=`) is a deliberate choice: edge scenario, no containers,
everything lives in `index/qdrant/<backend>-<fp>/`.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

from contextual_chunks.retrieval import corpus as corpus_mod
from contextual_chunks.retrieval.config import RetrievalConfig
from contextual_chunks.retrieval.dense import embed_texts, make_embedder
from contextual_chunks.retrieval.sparse import FastEmbedBM25, words

COLLECTION = "chunks"
META_FILE = "contextual_chunks_index.json"   # not "meta.json": that is a Qdrant service file


def avg_word_len(texts: list[str]) -> float:
    return sum(len(words(t)) for t in texts) / max(1, len(texts))


def make_client(cfg: RetrievalConfig, backend: str = "mlx", recreate: bool = False,
                in_memory: bool = False, tag: str = "") -> tuple[QdrantClient, Path | None]:
    if in_memory:
        return QdrantClient(":memory:"), None
    path = cfg.qdrant_path(backend, tag)
    if recreate and path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path)), path


def open_client(cfg: RetrievalConfig, backend: str = "mlx", tag: str = "") -> QdrantClient:
    return QdrantClient(path=str(cfg.qdrant_path(backend, tag)))


def exists(cfg: RetrievalConfig, backend: str = "mlx", tag: str = "") -> bool:
    return (cfg.qdrant_path(backend, tag) / META_FILE).exists()


def read_meta(cfg: RetrievalConfig, backend: str = "mlx", tag: str = "") -> dict | None:
    p = cfg.qdrant_path(backend, tag) / META_FILE
    return json.loads(p.read_text()) if p.exists() else None


def build(cfg: RetrievalConfig, chunks: list[corpus_mod.Chunk], backend: str = "mlx",
          batch_size: int = 64, recreate: bool = True, tag: str = "", log=print) -> dict:
    """Builds the whole index: dense (+BM25 sparse) vectors and the payload."""
    client, path = make_client(cfg, backend, recreate, tag=tag)
    texts = [c.text(cfg.index_text) for c in chunks]
    t0 = time.perf_counter()

    embedder = make_embedder(backend, dim=cfg.embed_dim, max_length=cfg.embed_max_length,
                             token_budget=cfg.token_budget)
    t_load = time.perf_counter() - t0
    dense, emb_info = embed_texts(embedder, texts, cache_path=cfg.dense_cache_path(backend),
                                  tag=f"{backend}-d{cfg.embed_dim or 0}-ml{cfg.embed_max_length}",
                                  log=log)
    avg_len = avg_word_len(texts)

    sparse_cfg = {}
    if cfg.bm25 == "fastembed":
        sparse_cfg = {"bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF if cfg.bm25_idf_modifier else None)}
    client.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=int(dense.shape[1]),
                                                    distance=models.Distance.COSINE)},
        sparse_vectors_config=sparse_cfg)

    t1 = time.perf_counter()
    sparse_vecs = (FastEmbedBM25(stem=cfg.bm25_stem, avg_len=avg_len).embed_documents(texts)
                   if cfg.bm25 == "fastembed" else None)
    t_sparse = time.perf_counter() - t1

    points = []
    for i, c in enumerate(chunks):
        vec: dict = {"dense": dense[i].tolist()}
        if sparse_vecs is not None:
            vec["bm25"] = sparse_vecs[i]
        points.append(models.PointStruct(id=i, vector=vec, payload=c.to_payload(cfg.index_text)))
    for s in range(0, len(points), batch_size):
        client.upsert(COLLECTION, points=points[s:s + batch_size])

    total = time.perf_counter() - t0
    meta = {"cfg": cfg.to_dict(), "backend": backend, "tag": tag,
            "index_fingerprint": cfg.index_fingerprint(),
            "qdrant_path": str(path) if path else ":memory:", "chunks": len(chunks),
            "docs": len({c.doc_id for c in chunks}), "dim": int(dense.shape[1]),
            "bm25_avg_word_len": round(avg_len, 2), "bm25_index_sec": round(t_sparse, 1),
            "embed": emb_info, "model_load_sec": round(t_load, 2),
            "total_sec": round(total, 1), "sec_per_chunk": round(total / max(1, len(chunks)), 4),
            "points_count": client.count(COLLECTION).count}
    if path:
        (path / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    client.close()
    log(f"  index ready: {meta['points_count']} points, dim {meta['dim']}, "
        f"{meta['total_sec']}s ({meta['sec_per_chunk']*1000:.0f} ms/chunk), "
        f"bm25 avg_len {meta['bm25_avg_word_len']:.0f} words")
    return meta
