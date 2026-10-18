"""Search: dense + BM25 -> RRF (native or manual) -> [reranker] -> top-k.

Arms A/B are without the reranker, C/D with it. The reranker (if enabled) sees the same
text as the embedder and BM25 (a requirement from the doc2seek audit: "the same crumb").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from contextual_chunks.retrieval import corpus as corpus_mod
from contextual_chunks.retrieval.config import RetrievalConfig
from contextual_chunks.retrieval.dense import make_embedder, make_reranker, query_vector
from contextual_chunks.retrieval.fuse import manual_rrf, sorted_ids
from contextual_chunks.retrieval.index import COLLECTION, avg_word_len, open_client
from contextual_chunks.retrieval.sparse import FastEmbedBM25, RankBM25


@dataclass
class Hit:
    idx: int
    cid: str
    doc_id: str
    score: float
    text: str
    stage: str
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None
    is_table: bool = False


@dataclass
class Searcher:
    cfg: RetrievalConfig
    backend: str = "mlx"
    chunks: list[corpus_mod.Chunk] | None = None
    limit: int | None = None
    client: QdrantClient | None = None
    tag: str = ""
    log = print
    _embedder: object = field(default=None, repr=False)
    _reranker: object = field(default=None, repr=False)
    _fast_bm25: object = field(default=None, repr=False)
    _rank_bm25: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.chunks is None:
            self.chunks = corpus_mod.load(limit=self.limit,
                                          context_source=self.cfg.context_source)
        if self.client is None:
            self.client = open_client(self.cfg, self.backend, self.tag)
        self.texts = [c.text(self.cfg.index_text) for c in self.chunks]
        self.by_idx = {c.idx: c for c in self.chunks}

    # -- lazy loading of the heavy parts ---------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = make_embedder(self.backend, dim=self.cfg.embed_dim,
                                           max_length=self.cfg.embed_max_length,
                                           token_budget=self.cfg.token_budget)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = make_reranker(self.backend, max_length=self.cfg.embed_max_length)
        return self._reranker

    @property
    def fast_bm25(self):
        if self._fast_bm25 is None:
            self._fast_bm25 = FastEmbedBM25(stem=self.cfg.bm25_stem,
                                            avg_len=avg_word_len(self.texts))
        return self._fast_bm25

    @property
    def rank_bm25(self):
        if self._rank_bm25 is None:
            self._rank_bm25 = RankBM25(self.texts, stem=self.cfg.bm25_stem)
        return self._rank_bm25

    # -- individual channels ----------------------------------------------
    def _dense_top(self, qv, limit: int) -> list[int]:
        r = self.client.query_points(COLLECTION, query=qv.tolist(), using="dense",
                                     limit=limit, with_payload=False)
        return [int(p.id) for p in r.points]

    def _sparse_top_qdrant(self, query: str, limit: int) -> list[int]:
        r = self.client.query_points(COLLECTION, query=self.fast_bm25.embed_query(query),
                                     using="bm25", limit=limit, with_payload=False)
        return [int(p.id) for p in r.points]

    def _sparse_top_memory(self, query: str, limit: int) -> list[int]:
        return [i for i, _ in self.rank_bm25.search(query, limit)]

    def _fused_native(self, qv, query: str, limit: int) -> list[tuple[int, float]]:
        prefetch = [models.Prefetch(query=qv.tolist(), using="dense",
                                    limit=self.cfg.prefetch_limit)]
        if self.cfg.bm25 == "fastembed":
            prefetch.append(models.Prefetch(query=self.fast_bm25.embed_query(query),
                                            using="bm25", limit=self.cfg.prefetch_limit))
        rrf = models.Rrf(k=self.cfg.rrf_k,
                         weights=list(self.cfg.weights) if self.cfg.weights else None)
        r = self.client.query_points(COLLECTION, prefetch=prefetch,
                                     query=models.RrfQuery(rrf=rrf), limit=limit,
                                     with_payload=False)
        return [(int(p.id), float(p.score)) for p in r.points]

    def _fused_manual(self, qv, query: str, limit: int, dense_rank: dict, sparse_rank: dict):
        dense_top = self._dense_top(qv, self.cfg.prefetch_limit)
        rankings = [dense_top]
        weights = [self.cfg.weights[0]] if self.cfg.weights else None
        if self.cfg.bm25 == "fastembed":
            sparse_top = self._sparse_top_qdrant(query, self.cfg.prefetch_limit)
        elif self.cfg.bm25 == "rank_bm25":
            sparse_top = self._sparse_top_memory(query, self.cfg.prefetch_limit)
        else:
            sparse_top = []
        if sparse_top:
            rankings.append(sparse_top)
            weights = list(self.cfg.weights) if self.cfg.weights else None
        for rank, ids in (("dense", dense_top), ("bm25", sparse_top)):
            for pos, i in enumerate(ids, 1):
                (dense_rank if rank == "dense" else sparse_rank).setdefault(i, pos)
        scores = manual_rrf(rankings, k=self.cfg.rrf_k, weights=weights)
        ids = sorted_ids(scores, limit)
        return [(int(i), scores[i]) for i in ids]

    @property
    def effective_fusion(self) -> str:
        """`native` is possible only if the sparse vectors live in Qdrant (fastembed BM25);
        otherwise we compute RRF by hand, with an explicit k and weights."""
        return "native" if (self.cfg.fusion == "native" and self.cfg.bm25 == "fastembed") \
            else "manual"

    def close(self) -> None:
        """Local Qdrant holds the folder: one client per path per process."""
        if self.client is not None:
            self.client.close()
            self.client = None

    # -- main entry point -------------------------------------------------
    def search(self, query: str, top_k: int | None = None, rerank: bool | None = None,
               timings: dict | None = None) -> list[Hit]:
        t_all = time.perf_counter()
        top_k = top_k or self.cfg.top_k
        rerank = self.cfg.rerank if rerank is None else rerank
        pool = max(top_k, self.cfg.rerank_pool) if rerank else top_k
        T = timings if timings is not None else {}

        t0 = time.perf_counter()
        qv = query_vector(self.embedder, query)
        T["embed_ms"] = (time.perf_counter() - t0) * 1000

        dense_rank: dict[int, int] = {}
        sparse_rank: dict[int, int] = {}
        t0 = time.perf_counter()
        if self.effective_fusion == "native":
            fused = self._fused_native(qv, query, pool)
        else:
            fused = self._fused_manual(qv, query, pool, dense_rank, sparse_rank)
        T["fuse_ms"] = (time.perf_counter() - t0) * 1000

        hits = [Hit(idx=i, cid=self.by_idx[i].cid, doc_id=self.by_idx[i].doc_id, score=s,
                    text=self.texts[i], stage="fused", dense_rank=dense_rank.get(i),
                    bm25_rank=sparse_rank.get(i), is_table=self.by_idx[i].is_table)
                for i, s in fused]

        if rerank and hits:
            t0 = time.perf_counter()
            pairs = [(query, h.text) for h in hits]
            scores = self.reranker.score(pairs)
            for h, s in zip(hits, scores):
                h.rerank_score = float(s)
            hits.sort(key=lambda h: (-h.rerank_score, h.idx))
            for pos, h in enumerate(hits, 1):
                h.rerank_rank = pos
                h.score = h.rerank_score
                h.stage = "reranked"
            T["rerank_ms"] = (time.perf_counter() - t0) * 1000
        T["total_ms"] = (time.perf_counter() - t_all) * 1000
        T["n_hits"] = len(hits)
        return hits[:top_k]
