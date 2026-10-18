"""BM25: fastembed (`Qdrant/bm25`, ONNX) and rank_bm25 as an independent implementation.

`avg_len` matters: fastembed assumes by default that a document is ~256 words, while our
chunks are several times longer; without calibration `b=0.75` skews length
normalization. So we compute avg_len over the corpus (the same at indexing and at query time).
"""

from __future__ import annotations

import re
from pathlib import Path

from qdrant_client import models

from contextual_chunks import paths

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def words(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class FastEmbedBM25:
    """Sparse vectors for Qdrant: TF saturation on the client side, IDF on the server."""

    DEFAULT_CACHE = Path(paths.CACHE) / "fastembed"

    def __init__(self, stem: bool = True, avg_len: float = 256.0, k: float = 1.2,
                 b: float = 0.75, cache_dir: str | None = None):
        from fastembed import SparseTextEmbedding

        self.avg_len, self.k, self.b = avg_len, k, b
        cache_dir = cache_dir or str(self.DEFAULT_CACHE)
        self.DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
        try:
            self.model = SparseTextEmbedding("Qdrant/bm25", cache_dir=cache_dir,
                                             language="russian", disable_stemmer=not stem,
                                             avg_len=avg_len, k=k, b=b)
        except Exception as e:  # the network/proxy ate the download
            raise RuntimeError(
                "failed to load Qdrant/bm25 (the model must be downloaded once: "
                f"check the proxy/VPN and the cache {cache_dir}). Error: {type(e).__name__}: {e}"
            ) from e

    @staticmethod
    def available() -> bool:
        """The model is already in the cache, no network needed (we look at files first, then load)."""
        try:
            import importlib.util
            if importlib.util.find_spec("fastembed") is None:
                return False
            if not any(FastEmbedBM25.DEFAULT_CACHE.rglob("*.onnx")):
                return False
            from fastembed import SparseTextEmbedding
            SparseTextEmbedding("Qdrant/bm25", cache_dir=str(FastEmbedBM25.DEFAULT_CACHE),
                                language="russian")
            return True
        except Exception:
            return False

    def embed_documents(self, texts: list[str]) -> list[models.SparseVector]:
        out = []
        for e in self.model.embed(texts):
            out.append(models.SparseVector(indices=[int(i) for i in e.indices],
                                           values=[float(v) for v in e.values]))
        return out

    def embed_query(self, query: str) -> models.SparseVector:
        e = next(iter(self.model.query_embed(query)))
        return models.SparseVector(indices=[int(i) for i in e.indices],
                                   values=[float(v) for v in e.values])


class RankBM25:
    """Classic BM25Okapi (in-memory), a control for Qdrant/fastembed.

    `stem=True` enables Russian Snowball (`snowballstemmer`, pure python).
    Lives in memory: 3134 chunks ~ 1-2 s, for an edge index this is also a fair option.
    """

    def __init__(self, texts: list[str], stem: bool = True):
        from rank_bm25 import BM25Okapi

        self.stem = stem
        self.stemmer = None
        if stem:
            import snowballstemmer
            self.stemmer = snowballstemmer.stemmer("russian")
        self.tokens = [self.tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokens)
        self.n = len(texts)

    def tokenize(self, text: str) -> list[str]:
        w = words(text)
        return self.stemmer.stemWords(w) if self.stemmer else w

    def search(self, query: str, limit: int = 64) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(self.tokenize(query))
        order = sorted(range(self.n), key=lambda i: -scores[i])[:limit]
        return [(i, float(scores[i])) for i in order]
