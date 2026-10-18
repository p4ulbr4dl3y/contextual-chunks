"""Config of the retrieval harness: arm matrix A/B/C/D + all pinned knobs.

Arm matrix:

| arm | index         | query path              |
|-----|---------------|-------------------------|
| A   | chunk as is   | BM25 + dense + RRF      |
| B   | context+chunk | BM25 + dense + RRF      |
| C   | context+chunk | BM25 + dense + RRF + rer|
| D   | chunk as is   | BM25 + dense + RRF + rer|

There are physically two indexes: flat (A, D) and ctx (B, C), the difference is in
`index_text`, the reranker works only on the query side and does not affect the index.

IMPORTANT about RRF: Qdrant computes `1/((pos+1)/w + k - 1)`, i.e. its default `k=2`
is numerically equal to the textbook `1/(k+rank)` at k=1. We set `rrf_k` explicitly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from contextual_chunks import paths
from contextual_chunks.retrieval.models import EMB_MODEL, RRK_MODEL

ROOT = Path(paths.ROOT)
INDEX_ROOT = Path(paths.INDEX)
CHUNKS_FILE = Path(paths.LABELED)
QUERIES_DIR = Path(paths.QUERIES).parent

INDEX_TEXT = {
    "chunk": "чанк как есть",
    "context_chunk": "контекст + чанк",
    "context": "только контекст",
}
BM25_BACKENDS = ("fastembed", "rank_bm25", "none")
FUSIONS = ("native", "manual")


@dataclass(frozen=True)
class RetrievalConfig:
    """One run = index (ingest side) + query side + fusion."""

    name: str = "custom"
    # --- ingest side (affects the index) ---
    index_text: str = "chunk"
    bm25: str = "rank_bm25"   # rank_bm25 (local) | fastembed (ONNX, cache required) | none
    bm25_stem: bool = True            # Snowball stemmer for Russian
    bm25_idf_modifier: bool = True    # Qdrant `Modifier.IDF` over TF vectors
    embed_model: str = EMB_MODEL
    embed_dim: int | None = None      # MRL truncation (1024 -> 256/512)
    embed_max_length: int = 2048
    token_budget: int = 32768
    # --- query side ---
    fusion: str = "native"            # native (Qdrant RRF) | manual (our RRF)
    rrf_k: int = 60
    weights: tuple[float, ...] | None = None   # (dense, bm25), dense-dominant = (0.7, 0.3)
    prefetch_limit: int = 64
    top_k: int = 50
    rerank: bool = False
    rerank_pool: int = 64
    context_source: str = "teacher"   # teacher | student (see artifacts/results/student_context.jsonl)
    rerank_model: str = RRK_MODEL

    def __post_init__(self) -> None:
        assert self.index_text in INDEX_TEXT, f"index_text: {self.index_text}"
        assert self.bm25 in BM25_BACKENDS, f"bm25: {self.bm25}"
        assert self.fusion in FUSIONS, f"fusion: {self.fusion}"
        if self.weights is not None:
            assert len(self.weights) == (2 if self.bm25 != "none" else 1)

    # -- identifiers ------------------------------------------------------
    @property
    def index_text_mode(self) -> str:
        return self.index_text

    def _index_payload(self) -> dict:
        """Fields that the index vectors depend on (the index cache key)."""
        return {"index_text": self.index_text, "context_source": self.context_source,
                "embed_model": self.embed_model,
                "embed_dim": self.embed_dim, "embed_max_length": self.embed_max_length,
                "token_budget": self.token_budget, "bm25": self.bm25,
                "bm25_stem": self.bm25_stem, "bm25_idf_modifier": self.bm25_idf_modifier}

    def index_fingerprint(self) -> str:
        blob = json.dumps(self._index_payload(), sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def query_fingerprint(self) -> str:
        blob = json.dumps({**self._index_payload(), "fusion": self.fusion, "rrf_k": self.rrf_k,
                           "weights": self.weights, "prefetch_limit": self.prefetch_limit,
                           "top_k": self.top_k, "rerank": self.rerank,
                           "rerank_pool": self.rerank_pool,
                           "rerank_model": self.rerank_model}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def dense_cache_path(self, backend: str) -> Path:
        tag = f"{backend}-{self.embed_model}-ml{self.embed_max_length}-d{self.embed_dim or 0}"
        return INDEX_ROOT / "cache" / f"{hashlib.sha1(tag.encode()).hexdigest()[:12]}.npz"

    def qdrant_path(self, backend: str = "mlx", tag: str = "") -> Path:
        """`tag` separates debug indexes (e.g. a smoke subsample) from the full one."""
        stem = f"{backend}-{tag}-" if tag else f"{backend}-"
        return INDEX_ROOT / "qdrant" / f"{stem}{self.index_fingerprint()}"

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> "RetrievalConfig":
        return replace(self, **kw)


# Article arm matrix: A/B/C/D
ARMS: dict[str, RetrievalConfig] = {
    "A": RetrievalConfig(name="A flat / RRF", index_text="chunk", rerank=False),
    "B": RetrievalConfig(name="B ctx / RRF", index_text="context_chunk", rerank=False),
    "C": RetrievalConfig(name="C ctx / RRF + rerank", index_text="context_chunk", rerank=True),
    "D": RetrievalConfig(name="D flat / RRF + rerank", index_text="chunk", rerank=True),
    # ablation: dense without BM25 (on a homogeneous corpus BM25 pulls in foreign documents)
    "E": RetrievalConfig(name="E flat / dense-only", index_text="chunk", bm25="none"),
    "F": RetrievalConfig(name="F ctx / dense-only", index_text="context_chunk", bm25="none"),
    # student context (distilled 0.6B) instead of the teacher context
    "G": RetrievalConfig(name="G student-ctx / dense-only", index_text="context_chunk",
                         bm25="none", context_source="student"),
    "H": RetrievalConfig(name="H student-ctx / RRF", index_text="context_chunk",
                         bm25="rank_bm25", context_source="student"),
}

# Base config for ablations (arm A/B without the reranker)
BASE = ARMS["B"]


def arm(name: str) -> RetrievalConfig:
    return ARMS[name.upper()]
