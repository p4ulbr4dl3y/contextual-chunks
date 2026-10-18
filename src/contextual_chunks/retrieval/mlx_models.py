"""MLX backend for Qwen3-Embedding / Qwen3-Reranker (Apple Silicon GPU, bf16).

Same API as the torch wrappers in `models.py`, but faster for local runs
(benchmarks, config sweeps). Weights come from the same HF snapshots as the torch path,
so the comparison is fair.

Taken from the production service `doc2seek` (`apps/ml/src/ml_service/backends/mlx/`):
* loading via `mlx_lm.utils._get_classes` + `args_cls.from_dict(conf)` and key prefix
  normalization (`model.`) with `strict=True`, so we do not hardcode the arch;
* token-budget bucketing (`bucket_indices`): padding in a batch goes up to the longest
  row, the doc2seek audit measured 4.21x overhead on a mix of lengths and 46.4x on
  47 short + 1 long;
* **right padding** instead of attention_mask: with a causal mask the output of a real
  position does not see the padding on the right, so we take the hidden state at index
  `len-1` of each row (`take_along_axis`), not the common `-1` column;
* logits only for the last position: `[B, L, 151k]` does not fit in memory
  (~20% of forward time in the doc2seek audit went to the full head);
* a check that the `yes`/`no` tokens exist and differ (otherwise you get a
  silent constant score of 0.5).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm.utils import _get_classes, load_tokenizer

from contextual_chunks.retrieval.models import (EMB_MODEL, EMB_QUERY_PROMPT, RRK_MODEL, Reranker,
                                          bucket_indices)

MLX_DTYPE = {"bf16": mx.bfloat16, "bfloat16": mx.bfloat16, "fp16": mx.float16,
             "float16": mx.float16, "fp32": mx.float32, "float32": mx.float32}


def _load_model(name: str, dtype: str = "float16"):
    """Qwen3 in MLX: the converted cache first, otherwise convert on the fly.

    Conversion (`mlx_convert.py`) puts fp16 weights on disk: no cast on every start,
    the file is half the size, precision is higher than bf16, and on M2 fp16 is also 1.39x faster.
    """
    from contextual_chunks.retrieval.mlx_convert import convert, target_dir

    local = target_dir(name, dtype)
    path = str(local) if (local / "model.safetensors").exists() else str(convert(name, dtype))

    conf = json.load(open(os.path.join(path, "config.json"), encoding="utf-8"))
    cls, args_cls = _get_classes(conf)
    model = cls(args_cls.from_dict(conf))
    model.load_weights(list(mx.load(os.path.join(path, "model.safetensors")).items()),
                       strict=True)
    mx.eval(model.parameters())

    tok = getattr(load_tokenizer(path), "_tokenizer", None) or load_tokenizer(path)
    tok.padding_side = "right"  # right padding -> the causal mask makes it invisible
    return model, tok


def _norm_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _gather_last(hidden: mx.array, lengths: np.ndarray) -> mx.array:
    """(B, L, H) -> (B, 1, H) at the last real position of each row."""
    pos = mx.array(lengths.astype(np.int32)) - 1
    return mx.take_along_axis(hidden, pos[:, None, None], axis=1)


class MlxEmbedder:
    TOKEN_BUDGET = 32768

    def __init__(self, name: str = EMB_MODEL, max_length: int = 2048, dim: int | None = None,
                 batch_size: int = 32, dtype: str = "float16", device: str | None = None,
                 token_budget: int | None = None):
        self.name = name
        self.model, self.tok = _load_model(name, dtype)
        self.max_length, self.dim, self.batch_size = max_length, dim, batch_size
        self.token_budget = token_budget or self.TOKEN_BUDGET
        self.device = "mlx-gpu"

    @property
    def out_dim(self) -> int:
        return self.dim or self.model.args.hidden_size

    def _embed(self, texts: list[str]) -> np.ndarray:
        enc = self.tok(texts, padding=True, pad_to_multiple_of=8, truncation=True,
                       max_length=self.max_length, return_tensors="np")
        ids = mx.array(enc["input_ids"].astype(np.int32))
        hidden = self.model.model(ids)
        lengths = enc["attention_mask"].sum(axis=1)
        emb = _gather_last(hidden, lengths)[:, 0, :]
        if self.dim:
            emb = emb[:, : self.dim]
        return np.array(emb.astype(mx.float32))

    def encode(self, texts: list[str], batch_size: int | None = None, prompt: str | None = None,
               normalize: bool = True, sort_by_length: bool = True, cache=None,
               dim: int | None = None, max_length: int | None = None,
               token_budget: int | None = None) -> np.ndarray:
        if ((dim is not None and dim != self.dim)
                or (max_length is not None and max_length != self.max_length)
                or (token_budget is not None and token_budget != self.token_budget)):
            prev = (self.dim, self.max_length, self.token_budget)
            self.dim = dim if dim is not None else self.dim
            self.max_length = max_length if max_length is not None else self.max_length
            self.token_budget = token_budget if token_budget is not None else self.token_budget
            try:
                return self.encode(texts, batch_size, prompt, normalize, sort_by_length, cache)
            finally:
                self.dim, self.max_length, self.token_budget = prev
        if not texts:
            return np.zeros((0, self.out_dim), dtype=np.float32)
        prompt = EMB_QUERY_PROMPT if prompt is None else prompt
        prep = [(prompt + t) if prompt else t for t in texts]
        tag = f"d{self.dim or 0}"
        out = np.zeros((len(prep), self.out_dim), dtype=np.float32)
        todo = []
        for i, t in enumerate(prep):
            hit = cache.get(t, tag) if cache else None
            if hit is None:
                todo.append(i)
            else:
                out[i] = hit
        if todo:
            bs = batch_size or self.batch_size
            lens = [len(x) for x in
                    self.tok([prep[i] for i in todo], add_special_tokens=False)["input_ids"]]
            batches = (bucket_indices(lens, self.token_budget, bs) if sort_by_length else
                       [list(range(i, min(i + bs, len(todo)))) for i in range(0, len(todo), bs)])
            for batch in batches:
                vecs = self._embed([prep[todo[j]] for j in batch])
                for k, j in enumerate(batch):
                    out[todo[j]] = vecs[k]
                    if cache:
                        cache.put(prep[todo[j]], vecs[k], tag)
        return _norm_rows(out) if normalize else out

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(t) for t in self.tok(texts, add_special_tokens=False)["input_ids"]]


class MlxReranker:
    TOKEN_BUDGET = 32768
    TRUE_ID, FALSE_ID = 9693, 2152  # 1_LogitScore/config.json ("yes" / "no")

    def __init__(self, name: str = RRK_MODEL, max_length: int = 2048, batch_size: int = 16,
                 dtype: str = "float16", device: str | None = None,
                 token_budget: int | None = None):
        from contextual_chunks.retrieval.models import RRK_INSTRUCT

        self.name = name
        self.model, self.tok = _load_model(name, dtype)
        self.max_length, self.batch_size = max_length, batch_size
        self.token_budget = token_budget or self.TOKEN_BUDGET
        self.device = "mlx-gpu"
        self.instr = RRK_INSTRUCT
        self.true_id = self.tok.convert_tokens_to_ids("yes")
        self.false_id = self.tok.convert_tokens_to_ids("no")
        unk = getattr(self.tok, "unk_token_id", None)
        if (self.true_id is None or self.false_id is None or self.true_id == self.false_id
                or unk in (self.true_id, self.false_id)):
            raise ValueError(
                f"no distinct yes/no tokens in {name}: yes={self.true_id} no={self.false_id} "
                f"unk={unk} (otherwise every document gets the same score)"
            )
        if (self.true_id, self.false_id) != (self.TRUE_ID, self.FALSE_ID):
            raise ValueError(f"verdict tokens changed: {self.true_id}/{self.false_id}")
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None \
            else self.tok.eos_token_id
        self.prefix = self.tok.encode(Reranker.PREFIX, add_special_tokens=False)
        self.suffix = self.tok.encode(Reranker.SUFFIX, add_special_tokens=False)
        self.body_max = max(64, self.max_length - len(self.prefix) - len(self.suffix))
        self._keep = 1

    def _verdict_logits(self, ids: mx.array, lengths: np.ndarray) -> mx.array:
        hidden = self.model.model(ids)                       # [B, L, H]
        last = _gather_last(hidden, lengths)                 # [B, 1, H]
        if getattr(self.model.args, "tie_word_embeddings", False):
            return self.model.model.embed_tokens.as_linear(last)   # tied head
        return self.model.lm_head(last)

    def _score(self, bodies: list[str]) -> np.ndarray:
        enc = self.tok(bodies, padding=False, truncation="longest_first",
                       max_length=self.body_max, return_attention_mask=False)
        rows = [self.prefix + x + self.suffix for x in enc["input_ids"]]
        lengths = np.array([len(r) for r in rows], dtype=np.int32)
        width = int(lengths.max())
        arr = np.full((len(rows), width), self.pad_id, dtype=np.int32)
        for i, r in enumerate(rows):
            arr[i, : len(r)] = r
        logits = self._verdict_logits(mx.array(arr), lengths)[:, 0, :].astype(mx.float32)
        two = np.stack([np.array(logits[:, self.false_id]), np.array(logits[:, self.true_id])], axis=1)
        e = np.exp(two - two.max(axis=1, keepdims=True))
        return e[:, 1] / e.sum(axis=1)

    def score(self, pairs: list[tuple[str, str]], batch_size: int | None = None,
              instruction: str | None = None, sort_by_length: bool = True) -> list[float]:
        if not pairs:
            return []
        instruction = instruction or self.instr
        bodies = [f"<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs]
        lens = [len(b) for b in self.tok(bodies, add_special_tokens=False)["input_ids"]]
        scores = [0.0] * len(bodies)
        bs = batch_size or self.batch_size
        batches = (bucket_indices(lens, self.token_budget, bs) if sort_by_length else
                   [list(range(i, min(i + bs, len(bodies)))) for i in range(0, len(bodies), bs)])
        for batch in batches:
            vals = self._score([bodies[i] for i in batch]).tolist()
            for k, i in enumerate(batch):
                scores[i] = vals[k]
        return scores

    def rank(self, query: str, documents: list[str], **kw) -> list[tuple[int, float]]:
        s = self.score([(query, d) for d in documents], **kw)
        return sorted(enumerate(s), key=lambda x: -x[1])
