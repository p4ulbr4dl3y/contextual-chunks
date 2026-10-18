"""Fast batched inference for Qwen3-Embedding / Qwen3-Reranker (CPU, MPS, CUDA).

Optimizations for the benchmarks:
* batches sorted by length (minimum padding) + order restoration;
* fp16 on MPS/CUDA, fp32 on CPU; SDPA attention;
* `logits_to_keep=1` on the reranker, so we do not compute logits [B, L, 151k];
* truncation: 2048 tokens by default (our chunk is <=1500), 32k is not needed;
* an on-disk embedding cache so repeated benchmark runs do not re-embed;
* MRL: dimension truncation (1024 -> 256/512) + re-normalization.

Prompts/tokens are taken from the models themselves:
`config_sentence_transformers.json` (prompt "query") and `1_LogitScore/config.json`
(true_token_id=9693 "yes", false_token_id=2152 "no").
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EMB_MODEL = os.environ.get("EMB_MODEL", "Qwen/Qwen3-Embedding-0.6B")
RRK_MODEL = os.environ.get("RRK_MODEL", "Qwen/Qwen3-Reranker-0.6B")

EMB_QUERY_PROMPT = os.environ.get(
    "EMB_QUERY_PROMPT",
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
)
RRK_INSTRUCT = os.environ.get(
    "RRK_INSTRUCT", "Given a web search query, retrieve relevant passages that answer the query"
)


def pick_device(explicit: str | None = None) -> str:
    dev = explicit or os.environ.get("OCR_DEVICE")
    if dev:
        return dev
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    return torch.float32 if device == "cpu" else torch.float16


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Hidden state of the last NON-padding token (reverse-argmax, any padding side)."""
    idx = mask.shape[1] - 1 - mask.flip(1).int().argmax(dim=1)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def bucket_indices(lengths, token_budget: int = 32768, max_rows: int = 128) -> list[list[int]]:
    """Groups indices by length with a padding budget: rows * longest <= token_budget.

    Why not a fixed batch: padding goes up to the longest row in the batch, so an
    unsorted mix of lengths inflates the work (in the doc2seek audit: 4.21x on a
    mix, 46.4x on 47 short + 1 long). Here the budget limits exactly the padding
    overhead, and `max_rows` is the batch size. A row longer than the budget forms
    a bucket on its own. The caller restores the input order.
    """
    buckets: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in sorted(range(len(lengths)), key=lambda j: lengths[j]):
        ln = max(1, lengths[i])
        nxt = max(cur_max, ln)
        if cur and ((len(cur) + 1) * nxt > token_budget or len(cur) + 1 > max_rows):
            buckets.append(cur)
            cur, nxt = [], ln
        cur.append(i)
        cur_max = nxt
    if cur:
        buckets.append(cur)
    return buckets


def _sorted_batches(lengths, batch_size):
    """Compatibility: length sort only, budget unlimited."""
    return bucket_indices(lengths, token_budget=10**12, max_rows=batch_size)


class Embedder:
    TOKEN_BUDGET = 32768

    def __init__(self, name: str = EMB_MODEL, device: str | None = None, dtype=None,
                 max_length: int = 2048, dim: int | None = None, batch_size: int = 32,
                 token_budget: int | None = None):
        self.name, self.device = name, pick_device(device)
        self.dtype = dtype or pick_dtype(self.device)
        self.max_length, self.dim, self.batch_size = max_length, dim, batch_size
        self.token_budget = token_budget or self.TOKEN_BUDGET
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        try:
            self.model = AutoModel.from_pretrained(
                name, dtype=self.dtype, attn_implementation="sdpa"
            ).to(self.device).eval()
        except Exception:  # fp16 on MPS sometimes fails -> fp32
            self.dtype = torch.float32
            self.model = AutoModel.from_pretrained(
                name, dtype=self.dtype, attn_implementation="sdpa"
            ).to(self.device).eval()

    @property
    def out_dim(self) -> int:
        return self.dim or self.model.config.hidden_size

    @torch.inference_mode()
    def _encode_batch(self, texts: list[str]) -> torch.Tensor:
        enc = self.tok(texts, padding=True, pad_to_multiple_of=8, truncation=True,
                       max_length=self.max_length, return_tensors="pt").to(self.device)
        hidden = self.model(**enc).last_hidden_state
        emb = last_token_pool(hidden, enc["attention_mask"])
        if self.dim:
            emb = emb[:, : self.dim]
        return F.normalize(emb, p=2, dim=1)

    def encode(self, texts: list[str], batch_size: int | None = None, prompt: str | None = None,
               normalize: bool = True, sort_by_length: bool = True, cache: EmbedCache | None = None,
               dim: int | None = None, max_length: int | None = None,
               token_budget: int | None = None):
        """-> np.ndarray [n, out_dim] float32 (L2-normalized).

        `dim` is a one-off MRL truncation (1024 -> 256/512), `max_length` a one-off
        token cap, `token_budget` a one-off padding budget; all of it works without
        recreating the model (needed for ablations and benchmarks).
        """
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
            toks = self.tok([prep[i] for i in todo], add_special_tokens=False)["input_ids"]
            if sort_by_length:
                batches = bucket_indices([len(t) for t in toks], self.token_budget, bs)
            else:
                batches = [[todo[j] for j in range(i, min(i + bs, len(todo)))]
                           for i in range(0, len(todo), bs)]
            for batch in batches:
                vecs = self._encode_batch([prep[i] for i in batch]).float().cpu().numpy()
                for k, i in enumerate(batch):
                    out[i] = vecs[k]
                    if cache:
                        cache.put(prep[i], vecs[k], tag)
        if not normalize:
            out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return out

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(t) for t in self.tok(texts, add_special_tokens=False)["input_ids"]]


class Reranker:
    """Qwen3-Reranker: score = P("yes") over the last-token logits."""

    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
        "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
        "<|im_end|>\n<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    TOKEN_BUDGET = 32768

    def __init__(self, name: str = RRK_MODEL, device: str | None = None, dtype=None,
                 max_length: int = 2048, batch_size: int = 16, token_budget: int | None = None):
        self.name, self.device = name, pick_device(device)
        self.dtype = dtype or pick_dtype(self.device)
        self.max_length, self.batch_size = max_length, batch_size
        self.token_budget = token_budget or self.TOKEN_BUDGET
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                name, dtype=self.dtype, attn_implementation="sdpa"
            ).to(self.device).eval()
        except Exception:
            self.dtype = torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                name, dtype=self.dtype, attn_implementation="sdpa"
            ).to(self.device).eval()
        cfg = json.load(open(os.path.join(_local_dir(name), "1_LogitScore", "config.json")))
        self.true_id, self.false_id = cfg["true_token_id"], cfg["false_token_id"]
        self.prefix = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suffix = self.tok.encode(self.SUFFIX, add_special_tokens=False)
        self.body_max = max(64, self.max_length - len(self.prefix) - len(self.suffix))
        self._keep: int | None = 1 if self._supports_logits_to_keep() else None

    def _supports_logits_to_keep(self) -> bool:
        try:
            probe = self.tok(["x"], return_tensors="pt").to(self.device)
            self.model(**probe, logits_to_keep=1)
            return True
        except Exception:
            return False

    def _format(self, pairs, instruction: str) -> list[str]:
        return [f"<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs]

    @torch.inference_mode()
    def _score_batch(self, bodies: list[str]) -> torch.Tensor:
        enc = self.tok(bodies, padding=False, truncation="longest_first",
                       max_length=self.body_max, return_attention_mask=False)
        ids = [self.prefix + x + self.suffix for x in enc["input_ids"]]
        batch = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        if self._keep:
            logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :]
        else:
            logits = self.model(**batch).logits[:, -1, :]
        two = torch.stack([logits[:, self.false_id], logits[:, self.true_id]], dim=1).float()
        return torch.softmax(two, dim=1)[:, 1]

    def score(self, pairs: list[tuple[str, str]], batch_size: int | None = None,
              instruction: str | None = None, sort_by_length: bool = True) -> list[float]:
        """pairs = [(query, document)] -> probability of "yes" (0..1)."""
        if not pairs:
            return []
        instruction = instruction or RRK_INSTRUCT
        bodies = self._format(pairs, instruction)
        lens = [len(b) for b in self.tok(bodies, add_special_tokens=False)["input_ids"]]
        scores = [0.0] * len(bodies)
        bs = batch_size or self.batch_size
        if sort_by_length:
            batches = bucket_indices(lens, self.token_budget, bs)
        else:
            batches = [list(range(i, min(i + bs, len(bodies)))) for i in range(0, len(bodies), bs)]
        for batch in batches:
            vals = self._score_batch([bodies[i] for i in batch]).tolist()
            for k, i in enumerate(batch):
                scores[i] = vals[k]
        return scores

    def rank(self, query: str, documents: list[str], **kw) -> list[tuple[int, float]]:
        s = self.score([(query, d) for d in documents], **kw)
        return sorted(enumerate(s), key=lambda x: -x[1])


class EmbedCache:
    """On-disk embedding cache: repeated benchmark runs are free."""

    def __init__(self, path: str, tag: str = ""):
        # np.savez_compressed appends .npz itself; without normalization the cache
        # was never read (the file exists, but os.path.exists(path) == False)
        if path and not path.endswith(".npz"):
            path += ".npz"
        self.path, self.tag = path, tag
        self._mem: dict[str, np.ndarray] = {}
        self.hits = self.misses = 0
        if path and os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            self._mem = {k: z[k] for k in z.files}

    @staticmethod
    def _key(text: str, tag: str) -> str:
        return hashlib.sha1((tag + "\x00" + text).encode("utf-8")).hexdigest()[:20]

    def get(self, text: str, extra: str = ""):
        v = self._mem.get(self._key(text, self.tag + extra))
        self.hits += 1 if v is not None else 0
        self.misses += 1 if v is None else 0
        return v

    def put(self, text: str, vec: np.ndarray, extra: str = "") -> None:
        self._mem[self._key(text, self.tag + extra)] = np.asarray(vec, dtype=np.float32)

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        np.savez_compressed(self.path, **self._mem)


def _local_dir(name: str) -> str:
    """Local snapshot from the HF cache (the models are already downloaded, no network)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(name, local_files_only=True)


def timeit(fn, *a, warmup: int = 1, repeats: int = 1, **kw) -> tuple[float, object]:
    """-> (best time, result); actually warms up MPS."""
    out = None
    for _ in range(warmup):
        out = fn(*a, **kw)
        _sync(pick_device())
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        _sync(pick_device())
        best = min(best, time.perf_counter() - t0)
    return best, out
