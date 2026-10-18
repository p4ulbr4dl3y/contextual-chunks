"""Qwen3 tokenizer helpers -- the pipeline sizes text in TOKENS, not chars.

Why: Russian legal prose tokenizes at ~2.85 chars/token, but markdown tables
collapse to ~1.2 chars/token (pipes, digits, spaces).  A char-based cap is a
2.4x lottery, so window/chunk limits are enforced with the real student
tokenizer here.

The tokenizer file is cached locally; if absent it is fetched once from the
HuggingFace hub (Qwen/Qwen3-0.6B, ~11 MB, apache-2.0).
"""

import os
import urllib.request

from contextual_chunks import paths

HF_URL = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"
DEFAULT_CACHE = os.path.join(paths.CACHE, "qwen3_tokenizer.json")

_TOKENIZER = None
_CACHE_PATH_USED = None


def cache_path():
    return os.environ.get("QWEN_TOKENIZER_JSON", DEFAULT_CACHE)


def _ensure_cached(path=None):
    path = path or cache_path()
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[tokenizer] downloading {HF_URL} -> {path}")
    urllib.request.urlretrieve(HF_URL, path)
    return path


def get_tokenizer(path=None):
    """Return a `tokenizers.Tokenizer` for Qwen3 (cached)."""
    global _TOKENIZER, _CACHE_PATH_USED
    if _TOKENIZER is not None and (path is None or path == _CACHE_PATH_USED):
        return _TOKENIZER
    from tokenizers import Tokenizer

    resolved = _ensure_cached(path)
    _TOKENIZER = Tokenizer.from_file(resolved)
    _CACHE_PATH_USED = resolved
    return _TOKENIZER


def count_tokens(text, tokenizer=None):
    """Number of Qwen3 tokens (special tokens excluded)."""
    if not text:
        return 0
    tok = tokenizer or get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False).ids)


def encode(text, tokenizer=None):
    tok = tokenizer or get_tokenizer()
    return tok.encode(text, add_special_tokens=False).ids


def decode(ids, tokenizer=None):
    tok = tokenizer or get_tokenizer()
    return tok.decode(ids)


def split_by_tokens(text, cap, tokenizer=None):
    """Split `text` into contiguous pieces of <= cap tokens (last resort).

    Cuts at real token boundaries using the tokenizer's character offsets, so
    every piece is an exact substring of `text` (the `chunk in document`
    invariant stays checkable) and the cap is never exceeded -- unlike an
    additive word-by-word estimate, which drifts because sub-word merges cross
    word boundaries.
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    tok = tokenizer or get_tokenizer()
    enc = tok.encode(text, add_special_tokens=False)
    if len(enc.ids) <= cap:
        return [text]

    offsets = enc.offsets
    pieces, start = [], 0
    while start < len(offsets):
        end = min(start + cap, len(offsets))
        piece = text[offsets[start][0]:offsets[end - 1][1]].strip()
        if piece:
            pieces.append(piece)
        start = end
    return pieces
