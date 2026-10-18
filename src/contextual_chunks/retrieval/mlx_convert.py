"""HF snapshot -> native MLX checkpoint (fp16 by default).

Why convert instead of casting on the fly:
* files are half the size (fp32 1.19 GB -> fp16 0.60 GB) -> faster cold start;
* no need to cast 310 tensors on every start, weights load via mmap as is;
* on M2 fp16 is **1.39x** faster than bf16 on embedding (bench measurement), and it is
  more precise than bf16 (10 mantissa bits vs 7), so this is a free win in speed and accuracy;
* the format is compatible with `mlx_lm.utils.load_tokenizer` and `_get_classes`.

Keys are normalized for mlx_lm: the HF checkpoint of Qwen3-Embedding sits without the
`model.` prefix, the reranker with it; we bring them to one form.

Run:
    .venv/bin/python -m contextual_chunks.retrieval.mlx_convert            # both models, fp16
    .venv/bin/python -m contextual_chunks.retrieval.mlx_convert --dtype bfloat16 --force
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

from contextual_chunks import paths
from contextual_chunks.retrieval.models import EMB_MODEL, RRK_MODEL

DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
COPY_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt", "chat_template.jinja", "generation_config.json",
              "config_sentence_transformers.json"]


def default_root() -> Path:
    return Path(paths.MLX_CACHE)


def target_dir(model_id: str, dtype: str, root: Path | None = None) -> Path:
    return (root or default_root()) / f"{model_id.replace('/', '--')}-{dtype}"


def convert(model_id: str, dtype: str = "float16", root: Path | None = None,
            force: bool = False) -> Path:
    out = target_dir(model_id, dtype, root)
    marker = out / "convert.json"
    if marker.exists() and not force:
        print(f"  {model_id} -> {out} (already converted, skipping)")
        return out
    snap = Path(snapshot_download(model_id, local_files_only=True))
    out.mkdir(parents=True, exist_ok=True)
    dt = DTYPES[dtype]

    t0 = time.perf_counter()
    weights: dict[str, mx.array] = {}
    for sf in sorted(glob.glob(str(snap / "*.safetensors"))):
        for k, v in mx.load(sf).items():
            key = k if k.startswith(("model.", "lm_head.")) else f"model.{k}"
            weights[key] = v.astype(dt)
    n_params = sum(int(v.size) for v in weights.values())
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    dt_conv = time.perf_counter() - t0

    for name in COPY_FILES:
        if (snap / name).exists():
            shutil.copy2(snap / name, out / name)
    # the reranker needs the true/false token ids
    if (snap / "1_LogitScore" / "config.json").exists():
        (out / "1_LogitScore").mkdir(exist_ok=True)
        shutil.copy2(snap / "1_LogitScore" / "config.json", out / "1_LogitScore" / "config.json")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    meta = {"source": model_id, "dtype": dtype, "params": n_params,
            "bytes": size, "bytes_per_param": round(size / max(1, n_params), 3),
            "convert_sec": round(dt_conv, 1), "python": "mlx"}
    marker.write_text(json.dumps(meta, indent=2))
    print(f"  {model_id} -> {out}")
    print(f"    {n_params/1e6:.1f}M params | {size/2**20:.0f} MB "
          f"({meta['bytes_per_param']} bytes/param) | conversion {dt_conv:.1f}s")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[EMB_MODEL, RRK_MODEL])
    ap.add_argument("--dtype", default="float16", choices=list(DTYPES))
    ap.add_argument("--root", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).expanduser() if args.root else default_root()
    print(f"MLX cache: {root}")
    for m in args.models:
        convert(m, args.dtype, root, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
