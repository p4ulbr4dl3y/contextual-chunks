"""MLX bf16 student checkpoint -> GGUF f16 -> a set of quantizations for llama.cpp.

The converter comes from llama.cpp pinned by tag (by default `b10180`, exactly the
build installed via brew), so the GGUF is guaranteed to be readable by the local
`llama-cli`/`llama-server`.

The source is the MLX snapshot from `export_mlx.py`: the tensor names there are the
standard HF ones (`model.embed_tokens.weight`, `model.layers.N...`),
`tie_word_embeddings=true`, so the LoRA is not merged again and
`convert_hf_to_gguf.py` eats the directory as is.

    .venv/bin/python -m contextual_chunks.gguf.convert
    .venv/bin/python -m contextual_chunks.gguf.convert --quants Q8_0 --keep-f16

Result: `~/.cache/contextual-chunks/gguf/qwen3-0.6b-contextual-chunks-<quant>.gguf`
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from contextual_chunks import paths

ROOT = Path(paths.ROOT)

ROOT_CACHE = Path(paths.GGUF_CACHE)
GIT_URL = "https://github.com/ggml-org/llama.cpp"
DEFAULT_TAG = "b10180"          # matches the version of the brew llama.cpp package
MODEL = Path(paths.MLX_CACHE) / \
    "qwen3-0.6b-contextual-chunks-bfloat16"
STEM = "qwen3-0.6b-contextual-chunks"
DEFAULT_QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def run(cmd: list[str], **kw) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


# ---------------------------------------------------------------------------
# llama.cpp sources (converter + gguf-py only, no build needed)
# ---------------------------------------------------------------------------
def ensure_source(src: Path, tag: str) -> Path:
    if (src / "convert_hf_to_gguf.py").exists() and (src / "gguf-py" / "gguf").exists():
        log(f"llama.cpp sources already present: {src}")
        return src
    src.parent.mkdir(parents=True, exist_ok=True)
    if not (src / ".git").exists():
        src.mkdir(parents=True, exist_ok=True)
        run(["git", "-C", src, "init", "-q"])
        run(["git", "-C", src, "remote", "add", "origin", GIT_URL])
    log(f"fetching llama.cpp {tag} (depth 1)")
    run(["git", "-C", src, "fetch", "-q", "--depth", "1", "origin", f"refs/tags/{tag}"])
    run(["git", "-C", src, "checkout", "-q", "FETCH_HEAD"])
    return src


def convert_to_f16(src: Path, model: Path, out: Path, outtype: str) -> None:
    """convert_hf_to_gguf.py: peek pass -> write; PYTHONPATH points to gguf-py from the same tag."""
    env = dict(os.environ, PYTHONPATH=str(src / "gguf-py"))
    cmd = [sys.executable, str(src / "convert_hf_to_gguf.py"), str(model),
           "--outfile", str(out), "--outtype", outtype]
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def quantize(bin_path: str, f16: Path, out: Path, quant: str) -> None:
    if out.exists():
        out.unlink()
    # positional arguments: model-quant.gguf type [nthreads]
    run([bin_path, str(f16), str(out), quant, str(os.cpu_count() or 8)])


def size_mb(p: Path) -> float:
    return p.stat().st_size / 1024 / 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODEL, help="student MLX bf16 snapshot")
    ap.add_argument("--out-dir", type=Path, default=ROOT_CACHE)
    ap.add_argument("--src", type=Path, default=ROOT_CACHE / "llama.cpp", help="llama.cpp clone")
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--quants", nargs="+", default=DEFAULT_QUANTS)
    ap.add_argument("--outtype", default="f16", choices=["f16", "bf16"])
    ap.add_argument("--f16-name", default=None, help="ready f16 GGUF (reuse instead of converting)")
    ap.add_argument("--keep-f16", action="store_true", help="do not delete the intermediate f16 GGUF")
    ap.add_argument("--quantize-bin", default="llama-quantize")
    args = ap.parse_args()

    if not args.model.exists():
        log(f"no model {args.model}; run `make export-mlx` first")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    f16 = Path(args.f16_name) if args.f16_name else args.out_dir / f"{STEM}-{args.outtype}.gguf"
    if f16.exists():
        log(f"f16 GGUF already present: {f16} ({size_mb(f16):.0f} MB)")
    else:
        src = ensure_source(args.src, args.tag)
        log(f"converting {args.model.name} -> {args.outtype} GGUF")
        convert_to_f16(src, args.model, f16, args.outtype)
        log(f"done: {f16} ({size_mb(f16):.0f} MB)")

    if args.quantize_bin and not shutil.which(args.quantize_bin):
        log(f"no {args.quantize_bin} in PATH; quantization skipped (brew install llama.cpp)")
        return 1

    report = {"f16": {"file": f16.name, "mb": round(size_mb(f16), 1)}}
    for q in args.quants:
        out = args.out_dir / f"{STEM}-{q.lower()}.gguf"
        log(f"quantizing {q}")
        quantize(args.quantize_bin, f16, out, q)
        report[q] = {"file": out.name, "mb": round(size_mb(out), 1)}
        log(f"  {out.name}: {size_mb(out):.0f} MB")

    if not args.keep_f16:
        log("deleting the intermediate f16 GGUF (--keep-f16 keeps it)")
        f16.unlink()

    log("total:")
    for k, v in report.items():
        log(f"  {k:8s} {v['mb']:7.1f} MB  {v['file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
