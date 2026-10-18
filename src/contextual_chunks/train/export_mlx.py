"""LoRA adapter -> merge with the base model -> MLX checkpoint (bf16 by default).

The merge happens **in memory**: no intermediate HF directories appear, only the
final MLX snapshot is written to disk. `--cleanup` deletes the downloaded adapter;
`--drop-base` also removes the base model from the HF cache.

    .venv/bin/python -m contextual_chunks.train.export_mlx --verify --cleanup
    .venv/bin/python -m contextual_chunks.train.export_mlx --dtype float16

Keys are normalized for mlx_lm: `model.` prefix, we drop `rotary_emb.inv_freq`
and `lm_head.weight` when `tie_word_embeddings` is set (otherwise strict loading fails).

Verification (`--verify`) compares three generations on the SAME tokens
(we apply the chat template ourselves: `mlx_lm.generate` does not apply it, and with a
raw string the model produces garbage) and prints the length of the matching prefix of
greedy tokens between torch fp32 and MLX bf16; that is the conversion fidelity test.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from contextual_chunks import paths

# the xet transport stalls mid-file on this channel -> classic HTTP
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER = "p4ulbr4dl3y/qwen3-0.6b-contextual-chunks"
OUT_ROOT = Path(paths.MLX_CACHE)
COPY_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
              "chat_template.jinja", "tokenizer.model"]

# prompt exactly as in the SFT dataset (we check that the adapter is alive)
VERIFY_PROMPT = (
    "<document>\nМУНИЦИПАЛЬНЫЙ КОНТРАКТ № 02/26 на поставку товаров.\n"
    "3.1. Поставка Товара осуществляется со дня заключения контракта до 31.08.2026.\n"
    "</document>\n\nВот фрагмент этого документа:\n\n"
    "<chunk>\n3.1. Поставка Товара осуществляется со дня заключения контракта до 31.08.2026 года.\n"
    "</chunk>\n\nСформулируй краткий контекст (1-2 предложения), который объясняет, "
    "что это за фрагмент и как он связан с остальным документом. "
    "Не пересказывай фрагмент, а дай контекст для поиска."
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# merge (in RAM)
# ----------------------------------------------------------------------------
def load_base(base: str):
    import torch
    from transformers import AutoModelForCausalLM

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    model.eval()
    log(f"  base loaded fp32: {time.perf_counter()-t0:.1f}s")
    return model


def merge_peft(base_model, adapter: str):
    """`merge_and_unload()` mutates the base model **in place**; call it last."""
    from peft import PeftModel

    t0 = time.perf_counter()
    merged = PeftModel.from_pretrained(base_model, adapter).merge_and_unload()
    merged.eval()
    log(f"  LoRA merged (fp32): {time.perf_counter()-t0:.1f}s")
    return merged


# ----------------------------------------------------------------------------
# MLX snapshot
# ----------------------------------------------------------------------------
def write_mlx(merged, tokenizer_src: Path, out: Path, dtype: str) -> dict:
    import mlx.core as mx
    import torch

    dtypes = {"bfloat16": mx.bfloat16, "float16": mx.float16}
    tied = bool(getattr(merged.config, "tie_word_embeddings", True))
    out.mkdir(parents=True, exist_ok=True)
    weights, dropped = {}, []
    for key, val in merged.state_dict().items():
        if key.endswith("rotary_emb.inv_freq") or (tied and key == "lm_head.weight"):
            dropped.append(key)
            continue
        name = key if key.startswith(("model.", "lm_head.")) else f"model.{key}"
        weights[name] = mx.array(val.to(torch.float32).numpy()).astype(dtypes[dtype])
    t0 = time.perf_counter()
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    n_params = sum(int(w.size) for w in weights.values())

    merged.config.to_json_file(str(out / "config.json"))
    for name in COPY_FILES:
        src = tokenizer_src / name
        if name != "config.json" and src.exists():
            shutil.copy2(src, out / name)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    info = {"params": n_params, "tensors": len(weights), "dtype": dtype, "bytes": size,
            "save_sec": round(time.perf_counter() - t0, 1), "dropped_keys": dropped,
            "tied_embeddings": tied, "tokenizer_src": str(tokenizer_src)}
    log(f"  MLX snapshot: {n_params/1e6:.1f}M params, {len(weights)} tensors, "
        f"{size/2**20:.0f} MB, {dtype}, saved in {info['save_sec']}s")
    if dropped:
        log(f"  dropped keys: {dropped}")
    return info


# ----------------------------------------------------------------------------
# verification: same input tokens, greedy on output
# ----------------------------------------------------------------------------
def chat_ids(tokenizer_src: Path, prompt: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_src)
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = enc["input_ids"]
    log(f"  prompt via chat template: {int(ids.shape[1])} tokens")
    return tok, ids, ids[0].tolist()


def torch_generate(model, tok, ids, max_tokens: int) -> tuple[str, list[int]]:
    import torch

    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False)
    new = out[0][ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True), new.tolist()


def mlx_generate(out_dir: Path, ids, max_tokens: int) -> tuple[str, list[int], list[str]]:
    import mlx.utils
    from mlx_lm import load, stream_generate

    model, tokenizer = load(str(out_dir))
    dtypes = sorted({str(v.dtype).split(".")[-1]
                     for _, v in mlx.utils.tree_flatten(model.parameters())})
    log(f"  weight dtype after load: {dtypes}")
    assert dtypes == ["bfloat16"], f"expected bfloat16, got {dtypes}"
    tokens, text = [], ""
    # without a sampler mlx_lm runs greedy, the same as do_sample=False in torch
    for resp in stream_generate(model, tokenizer, prompt=list(ids), max_tokens=max_tokens):

        tokens.append(resp.token)
        text += resp.text
    del model
    return text, tokens, dtypes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--out", default=None, help="MLX snapshot directory")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--verify", action="store_true", help="generate a verification context")
    ap.add_argument("--verify-tokens", type=int, default=96)
    ap.add_argument("--cleanup", action="store_true", help="delete the downloaded adapter")
    ap.add_argument("--drop-base", action="store_true", help="also delete the base model from the cache")
    args = ap.parse_args()

    out = Path(args.out) if args.out else OUT_ROOT / f"{args.adapter.split('/')[-1]}-{args.dtype}"
    log(f"base={args.base} adapter={args.adapter}\n-> {out}")

    t_all = time.perf_counter()
    base_dir = fetch(args.base)
    adapter_dir = fetch(args.adapter)
    log(f"  downloaded: base {base_dir.name}, adapter {adapter_dir}")

    base_model = load_base(args.base)
    tok, ids, ids_list = chat_ids(base_dir, VERIFY_PROMPT)
    info = {"base": args.base, "adapter": args.adapter, "out": str(out)}
    if args.verify:
        t_base, _ = torch_generate(base_model, tok, ids, args.verify_tokens)
        log(f"  [base without LoRA] {t_base.strip()[:200]}")

    merged = merge_peft(base_model, args.adapter)          # mutates base_model in place
    tok_src = adapter_dir if (adapter_dir / "tokenizer.json").exists() else base_dir
    info.update(write_mlx(merged, tok_src, out, args.dtype))

    if args.verify:
        log("\n=== verification (same input tokens, greedy) ===")
        try:
            t_merged, ids_torch = torch_generate(merged, tok, ids, args.verify_tokens)
            log(f"  [torch merged fp32] {t_merged.strip()[:200]}")
        except Exception as e:
            t_merged, ids_torch = "", []
            log(f"  [torch merged fp32] error: {type(e).__name__}: {e}")
        t_mlx, ids_mlx, dtypes = mlx_generate(out, ids_list, args.verify_tokens)
        log(f"  [mlx {args.dtype}]       {t_mlx.strip()[:200]}")
        same = 0
        for a, b in zip(ids_torch, ids_mlx):
            if a != b:
                break
            same += 1
        log(f"  matching greedy-token prefix torch fp32 vs mlx {args.dtype}: "
            f"{same}/{len(ids_torch)} (total unique tokens {len(ids_mlx)})")
        info["verify"] = {"torch": t_merged, "mlx": t_mlx, "same_prefix": same,
                          "n_tokens": len(ids_torch), "mlx_dtypes": dtypes}
        (out / "export_meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))

    log(f"\ntotal {time.perf_counter()-t_all:.1f}s, snapshot {out} ({info['bytes']/2**20:.0f} MB)")

    if args.cleanup:
        log("\n=== artifact cleanup ===")
        drop_from_cache([args.adapter] + ([args.base] if args.drop_base else []))
        for tmp in out.glob("*.tmp"):
            tmp.unlink()
        log("  base deleted as well" if args.drop_base
            else f"  base left in cache: {base_dir}")
    (out / "export_meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def fetch(repo: str, tries: int = 5) -> Path:
    """snapshot_download with retries: the system proxy at 127.0.0.1:1082 returns 503 in bursts."""
    from huggingface_hub import snapshot_download

    for attempt in range(1, tries + 1):
        try:
            return Path(snapshot_download(repo))
        except Exception as e:
            log(f"  downloading {repo}: attempt {attempt}/{tries} - {type(e).__name__}: {str(e)[:120]}")
            if attempt == tries:
                raise
            time.sleep(2 * attempt)
    raise RuntimeError("unreachable")


def drop_from_cache(repos: list[str]) -> None:
    """Delete repositories from the HF cache the proper way (together with blobs)."""
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_id in repos:
            size = repo.size_on_disk
            strategy = cache.delete_revisions(*[r.commit_hash for r in repo.revisions])
            strategy.execute()
            log(f"  removed from cache {repo.repo_type} {repo.repo_id}: {size/2**20:.0f} MB")


if __name__ == "__main__":
    sys.exit(main())
