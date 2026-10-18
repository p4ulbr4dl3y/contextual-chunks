"""Evaluation of the student GGUF quantizations on 60 validation rows (word-F1 vs teacher).

The metric and the STOP list come exactly from the training notebook
(`notebooks/sft_final.ipynb`); the references are the same rows as in the comparison
run (`artifacts/training/per_row_comparison.jsonl`), so the GGUF numbers are
comparable with the distillation table in the README (bf16 SFT = 0.5373).

The prompt is built with the same chat template with `enable_thinking=False` as during
training, and is sent to `llama-server` in raw-completion mode; the llama.cpp template
is not applied, so the tokens match the torch/MLX run.

    .venv/bin/python -m contextual_chunks.gguf.eval --gguf ~/.cache/contextual-chunks/gguf/qwen3-0.6b-contextual-chunks-q8_0.gguf --label q8_0
    .venv/bin/python -m contextual_chunks.gguf.eval --metric-check   # metric self-check
    .venv/bin/python -m contextual_chunks.gguf.eval --summarize
    .venv/bin/python -m contextual_chunks.gguf.eval --paired q8_0 f16

The server is started with `-np 1` and `--cache-prompt`: one slot and batch prefix reuse
enabled. This is the correct mode for ingest: the prompt is `<document>` + the whole
window + `<chunk>` + instruction, so neighboring chunks of one window share a prefix,
and llama.cpp recomputes only the tail. The corpus ingest estimate in summary is built
from a linear fit of prefill time against the number of recomputed tokens and two
corpus modes (the first chunk of a window in full, and the tails); it is rebuilt
without runs via `--recompute`.

Result: `artifacts/results/gguf_eval_<label>.json` (+ the aggregate `gguf_quant.md`).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from contextual_chunks import paths

ROOT = Path(paths.ROOT)
MLX_MODEL = Path(paths.MLX_CACHE) / \
    "qwen3-0.6b-contextual-chunks-bfloat16"
LABELED = Path(paths.LABELED)
PER_ROW = Path(paths.ARTIFACTS) / "training" / "per_row_comparison.jsonl"
RESULTS = Path(paths.RESULTS)
LOG_DIR = Path(paths.GGUF_CACHE) / "logs"

CORPUS_CHUNKS = 3134            # chunks in the v2 dataset
CORPUS_WINDOWS = 320            # windows in the v2 dataset
# Sum of prompt_tokens over the corpus, from the ingest bench (artifacts/results/ingest_student.json).
CORPUS_PROMPT_TOKENS = 24308045
# The prompt is `<document>` + the whole window + `<chunk>` + instruction, so neighboring
# chunks of one window share a prefix, everything up to the chunk. Over the corpus: 320
# windows, 9.8 chunks per window, 81.4% shared prefix, so only the first occurrence of a
# window and the tail after the shared prefix really need prefill. The numbers are computed
# over labeled_dataset_metadata.jsonl (student tokenizer, window/chunk order).
CORPUS_FIRST_CHUNK_TOKENS = 2504569   # first chunks of windows: no prefix to reuse
CORPUS_SUFFIX_TOKENS = 2019149        # tail after the shared prefix (p50 of shared prefix 93.7%)

# metric from the training notebook (word-F1 against the teacher answer)
STOP = {"фрагмент", "контракт", "документ", "который", "описывает", "содержит", "оказание",
        "услуг", "услуги", "материалов", "строительных", "рамках", "также", "данный",
        "этого", "всего", "всех", "которые"}
CLOSE_THINK = "<" + "/" + "think" + ">"

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def words(t: str) -> set[str]:
    return {x for x in re.findall(r"[а-яёa-z0-9]+", t.lower()) if len(x) >= 5 and x not in STOP}


def f1(pred: str, ref: str) -> float:
    P, R = words(pred), words(ref)
    if not P or not R:
        return 0.0
    i = len(P & R)
    pr, rc = i / len(P), i / len(R)
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def numbers(t: str) -> set[str]:
    return set(re.findall(r"\d+", t))


def strip_think(text: str) -> str:
    """Qwen3 generates `thinking...` before the answer; the teacher does not have it."""
    if CLOSE_THINK in text:
        text = text.split(CLOSE_THINK, 1)[1]
    text = re.sub(r"^<\|im_end\|>", "", text.strip())
    return text.strip()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_per_row() -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l)
            for l in PER_ROW.read_text(encoding="utf-8").splitlines() if l.strip()}


def load_rows(n: int) -> list[dict]:
    """Validation rows in per_row_comparison order; the prompt comes from the labeled dataset."""
    per_row = load_per_row()
    rows, seen = [], set()
    for line in LABELED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"] in per_row and r["id"] not in seen:
            seen.add(r["id"])
            rows.append({"id": r["id"], "category": r["category"], "prompt_raw": r["prompt"],
                         "chunk": r.get("chunk", ""),
                         "window_hash": r.get("window_hash", ""),
                         "chunk_index": r.get("chunk_index", 0),
                         "teacher": per_row[r["id"]]["teacher"],
                         "mlx_bf16": per_row[r["id"]].get("tuned_out", ""),
                         "prompt_tokens": r.get("prompt_tokens")})
    missing = set(per_row) - seen
    if missing:
        log(f"WARNING: {len(missing)} per_row rows not found in the labeled dataset")
    return rows[:n] if n else rows


def render_prompts(rows: list[dict]) -> list[dict]:
    """apply_chat_template(enable_thinking=False), the same as in training and export_mlx --verify."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MLX_MODEL))
    for r in rows:
        r["prompt"] = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt_raw"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        r["prompt_ids"] = len(tok.encode(r["prompt"]))
    return rows


# ---------------------------------------------------------------------------
# llama-server
# ---------------------------------------------------------------------------
def wait_health(port: int, timeout: float = 300.0) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server did not come up on port {port}")


def start_server(bin_path: str, gguf: Path, port: int, ctx: int, ngl: int, logfile: Path,
                 np: int = 1, cache_prompt: bool = True, extra: list[str] | None = None):
    """`-np 1` matters: with several slots and a shared KV buffer prefix reuse breaks.

    `extra` are flags on top of the base ones (for example `--override-kv tokenizer.ggml.pre=str:gpt-4o`
    for the negative control in `verify_gguf.py`).
    """
    cmd = [bin_path, "-m", str(gguf), "-ngl", str(ngl), "-c", str(ctx), "-np", str(np),
           "--cache-prompt" if cache_prompt else "--no-cache-prompt",
           "--host", "127.0.0.1", "--port", str(port), *(extra or [])]
    log("$ " + " ".join(cmd))
    logfile.parent.mkdir(parents=True, exist_ok=True)
    fh = open(logfile, "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    try:
        wait_health(port)
        log("server ready")
    except TimeoutError:
        proc.kill()
        raise
    return proc


def complete(port: int, prompt: str, max_tokens: int, timeout: float = 600.0) -> dict:
    body = json.dumps({"prompt": prompt, "n_predict": max_tokens, "temperature": 0.0,
                       "top_k": 1, "cache_prompt": True, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def metric_check() -> int:
    """Self-check: word-F1 of the stored bf16 answer must give 0.5373 from report 03."""
    ref_mean, ref_base = 0.5373, 0.1893
    per_row = load_per_row()
    vals = [f1(v.get("tuned_out", ""), v["teacher"]) for v in per_row.values()]
    base = [f1(v.get("base_out", ""), v["teacher"]) for v in per_row.values()]
    mean, median = statistics.mean(vals), statistics.median(vals)
    base_mean, base_median = statistics.mean(base), statistics.median(base)
    print(f"SFT bf16: mean {mean:.4f}  median {median:.4f} (bf16 reference: 0.5373 / 0.534)")
    print(f"base 4bit: mean {base_mean:.4f}  median {base_median:.4f} "
          f"(base reference: 0.1893 / 0.1764)")
    if abs(mean - ref_mean) > 5e-5 or abs(base_mean - ref_base) > 5e-5:
        print(f"MISMATCH: expected means {ref_mean} and {ref_base}")
        return 1
    print("mean matches to 4 digits; the medians are recomputed over the same rows")

    # check the hallucinated-number detector on the same stored answers; it reads the
    # prompts, which only the dataset pipeline writes, so a fresh clone skips this part
    if not LABELED.exists():
        print(f"skipped the hallucinated-number check: {LABELED} is absent "
              "(build it with `make pipeline`)")
        return 0
    rows = load_rows(0)
    for field, label in (("mlx_bf16", "SFT bf16"), ):
        n = sum(len(numbers(r[field]) - numbers(r["prompt_raw"])) for r in rows)
        print(f"hallucinated numbers in {label}: {n} (for the final epoch: 0)")
    return 0


def bootstrap_diff(a: dict[str, float], b: dict[str, float], iters: int = 2000,
                   seed: int = 3407) -> dict:
    """Paired bootstrap of the difference of mean word-F1 over the common rows."""
    ids = sorted(set(a) & set(b))
    diffs = [a[i] - b[i] for i in ids]
    rnd = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rnd.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return {"n": n, "mean": sum(diffs) / n,
            "lo": means[int(0.025 * iters)], "hi": means[int(0.975 * iters) - 1],
            "share_gt0": sum(1 for m in means if m > 0) / iters}


def _load_f1(label: str) -> dict[str, float]:
    if label == "mlx":                     # bf16 MLX reference from report 03
        return {k: f1(v.get("tuned_out", ""), v["teacher"]) for k, v in load_per_row().items()}
    f = RESULTS / f"gguf_eval_{label}.json"
    if not f.exists():
        return {}
    return {r["id"]: r["f1"] for r in json.loads(f.read_text(encoding="utf-8"))["rows"]}


def paired(a_label: str, b_label: str) -> int:
    """Paired bootstrap over per-row word-F1: is A significantly better than B on the same 60 rows."""
    a, b = _load_f1(a_label), _load_f1(b_label)
    if not a or not b:
        log(f"missing gguf_eval_{a_label}.json or gguf_eval_{b_label}.json")
        return 1
    st = bootstrap_diff(a, b)
    print(f"{a_label} - {b_label} (n={st['n']}): mean {st['mean']:+.4f}, "
          f"95% CI [{st['lo']:+.4f}, {st['hi']:+.4f}], share of bootstrap means > 0: {st['share_gt0']:.3f}")
    return 0


def summarize() -> int:
    files = sorted(RESULTS.glob("gguf_eval_*.json"))
    if not files:
        log("no gguf_eval_*.json; run eval first")
        return 1
    runs_all = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    # ingest-order runs, for timing only, they do not go into the quality table
    runs = [r for r in runs_all if "ingest" not in r["label"]]
    if not runs:
        log("no quality runs (all have 'ingest' in the name)")
        return 1
    runs.sort(key=lambda r: r["summary"]["f1_mean"], reverse=True)
    mlx = _load_f1("mlx")   # bf16 MLX reference over the same 60 rows
    lines = ["# GGUF quantizations of the student (llama.cpp / Metal)",
             "",
             f"Metric: word-F1 vs the teacher context on {runs[0]['summary']['rows']} validation rows, "
             "greedy, `max_new_tokens=160`, prompt with `enable_thinking=False`.",
             "bf16 SFT (MLX) reference, the same rows, recomputed from `per_row_comparison.jsonl` "
             "(the median is the mean of two middle values; previously in the training artifacts "
             "it was taken as `scores[n//2]` and was biased upward, now fixed).",
             "",
             "| quantization | size, MB | word-F1 mean | median | p10 | empty | halluc. numbers |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        s = r["summary"]
        lines.append(f"| `{r['label']}` | {s['file_mb']:.0f} | {s['f1_mean']:.3f} | {s['f1_median']:.3f} | "
                     f"{s['f1_p10']:.3f} | {s['empty_rate']:.3f} | {s['halluc_numbers']} |")
    lines += ["", "bf16 SFT (MLX, the same 60 rows from `per_row_comparison.jsonl`): "
              f"mean {statistics.mean(mlx.values()):.3f} / median {statistics.median(mlx.values()):.3f}.", ""]

    # speed: the ingest-order run (window by window), where cold prefill is measured
    # on prompts of real length; if there is none, take any run with reuse
    reuse = [r for r in runs_all if "ingest" in r["label"] and r["summary"].get("reuse_share")]
    if not reuse:
        reuse = [r for r in runs_all if r["summary"].get("reuse_share")]
    if reuse:
        lines += ["## Corpus ingest: prompt prefix reuse", "",
                  "The prompt is `<document>` + the whole window + `</document>` + `<chunk>` + "
                  "instruction (the window takes ~82% of the prompt), so neighboring chunks of one "
                  "window share a prefix, and llama.cpp (`--cache-prompt`, `-np 1`) recomputes only "
                  "the tail. Over the corpus the shared prefix is 81.4% (p50 per chunk 93.7%), so we "
                  "must prefill 2.50M tokens of the first chunks of windows and 2.02M tokens of tails. "
                  "The measurement is an ingest-order run: window by window, as in a real corpus run.", "",
                  "| quantization | slots | reused, % | window prefill, tok/s | tail prefill, ",
                  "tok/s | decode, tok/s | corpus ingest, h |", "|---|---:|---:|---:|---:|---:|---:|"]
        for r in reuse:
            s = r["summary"]
            lines.append(f"| `{r['label']}` | {s.get('np', '?')} | {s['reuse_share']:.1%} | "
                         f"{s.get('prefill_window_tok_per_sec', 0):.0f} | "
                         f"{s.get('prefill_tail_tok_per_sec', 0):.0f} | "
                         f"{s['decode_tok_per_sec_p50']:.0f} | {s['est_ingest_hours']:.2f} |")
        nun = [r["label"] for r in runs if not r["summary"].get("reuse_share")]
        if nun:
            lines += ["", "Runs " + ", ".join(f"`{x}`" for x in nun) + " were made in the first "
                      "configuration (server without `-np 1`); their `est_ingest_hours` field is an "
                      "estimate of \"the whole corpus from scratch\", that is an upper bound; word-F1 "
                      "does not depend on the server configuration."]
        lines += [""]

    # significance: each quantization against f16 (paired bootstrap over the same rows)
    base = _load_f1("f16")
    if base:
        lines += ["## Significance of differences from f16 (paired bootstrap, n=%d)" % len(base), "",
                  "| comparison | Δ word-F1 | 95% CI |", "|---|---:|---|"]
        for label in [r["label"] for r in runs if r["label"] != "f16"]:
            other = _load_f1(label)
            if not other:
                continue
            st = bootstrap_diff(other, base)
            lines.append(f"| `{label}` − `f16` | {st['mean']:+.4f} | "
                         f"[{st['lo']:+.4f}, {st['hi']:+.4f}] |")
        lines += ["", "A CI that covers zero means the loss from quantization is "
                  "indistinguishable from zero on this slice.", ""]

    out = RESULTS / "gguf_quant.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {out}")
    print("\n".join(lines))
    return 0


def build_summary(recs: list[dict], file_mb: float, np: int, prompt_cache: bool) -> dict:
    """Summary of a run. The corpus ingest estimate comes from a linear fit of prefill over rows.

    Prefill time is almost linear in the number of recomputed tokens (`sec = a + b*tok`, ignoring
    the window length), so we fit over all rows and compute two corpus modes:
    the first chunk of a window is prefilled in full, the rest only the tail after the shared prefix.
    """
    f1s = [r["f1"] for r in recs]
    prefill = [r["prefill_tok_per_sec"] for r in recs if r["prefill_tok_per_sec"]]
    decode = [r["decode_tok_per_sec"] for r in recs if r["decode_tok_per_sec"]]
    # in runs before 2026-10 the prompt_full_tokens field was not written; we assume there was
    # no reuse (for those runs this is an upper bound)
    full_tok = sum(r.get("prompt_full_tokens") or r["prompt_tokens"] for r in recs)
    cached_tok = sum(r.get("cached_tokens", 0) for r in recs)
    reuse_share = cached_tok / full_tok if full_tok else 0.0

    xs = [r["prompt_tokens"] for r in recs]
    ys = [r["prefill_sec"] for r in recs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    first_len = CORPUS_FIRST_CHUNK_TOKENS / CORPUS_WINDOWS
    tail_len = CORPUS_SUFFIX_TOKENS / (CORPUS_CHUNKS - CORPUS_WINDOWS)
    if cached_tok:   # reuse worked: the corpus pays for whole windows and for tails
        est_prefill_h = (CORPUS_WINDOWS * (a + b * first_len) +
                         (CORPUS_CHUNKS - CORPUS_WINDOWS) * (a + b * tail_len)) / 3600
    else:            # cache off: we pay for all prompt tokens
        est_prefill_h = (CORPUS_WINDOWS * a + b * CORPUS_PROMPT_TOKENS) / 3600
    est_decode_h = (sum(r["decode_sec"] for r in recs) / len(recs) * CORPUS_CHUNKS) / 3600

    return {
        "rows": len(recs),
        "file_mb": file_mb,
        "f1_mean": round(statistics.mean(f1s), 4),
        "f1_median": round(statistics.median(f1s), 4),
        "f1_p10": round(sorted(f1s)[max(0, int(0.1 * len(f1s)) - 1)], 4),
        "f1_vs_mlx_bf16_mean": round(statistics.mean(r["f1_vs_mlx_bf16"] for r in recs), 4),
        "empty_rate": round(sum(r["empty"] for r in recs) / len(recs), 4),
        "halluc_numbers": sum(len(r["halluc_numbers"]) for r in recs),
        "prefill_tok_per_sec_p50": round(statistics.median(prefill), 1),
        "decode_tok_per_sec_p50": round(statistics.median(decode), 1),
        "np": np,
        "prompt_cache": prompt_cache,
        "reuse_share": round(reuse_share, 4),
        "prefill_fit_ms_per_tok": round(b * 1000, 4),
        "prefill_fit_sec_per_req": round(a, 3),
        "prefill_fit_r2": round(r2, 3),
        "prefill_window_tok_per_sec": round(first_len / (a + b * first_len), 1),
        "prefill_tail_tok_per_sec": round(tail_len / (a + b * tail_len), 1),
        "est_prefill_hours": round(est_prefill_h, 2),
        "est_decode_hours": round(est_decode_h, 2),
        "est_ingest_hours": round(est_prefill_h + est_decode_h, 2),
    }


def recompute() -> int:
    """Rebuild summary in all gguf_eval_*.json from the stored rows (without runs)."""
    n = 0
    for f in sorted(RESULTS.glob("gguf_eval_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        old = data.get("summary", {})
        data["summary"] = build_summary(data["rows"], file_mb=old.get("file_mb", 0.0),
                                        np=old.get("np", 1), prompt_cache=old.get("prompt_cache", True))
        f.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"{f.name}: ingest {data['summary']['est_ingest_hours']:.2f} h, "
            f"reused {data['summary']['reuse_share']:.1%}")
        n += 1
    log(f"files recomputed: {n}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, help="GGUF file to evaluate")
    ap.add_argument("--label", default=None, help="run name (by default from the file name)")
    ap.add_argument("--rows", type=int, default=60, help="how many validation rows (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--ctx", type=int, default=16384, help="must be >= the longest prompt")
    ap.add_argument("--ngl", type=int, default=99, help="layers on GPU (99 = all, Metal)")
    ap.add_argument("--np", type=int, default=1,
                    help="server slots; 1 is the correct mode for ingest (prefix reuse)")
    ap.add_argument("--no-prompt-cache", action="store_true", help="disable --cache-prompt")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--server-bin", default="llama-server")
    ap.add_argument("--url", default=None, help="an already running server instead of our own")
    ap.add_argument("--order", choices=("id", "ingest"), default="id",
                    help="row order: id is the per_row_comparison protocol (for the quality table), "
                         "ingest is window by window (for timing)")
    ap.add_argument("--metric-check", action="store_true", help="metric self-check only")
    ap.add_argument("--summarize", action="store_true", help="build gguf_quant.md from json")
    ap.add_argument("--recompute", action="store_true",
                    help="rebuild summary in the existing gguf_eval_*.json (without runs)")
    ap.add_argument("--paired", nargs=2, metavar=("A", "B"),
                    help="paired word-F1 bootstrap between two runs")
    args = ap.parse_args()

    if args.metric_check:
        return metric_check()
    if args.summarize:
        return summarize()
    if args.recompute:
        return recompute()
    if args.paired:
        return paired(*args.paired)
    if not args.gguf:
        ap.error("--gguf is required (or --metric-check / --summarize)")
    if not args.gguf.exists():
        log(f"no file {args.gguf}")
        return 1

    label = args.label or args.gguf.stem.split("-")[-1]
    rows = load_rows(args.rows)
    if args.order == "ingest":
        # the real run goes window by window, so this is the only way cold prefill
        # is measured on prompts of real length (the whole window)
        rows.sort(key=lambda r: (r["window_hash"], r["chunk_index"]))
    rows = render_prompts(rows)
    max_prompt = max(r["prompt_ids"] for r in rows)
    log(f"{len(rows)} rows, total prompt_tokens {sum(r['prompt_ids'] for r in rows)}, "
        f"max {max_prompt}")
    ctx = args.ctx
    if max_prompt + args.max_tokens + 32 > ctx:
        ctx = max_prompt + args.max_tokens + 32
        log(f"ctx raised to {ctx} for the longest prompt")
    # for prefix reuse the buffer must hold both the prefix and the tail of the new prompt
    if args.np > 1:
        log(f"WARNING: -np {args.np}; prefix reuse may break between slots")

    proc, port, logfile = None, args.port, LOG_DIR / f"gguf_server_{label}.log"
    if args.url:
        port = int(args.url.rsplit(":", 1)[1])
    else:
        proc = start_server(args.server_bin, args.gguf, port, ctx, args.ngl, logfile,
                            np=args.np, cache_prompt=not args.no_prompt_cache)

    recs = []
    try:
        for i, r in enumerate(rows, 1):
            resp = complete(port, r["prompt"], args.max_tokens)
            text = strip_think(resp.get("content", ""))
            tim = resp.get("timings", {})
            doc_nums = numbers(r["prompt_raw"])   # the document window + chunk are inside the prompt itself
            halluc = sorted(numbers(text) - doc_nums)
            rec = {
                "id": r["id"], "category": r["category"],
                "gen": text,
                "f1": round(f1(text, r["teacher"]), 4),
                "f1_vs_mlx_bf16": round(f1(text, r["mlx_bf16"]), 4),
                "out_tokens": tim.get("predicted_n"),
                "prompt_tokens": tim.get("prompt_n", r["prompt_ids"]),
                "prompt_full_tokens": r["prompt_ids"],
                "cached_tokens": tim.get("cache_n", 0),
                "prefill_sec": round(tim.get("prompt_ms", 0) / 1000, 3),
                "decode_sec": round(tim.get("predicted_ms", 0) / 1000, 3),
                "prefill_tok_per_sec": round(tim.get("prompt_per_second", 0), 1),
                "decode_tok_per_sec": round(tim.get("predicted_per_second", 0), 1),
                "empty": not text.strip(),
                "halluc_numbers": halluc,
            }
            recs.append(rec)
            if i % 10 == 0 or i == len(rows):
                log(f"{i}/{len(rows)} f1={rec['f1']:.3f} prefill={rec['prefill_tok_per_sec']:.0f} tok/s "
                    f"decode={rec['decode_tok_per_sec']:.0f} tok/s")
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()

    summary = build_summary(recs, file_mb=round(args.gguf.stat().st_size / 1024 / 1024, 1),
                            np=args.np, prompt_cache=not args.no_prompt_cache)
    out = RESULTS / f"gguf_eval_{label}.json"
    out.write_text(json.dumps({"label": label, "gguf": str(args.gguf), "server_log": str(logfile),
                               "summary": summary, "rows": recs}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    log(f"wrote {out}")
    log(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
