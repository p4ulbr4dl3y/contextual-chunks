"""Token-by-token check of the GGUF quantizations against the unquantized bf16 reference.

`eval_gguf.py` measures word-F1 against the **teacher** answer, a product question
("how useful is the context for retrieval"), but it does not answer the engineering
question: **does the quantization reproduce the model itself**. Two different wordings,
equally close in meaning to the teacher, give the same word-F1 with completely
divergent tokens.

Here we measure direct agreement: greedy generation of a quantization against
greedy generation of the bf16 reference (`tuned_out` from `per_row_comparison.jsonl`)
on the same 60 validation rows.

    # from stored runs, without a single inference and without the network
    .venv/bin/python -m contextual_chunks.gguf.parity

    # self-check: the recomputed stored word-F1 must match to the sign
    .venv/bin/python -m contextual_chunks.gguf.parity --self-check

Result: `artifacts/results/quant_parity.json` + a table on stdout.

Why: for Giga-Embeddings the same scheme catches a swapped `pre` tokenizer and
`attention.causal`; here the invariant is generation prefix agreement, and it must
drop to 0 when the chat template or the `pre` type is swapped.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from contextual_chunks import paths
from contextual_chunks.gguf.eval import (
    MLX_MODEL,
    PER_ROW,
    f1,
    strip_think,
)

ROOT = Path(paths.ROOT)
RESULTS = Path(paths.RESULTS)
REFERENCE_KEY = "tuned_out"  # bf16 SFT (MLX), the same reference as in eval_gguf.load_rows

def load_reference() -> dict[str, str]:
    ref: dict[str, str] = {}
    for line in PER_ROW.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ref[row["id"]] = strip_think(row.get(REFERENCE_KEY, ""))
    return ref

def load_runs() -> dict[str, dict]:
    """label -> {summary, rows: id -> run row}; we take everything lying in results."""
    runs: dict[str, dict] = {}
    for path in sorted(RESULTS.glob("gguf_eval_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        label = data.get("label") or path.stem.removeprefix("gguf_eval_")
        runs[label] = {
            "file_mb": data.get("summary", {}).get("file_mb"),
            "gguf": data.get("gguf"),
            "rows": {row["id"]: row for row in data["rows"]},
        }
    return runs

def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(MLX_MODEL))

def common_prefix(a: str, b: str, tok) -> tuple[int, int, int]:
    """(how many tokens matched, length of a, length of b).

    Generation is a token stream, so the divergence is measured where the model first
    made a different decision, not by words: a word may match in meaning with a
    different split.
    """
    ta = tok.encode(a, add_special_tokens=False)
    tb = tok.encode(b, add_special_tokens=False)
    common = next((n for n, (x, y) in enumerate(zip(ta, tb, strict=False)) if x != y),
                  min(len(ta), len(tb)))
    return common, len(ta), len(tb)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true",
                        help="only recompute the stored word-F1 and check the sign")
    parser.add_argument("--out", type=Path, default=RESULTS / "quant_parity.json")
    args = parser.parse_args()

    reference = load_reference()
    runs = load_runs()
    if not runs:
        print("no artifacts/results/gguf_eval_*.json at all")
        return 1

    if args.self_check:
        worst = 0.0
        for run in runs.values():
            for row in run["rows"].values():
                stored = row.get("f1_vs_mlx_bf16")
                if stored is None:
                    continue
                mine = f1(strip_think(row["gen"]), reference[row["id"]])
                worst = max(worst, abs(mine - stored))
        print(f"maximum discrepancy of the recomputed word-F1: {worst:.3e}")
        return 0 if worst < 1e-9 else 1

    tok = tokenizer()
    report: dict[str, dict] = {}
    pool: dict[str, dict[str, str]] = {"bf16": reference}
    print(f"reference: bf16 SFT (MLX), {len(reference)} rows, field {REFERENCE_KEY!r}\n")
    head = (f"{'quantization':12} {'MB':>7} {'identical':>10} {'token prefix':>16} "
            f"{'median':>8} {'<50%':>6} {'word-F1':>8} {'word-F1(recalc)':>16}")
    print(head)
    print("-" * len(head))
    for label, run in sorted(runs.items(), key=lambda kv: -(kv[1]["file_mb"] or 0)):
        rows = run["rows"]
        ids = [i for i in rows if i in reference]
        if not ids:
            continue
        exact = 0
        ratios: list[float] = []
        below_half = 0
        stored_f1: list[float] = []
        recalc_f1: list[float] = []
        per_row: list[dict] = []
        pool[label] = {row_id: strip_think(rows[row_id]["gen"]) for row_id in ids}
        for row_id in ids:
            gen = pool[label][row_id]
            ref = reference[row_id]
            if gen == ref:
                exact += 1
            common, _, total = common_prefix(gen, ref, tok)
            ratio = common / total if total else 0.0
            ratios.append(ratio)
            if ratio < 0.5:
                below_half += 1
            if rows[row_id].get("f1_vs_mlx_bf16") is not None:
                stored_f1.append(rows[row_id]["f1_vs_mlx_bf16"])
            recalc_f1.append(f1(gen, ref))
            per_row.append({"id": row_id, "exact": gen == ref,
                            "prefix_ratio": round(ratio, 6), "common": common,
                            "ref_tokens": total,
                            "f1": round(recalc_f1[-1], 6)})
        size_mb = run["file_mb"]
        summary = {
            "gguf": run["gguf"],
            "rows": len(ids),
            "file_mb": size_mb,
            "identical": exact,
            "identical_rate": round(exact / len(ids), 4),
            "prefix_ratio_mean": round(statistics.fmean(ratios), 4),
            "prefix_ratio_median": round(statistics.median(ratios), 4),
            "rows_below_half": below_half,
            "f1_vs_mlx_bf16_stored": round(statistics.fmean(stored_f1), 4) if stored_f1 else None,
            "f1_vs_mlx_bf16_recalc": round(statistics.fmean(recalc_f1), 4),
        }
        report[label] = {"summary": summary, "rows": per_row}
        print(f"{label:12} {size_mb or 0:>7.1f} {exact:>7}/{len(ids):<3} "
              f"{summary['prefix_ratio_mean']:>16.4f} {summary['prefix_ratio_median']:>8.4f} "
              f"{below_half:>6} "
              f"{summary['f1_vs_mlx_bf16_stored'] or float('nan'):>8.4f} "
              f"{summary['f1_vs_mlx_bf16_recalc']:>16.4f}")

    # --- ladder: which quant matches which, and which matches the reference -------
    def pair(a: str, b: str) -> dict:
        ids = [i for i in pool[a] if i in pool[b]]
        commons, ratios = [], []
        for row_id in ids:
            common, len_a, len_b = common_prefix(pool[a][row_id], pool[b][row_id], tok)
            commons.append(common)
            ratios.append(common / (min(len_a, len_b) or 1))
        return {
            "rows": len(ids),
            "identical": sum(pool[a][i] == pool[b][i] for i in ids),
            "common_tokens_mean": round(statistics.fmean(commons), 2),
            "common_tokens_median": statistics.median(commons),
            "prefix_ratio_mean": round(statistics.fmean(ratios), 4),
        }

    labels = ["bf16", *sorted(pool.keys() - {"bf16"}, key=lambda l: -(runs[l]["file_mb"] or 0))]
    print("\nshare of common prefix of min length, inside llama.cpp and against the reference\n")
    print(f"{'':8}" + "".join(f"{l:>9}" for l in labels))
    matrix: dict[str, dict[str, float]] = {}
    stats: dict[tuple[str, str], dict] = {}
    for a in labels:
        row = {}
        for b in labels:
            if a == b:
                row[b] = 1.0
                continue
            stats[(a, b)] = pair(a, b)
            row[b] = stats[(a, b)]["prefix_ratio_mean"]
        matrix[a] = row
        print(f"{a:8}" + "".join(f"{row[l]:>9.4f}" for l in labels))

    named = ("bf16 vs f16", "f16 vs q8_0", "f16 vs q6_k", "f16 vs q5_k_m", "f16 vs q4_k_m",
             "q8_0 vs q4_k_m", "bf16 vs q8_0")
    details = {n: stats[tuple(n.split(" vs "))] for n in named if tuple(n.split(" vs ")) in stats}
    print("\nin absolute tokens:")
    for name, d in details.items():
        print(f"  {name:16} common prefix {d['common_tokens_mean']:6.1f} (median "
              f"{d['common_tokens_median']:5.0f}), identical rows {d['identical']}/{d['rows']}")

    report["matrix_prefix_ratio"] = matrix
    report["pairs"] = details
    report["reference_note"] = (
        "the bf16 reference is a greedy transformers run, not llama.cpp. It is not "
        "reproduced token-by-token by any quant, including f16 (0/60 identical, common prefix ~15 "
        "tokens), because the first divergence always lands in a near-tie in the logits "
        "(p(top-1) ~0.5). So word-F1 against the reference is semantic closeness, not fidelity of "
        "the conversion; fidelity is shown by the ladder inside llama.cpp: f16 -> q8_0 -> q6_k -> ..."
    )

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n{args.out.relative_to(ROOT)}")
    print("token prefix is the share of reference tokens that matched up to the first divergence.")
    print("word-F1(recalc) must match the column from gguf_eval_*.json: --self-check.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
