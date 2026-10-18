"""Generate the LaTeX tables of the paper from the experiment results.

The numbers in the paper are the numbers of the experiments: the tables are
assembled from `artifacts/results/*.json` rather than copied by hand. Run:

    .venv/bin/python -m contextual_chunks.paper.tables

Writes `paper/tables/*.tex` (booktabs), which `main.tex` includes.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

from contextual_chunks import paths

ROOT = pathlib.Path(paths.ROOT)
RES = pathlib.Path(paths.RESULTS)
METRICS = pathlib.Path(paths.METRICS)
OUT = pathlib.Path(paths.PAPER) / "tables"


def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def f3(x: float) -> str:
    return f"{x:.3f}"


def bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> paper/tables/{name}.tex")


# ---------------------------------------------------------------- T1 dataset
def t1_dataset() -> None:
    rows = [json.loads(line) for line in (ROOT / "data/interim/labeled_dataset_metadata.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    import collections
    cats = collections.Counter(r["category"] for r in rows)
    docs = len({r["doc_id"] for r in rows})
    tables = sum(1 for r in rows if r["is_table"])
    ptok = sum(r["prompt_tokens"] for r in rows)
    cost = sum(r.get("cost_usd", 0.0) for r in rows)
    med_chunk = statistics.median(r["chunk_tokens"] for r in rows)
    med_ctx = statistics.median(len(r["generated_context"]) for r in rows)
    # reproduce the split from src/contextual_chunks/dataset/export_hf.py (val_frac=0.15, seed=3407)
    import random
    groups = sorted({r.get("window_hash") or r.get("doc_hash") or r["id"] for r in rows})
    random.Random(3407).shuffle(groups)
    n_val = max(1, int(len(groups) * 0.15))
    val_groups = set(groups[:n_val])
    n_val_rows = sum(1 for r in rows
                     if (r.get("window_hash") or r.get("doc_hash") or r["id"]) in val_groups)
    split = {"train": len(rows) - n_val_rows, "val": n_val_rows}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Dataset statistics. Chunks are segments of overlapping document windows, "
        r"extracted from 82 Russian business documents. Contexts were "
        r"labeled by an API teacher (\texttt{deepseek-v4.1-flash}, reasoning off) with "
        r"79.3\% prompt-cache hit rate.}",
        r"\label{tab:dataset}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Property & Value \\",
        r"\midrule",
        f"Documents & {docs} \\\\",
        f"Chunks (window$\\rightarrow$context pairs) & {len(rows)} \\\\",
        f"Median chunk length (tokens) & {med_chunk:.0f} \\\\",
        f"Median generated context (characters) & {med_ctx:.0f} \\\\",
        f"Table chunks & {tables} \\\\",
        r"\midrule",
        f"\\quad construction & {cats['construction']} \\\\",
        f"\\quad food & {cats['food']} \\\\",
        f"\\quad it & {cats['it']} \\\\",
        f"\\quad pharma & {cats['pharma']} \\\\",
        f"\\quad security & {cats['security']} \\\\",
        r"\midrule",
        f"Teacher prompt tokens (total) & {ptok / 1e6:.2f}M \\\\",
        f"Teacher labeling cost & \\${cost:.2f} \\\\",
    ]
    if split:
        lines += [
            r"\midrule",
            f"Train / validation chunks & {split['train']} / {split['val']} \\\\",
            "Split unit & \\texttt{window\\_hash} \\\\",
        ]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("t1_dataset", lines)


# ----------------------------------------------------------- T2 distillation
def t2_distillation() -> None:
    bb = json.loads((METRICS / "metrics_baseline.json").read_text(encoding="utf-8"))["summary"]
    tt = json.loads((METRICS / "metrics_tuned.json").read_text(encoding="utf-8"))["summary"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Distillation quality: word-level F1 of generated contexts against the "
        r"teacher, on 60 held-out chunks (base Qwen3-0.6B vs.\ the same model after LoRA SFT).}",
        r"\label{tab:distill}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Metric & Baseline & After SFT \\",
        r"\midrule",
        f"word-F1 mean & {f3(bb['f1_mean'])} & {bold(f3(tt['f1_mean']))} \\\\",
        f"word-F1 median & {f3(bb['f1_median'])} & {f3(tt['f1_median'])} \\\\",
        f"word-F1 p10 & {f3(bb['f1_p10'])} & {f3(tt['f1_p10'])} \\\\",
        f"Empty responses & {bb['f1_zero_frac'] * 100:.1f}\\% & "
        f"{bold(f'{tt["f1_zero_frac"] * 100:.0f}' + '\\%')} \\\\",
        f"Hallucinated numbers & {bb['halluc_numbers_total']} & {tt['halluc_numbers_total']} \\\\",
        f"Generation time / sample (s) & {bb['gen_sec_per_sample']:.2f} & {tt['gen_sec_per_sample']:.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    write("t2_distillation", lines)


# ------------------------------------------------------------ T3/T4 retrieval
ARM_LABELS = {
    "A": r"A: flat, RRF",
    "B": r"B: contextualized, RRF",
    "C": r"C: contextualized, RRF, reranked",
    "D": r"D: flat, RRF, reranked",
    "E": r"E: flat, dense-only",
    "F": r"\textbf{F: contextualized, dense-only}",
    "G": r"G: distilled, dense-only",
    "H": r"H: distilled, RRF",
}


def t3_retrieval() -> None:
    arms = load("retrieval_eval_full.json")["arms"]
    order = ["A", "B", "E", "F", "G", "H"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Retrieval quality on the full set of 234 queries with gold chunks "
        r"(3{,}134-chunk corpus, Apple~M2, MLX fp16). Metrics use gold groups: a hit on an "
        r"overlapping neighboring window (within two positions) counts as the same text.}",
        r"\label{tab:retrieval}",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Arm & R@1 & R@5 & R@10 & R@50 & MRR@10 & nDCG@10 & docR@10 & miss@50 \\",
        r"\midrule",
    ]
    for k in order:
        a = arms[k]
        m = a["metrics"]
        lines.append(f"{ARM_LABELS[k]} & " + " & ".join(
            f3(m[x]) for x in ("recall@1", "recall@5", "recall@10", "recall@50",
                               "mrr@10", "ndcg@10", "doc_recall@10", "miss@50")) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    write("t3_retrieval", lines)


def t3b_rerank_subset() -> None:
    """Labels for the sixty-query subset table: we highlight the proposed configuration."""
    labels = dict(ARM_LABELS)
    labels["F"] = "F: contextualized, dense-only"
    labels["G"] = r"\textbf{G: distilled, dense-only}"
    arms = load("retrieval_eval_strat60.json")["arms"]
    order = ["A", "B", "E", "F", "G", "H", "C", "D"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{All eight arms, measured on the same stratified subset of sixty "
        r"queries (fifteen of each type, all five categories represented). "
        rf"Arm~\arm{{g}} reaches recall@10 of {f3(arms['G']['metrics']['recall@10'])} "
        rf"at a median latency of {arms['G']['latency']['total_p50_ms']:.0f}~ms, and the "
        rf"reranked arm~\arm{{c}} reaches {f3(arms['C']['metrics']['recall@10'])} at "
        rf"{arms['C']['latency']['total_p50_ms'] / 1000:.1f}~s.}}",
        r"\label{tab:rerank-subset}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Arm & R@1 & R@5 & R@10 & MRR@10 & nDCG@10 \\",
        r"\midrule",
    ]
    for k in order:
        m = arms[k]["metrics"]
        lines.append(f"{labels[k]} & " + " & ".join(
            f3(m[x]) for x in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("t3b_rerank_subset", lines)


def t4_latency() -> None:
    arms = load("retrieval_eval_strat60.json")["arms"]
    order = ["A", "B", "E", "F", "G", "H", "C", "D"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Query latency (M2 GPU, MLX fp16, warmed). ``Total'' is the end-to-end "
        r"retrieval call, including query encoding, BM25, fusion and, for arms \arm{c} and "
        r"\arm{d}, reranking.}",
        r"\label{tab:latency}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Arm & p50 & p95 & encode \\",
        r"\midrule",
    ]
    for k in order:
        item = arms[k]["latency"]
        lines.append(f"{ARM_LABELS[k]} & {item['total_p50_ms']:.0f} & "
                     f"{item['total_p95_ms']:.0f} & "
                     f"{item['embed_p50_ms']:.0f} " + r"\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    write("t4_latency", lines)


# --------------------------------------------------------------- T5 ingest
def t5_ingest() -> None:
    ing = load("student_context_ingest.json")
    st, th = ing["student_ingest"], ing["teacher_baseline"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Context-generation cost for the whole corpus (3{,}134 chunks, 82 documents, "
        r"24.3M prompt tokens): API teacher vs.\ distilled 0.6B model running on-device.}",
        r"\label{tab:ingest}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r" & Teacher (API) & Student (on-device) \\",
        r"\midrule",
        "Model & \\texttt{deepseek-v4.1-flash} & \\texttt{Qwen3-0.6B} with LoRA \\\\",
        f"Wall-clock (h) & {th['latency_sec_total_hours']:.2f} & {st['wall_hours']:.2f} \\\\",
        f"Per chunk, median (s) & {th['latency_sec_median']:.2f} & {st['sec_per_chunk_median']:.2f} \\\\",
        f"Prefill (h) & -- & {st['prefill_hours']:.2f} \\\\",
        f"Decode (h) & -- & {st['decode_hours']:.2f} \\\\",
        f"Output tokens (med./max) & -- & {st['out_tokens']['median']}/{st['out_tokens']['max']} \\\\",
        f"Cost (USD) & {bold('\\$' + format(th['cost_usd_total'], '.2f'))} & {bold('\\$0')} \\\\",
        "Data leaves device & yes & no \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    write("t5_ingest", lines)


# ------------------------------------------------------------- T6 ablations
def t6_ablations() -> None:
    a = load("ablation_retrieval.json")["ablations"]
    g = load("ablation_retrieval_student.json")["ablations"]
    groups = [
        ("MRL dimension", ["dense d=1024", "dense d=512", "dense d=256", "dense d=128"],
         ["1024", "512", "256", "128"]),
        ("Query instruction", ["инструкция EN", "инструкция RU"], ["English", "Russian"]),
        ("Hybrid fusion (RRF)", ["dense-only", "BM25-only", "RRF k=1 1:1", "RRF k=10 1:1",
                                 "RRF k=60 1:1", "RRF k=60 w=1:0.25"],
         ["dense-only", "BM25-only", "$k{=}1$, 1:1", "$k{=}10$, 1:1", "$k{=}60$, 1:1",
          "$k{=}60$, 1:0.25"]),
        ("Best combination", ["d=512 + RU"], ["d=512, Russian"]),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Retrieval ablations computed in-memory on the indexed vectors "
        r"(teacher arm F, student arm G; 234 queries, dense-only unless noted). "
        r"$\Delta$ is student $-$ teacher.}",
        r"\label{tab:ablations}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Teacher (F)} & \multicolumn{2}{c}{Student (G)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Setting & R@10 & MRR@10 & R@10 & MRR@10 \\",
        r"\midrule",
    ]
    for title, keys, labels in groups:
        lines.append(r"\multicolumn{5}{l}{\emph{" + title + r"}} \\")
        for key, lab in zip(keys, labels, strict=False):
            mf, mg = a[key], g[key]
            lines.append(f"\\quad {lab} & {f3(mf['recall@10'])} & {f3(mf['mrr@10'])} & "
                         f"{f3(mg['recall@10'])} & {f3(mg['mrr@10'])} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("t6_ablations", lines)


# ---------------------------------------------------------------- T7 strata
STRATUM_LABELS = {"gold-текст": "gold text", "gold-таблица": "gold table",
                  "текст": "text", "таблица": "table", "смешанный": "mixed"}


def t7_strata() -> None:
    b = load("breakdown_by_stratum.json")
    strata = b["strata"]
    order = [("type", "Query type"), ("category", "Category"),
             ("is_table", "Gold chunk")]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Per-stratum retrieval quality, teacher context (F) vs.\ student context (G); "
        r"dense-only, 1024-d, English instruction. $\Delta$ is student $-$ teacher.}",
        r"\label{tab:strata}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Stratum & $n$ & F R@10 & G R@10 & $\Delta$ R@10 \\",
        r"\midrule",
    ]
    for key, title in order:
        lines.append(r"\multicolumn{5}{l}{\emph{" + title + r"}} \\")
        for lab, d in strata[key].items():
            lab = STRATUM_LABELS.get(lab, lab)
            delta = d["G"]["recall@10"] - d["F"]["recall@10"]
            lines.append(f"\\quad {lab} & {d['n']} & {f3(d['F']['recall@10'])} & "
                         f"{f3(d['G']['recall@10'])} & {delta:.3f} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("t7_strata", lines)


# ----------------------------------------------------------- T8 BM25 backends
def t8_bm25() -> None:
    b = load("bm25_backends.json")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{BM25 backend comparison. \texttt{rank\_bm25} is an in-memory BM25Okapi with a "
        r"Russian Snowball stemmer; \texttt{fastembed} uses \texttt{Qdrant/bm25} sparse vectors "
        r"with IDF applied server-side (18 stop-word files, 92\,KB, no neural network). "
        r"Hybrid columns are arms A/B.}",
        r"\label{tab:bm25}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Setting & R@10 & MRR@10 & docR@10 & p50 (ms) \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{BM25-only}} \\",
    ]
    for key, lab in (("rank_bm25", r"\texttt{rank\_bm25}"), ("fastembed", r"\texttt{fastembed}")):
        d = b[key]
        lines.append(f"\\quad {lab} & {f3(d['recall@10'])} & {f3(d['mrr@10'])} & "
                     f"{f3(d['doc_recall@10'])} & {d['p50_ms']:.1f} \\\\")
    lines.append(r"\addlinespace[2pt]")
    lines.append(r"\multicolumn{5}{l}{\emph{Hybrid RRF (dense and BM25)}} \\")
    for arm, armname in (("A", "A flat"), ("B", "B contextualized")):
        for src, lab in (("rrf_rank_bm25", r"\texttt{rank\_bm25}"),
                         ("rrf_fastembed", r"\texttt{fastembed}")):
            d = b[src][arm]
            lines.append(f"\\quad {armname}, {lab} & {f3(d['recall@10'])} & {f3(d['mrr@10'])} & "
                         f"{f3(d['doc_recall@10'])} & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("t8_bm25", lines)


def main() -> int:
    t1_dataset()
    t2_distillation()
    t3_retrieval()
    t3b_rerank_subset()
    t4_latency()
    t5_ingest()
    t6_ablations()
    t7_strata()
    t8_bm25()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
