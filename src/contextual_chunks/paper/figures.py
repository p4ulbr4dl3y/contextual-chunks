"""Paper figures: PDF (vector, for pdflatex), assembled from the results.

    .venv/bin/python -m contextual_chunks.paper.figures
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from contextual_chunks import paths

RES = pathlib.Path(paths.RESULTS)
OUT = pathlib.Path(paths.PAPER) / "figures"

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_DENSE = "#1f77b4"
C_RERANK = "#d62728"
C_BM25 = "#7f7f7f"

LABELS = {"gold-текст": "gold text", "gold-таблица": "gold table",
          "текст": "text", "таблица": "table", "смешанный": "mixed"}

def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))

def fmt_ms(x, _):
    if x >= 1000:
        return f"{x / 1000:.0f}s"
    return f"{x:.0f}ms"

# ------------------------------------------------------------------ figure 1
def fig_quality_latency() -> None:
    """All eight arms on the same subset of 60 queries."""
    arms = load("retrieval_eval_strat60.json")["arms"]

    pts = [
        ("A: flat, RRF", "A", C_BM25, "o"),
        ("B: contextualised, RRF", "B", C_BM25, "s"),
        ("E: flat, dense-only", "E", C_DENSE, "o"),
        ("F: contextualised, dense-only", "F", C_DENSE, "s"),
        ("G: distilled, dense-only", "G", C_DENSE, "^"),
        ("H: distilled, RRF", "H", C_BM25, "^"),
        ("C: contextualised, reranked", "C", C_RERANK, "s"),
        ("D: flat, reranked", "D", C_RERANK, "o"),
    ]

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    handles = []
    for label, key, color, marker in pts:
        a = arms[key]
        x = a["latency"]["total_p50_ms"]
        y = a["metrics"]["recall@10"]
        h = ax.scatter(x, y, color=color, marker=marker, s=32, zorder=3, label=label)
        handles.append(h)

    for lab, key, color in (("F", "F", C_DENSE), ("C", "C", C_RERANK)):
        a = arms[key]
        ms = a["latency"]["total_p50_ms"]
        txt = f"{lab}: {ms:.0f} ms" if ms < 1000 else f"{lab}: {ms / 1000:.1f} s"
        ax.annotate(txt, (ms, a["metrics"]["recall@10"]), textcoords="offset points",
                    xytext=(16, 4) if lab == "F" else (0, 9),
                    ha="left" if lab == "F" else "center", fontsize=6.4, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Query latency, p50 (log scale)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0.40, 0.78)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_ms))
    ax.axvspan(30, 130, color=C_DENSE, alpha=0.07, zorder=0)
    factor = arms["C"]["latency"]["total_p50_ms"] / arms["F"]["latency"]["total_p50_ms"]
    ax.annotate(rf"$F\rightarrow C$: ${factor:.0f}$-fold", (1400, 0.735), fontsize=7,
                color=C_RERANK, ha="center")
    ax.annotate("", xy=(31000, 0.72), xytext=(40, 0.72),
                arrowprops=dict(arrowstyle="<->", color=C_RERANK, lw=0.7))
    ax.legend(handles=handles, frameon=False, fontsize=6.3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_quality_latency.pdf")
    plt.close(fig)
    print("-> paper/figures/fig_quality_latency.pdf")

# ------------------------------------------------------------------ figure 2
def fig_breakeven() -> None:
    ing = load("student_context_ingest.json")
    t_student = ing["student_ingest"]["wall_hours"]
    t_teacher = ing["teacher_baseline"]["latency_sec_total_hours"]
    docs = ing["corpus"]["docs"]
    cost_teacher = ing["teacher_baseline"]["cost_usd_total"]

    strat = load("retrieval_eval_strat60.json")["arms"]
    t_q_dense = strat["F"]["latency"]["total_p50_ms"] / 1000 / 3600
    t_q_rerank = strat["C"]["latency"]["total_p50_ms"] / 1000 / 3600

    qs = [10 ** e for e in range(1, 6)]
    D = 500
    scale = D / docs

    def total(t_ing_per_doc, t_q, q):
        return D * t_ing_per_doc + q * t_q

    series = [
        ("teacher context, dense-only", t_teacher / docs, t_q_dense, C_DENSE, "-"),
        ("student context, dense-only", t_student / docs, t_q_dense, C_DENSE, "--"),
        ("teacher context, reranked", t_teacher / docs, t_q_rerank, C_RERANK, "-"),
        ("student context, reranked", t_student / docs, t_q_rerank, C_RERANK, "--"),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for label, t_ing, t_q, color, ls in series:
        ys = [total(t_ing, t_q, q) for q in qs]
        ax.plot(qs, ys, color=color, ls=ls, lw=1.3, marker="o", ms=2.6, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of queries $Q$")
    ax.set_ylabel(r"Total time $T(D{=}500, Q)$, hours")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_breakeven.pdf")
    plt.close(fig)
    print("-> paper/figures/fig_breakeven.pdf")

# ------------------------------------------------------------------ figure 3
def fig_strata() -> None:
    b = load("breakdown_by_stratum.json")["strata"]
    labels, deltas, colors = [], [], []
    for key in ("type", "category", "is_table"):
        for lab, d in b[key].items():
            labels.append(LABELS.get(lab, lab.replace("_", " ")))
            delta = d["G"]["recall@10"] - d["F"]["recall@10"]
            deltas.append(delta)
            colors.append("#2ca02c" if delta >= 0 else "#d62728")

    order = sorted(range(len(deltas)), key=lambda i: deltas[i])
    labels = [labels[i] for i in order]
    deltas = [deltas[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    y = range(len(labels))
    ax.barh(list(y), deltas, color=colors, height=0.68)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel(r"$\Delta$ Recall@10 (student $-$ teacher)")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    for i, d in enumerate(deltas):
        ax.text(d + (0.006 if d >= 0 else -0.006), i, f"{d:.3f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=6.2)
    ax.set_xlim(min(deltas) - 0.05, max(deltas) + 0.05)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_strata.pdf")
    plt.close(fig)
    print("-> paper/figures/fig_strata.pdf")

# ------------------------------------------------------------------ figure 4
def fig_ingest() -> None:
    ing = load("student_context_ingest.json")
    st = ing["student_ingest"]
    prefill_cached = st["prefill_hours"]
    prefill_plain = st["prefix_cache"]["prefill_no_cache_est_hours"]
    decode = st["decode_hours"]
    teacher = ing["teacher_baseline"]["latency_sec_total_hours"]

    lat = load("latency_models_mlx.json")
    rate_embed = lat["best_embed"]["rate"]
    rate_rerank = lat["best_rerank"]["rate"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ax = axes[0]
    ax.bar([0], [teacher], color="#7f7f7f", width=0.5)
    ax.bar([1], [prefill_cached], color="#1f77b4", width=0.5)
    ax.bar([1], [decode], bottom=[prefill_cached], color="#2ca02c", width=0.5)
    # ghost: what it would be without the prefix KV cache
    ax.bar([1], [prefill_plain + decode], color="none", edgecolor="#bbbbbb",
           linestyle="--", lw=0.9, width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["API teacher", "student\n(on-device)"])
    ax.set_ylabel("Hours (3{,}134 chunks)")
    ax.set_title("Corpus-wide context generation", fontsize=8.5)
    ax.text(0, teacher + 0.15, f"{teacher:.2f}", ha="center", fontsize=6.8, color="#555555")
    ax.text(1, prefill_cached + decode + 0.15, f"{prefill_cached + decode:.2f}",
            ha="center", fontsize=6.8, color="#2ca02c")
    ax.text(1.28, prefill_plain + decode, f"{prefill_plain + decode:.2f} h\nno KV cache",
            ha="center", va="center", fontsize=6.2, color="#999999")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#1f77b4", label="prefill"),
                       Patch(facecolor="#2ca02c", label="decode"),
                       Patch(facecolor="#7f7f7f", label="teacher (API)")],
              frameon=False, fontsize=6.5, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    ax = axes[1]
    models = ["Qwen3-Embedding", "Qwen3-Reranker"]
    rates = [rate_embed, rate_rerank]
    ax.barh([1, 0], rates, color=["#1f77b4", "#d62728"], height=0.5)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(models, fontsize=7)
    ax.set_xlabel("Throughput (items/s, M2 MLX fp16)")
    ax.set_title("On-device model throughput", fontsize=8.5)
    for i, r in enumerate(rates):
        ax.text(r + 0.15, [1, 0][i], f"{r:.2f}", va="center", fontsize=7)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_ingest.pdf")
    plt.close(fig)
    print("-> paper/figures/fig_ingest.pdf")

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_quality_latency()
    fig_breakeven()
    fig_strata()
    fig_ingest()
    return 0

if __name__ == "__main__":
    sys.exit(main())
