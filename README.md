# contextual-chunks — Reranker-Free Contextual Retrieval on the Edge

[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-ru--contextual--chunks-yellow)](https://huggingface.co/datasets/p4ulbr4dl3y/ru-contextual-chunks)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20model-qwen3--0.6b--contextual--chunks-yellow)](https://huggingface.co/p4ulbr4dl3y/qwen3-0.6b-contextual-chunks)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](pyproject.toml)

Contextual retrieval prepends a short document-level summary to every chunk
before indexing, and is normally paired with a cross-encoder reranker. This
repository asks whether the reranker is necessary once chunks are contextualized,
and whether the contextual enrichment itself can run on a laptop.

The answer, on a corpus of 82 Russian official documents (3,134 chunks): a
single dense index of contextualized chunks comes within 0.05 recall@10 of the
same index with a 0.6B cross-encoder, a difference whose 95% paired bootstrap
interval includes zero, at **1/1,322 of the query latency**; the API teacher can
be distilled into a **0.6B local model** that regenerates the whole corpus on the
device in 4.49 h at zero marginal cost.

The full study is the technical report in [`paper/`](paper) (`make paper` builds
the PDF); the code, the query set and every artefact behind its tables and
figures are here.

## Results

**Retrieval quality on the full set** (M2 GPU, MLX fp16, 234 gold-chunk queries,
3,134 chunks). `R@10` is recall@10.

| Arm | Index text | Query path | R@10 | MRR@10 |
|---|---|---|---|---|
| A | flat chunk | dense + BM25 + RRF | 0.466 | 0.207 |
| B | context ∥ chunk | dense + BM25 + RRF | 0.530 | 0.288 |
| E | flat chunk | dense only | 0.517 | 0.259 |
| **F** | **context ∥ chunk** | **dense only** | **0.658** | **0.328** |
| G | distilled context ∥ chunk | dense only | 0.620 | 0.342 |
| H | distilled context ∥ chunk | dense + BM25 + RRF | 0.530 | 0.288 |

**All eight arms on one stratified subset of sixty queries** (fifteen per query
type, all five categories), `p50` is median end-to-end query latency.

| Arm | Index text | Query path | R@10 | MRR@10 | p50 |
|---|---|---|---|---|---|
| A | flat chunk | dense + BM25 + RRF | 0.400 | 0.171 | 44 ms |
| B | context ∥ chunk | dense + BM25 + RRF | 0.450 | 0.233 | 45 ms |
| E | flat chunk | dense only | 0.467 | 0.177 | 39 ms |
| F | context ∥ chunk | dense only | 0.550 | 0.265 | 40 ms |
| **G** | **distilled context ∥ chunk** | **dense only** | **0.600** | **0.327** | **41 ms** |
| H | distilled context ∥ chunk | dense + BM25 + RRF | 0.450 | 0.237 | 46 ms |
| C | context ∥ chunk | dense + BM25 + RRF + rerank | 0.650 | 0.339 | 54,732 ms |
| D | flat chunk | dense + BM25 + RRF + rerank | 0.467 | 0.238 | 43,383 ms |

The headline result is the comparison of G and C on that identical subset: 0.600
against 0.650 recall@10, with a 95% paired bootstrap interval of [-0.150, 0.050],
at 41 ms against 54.7 s. The reranker cannot compensate for a flat index either:
arm D matches arm E, which runs in 39 ms.

**Distillation** (60 held-out chunks, word-F1 against the teacher context):

| metric | base `Qwen3-0.6B` | after LoRA SFT |
|---|---|---|
| word-F1 mean | 0.189 | **0.537** |
| word-F1 median | 0.176 | 0.534 |
| word-F1 p10 | 0.064 | 0.319 |
| empty responses | 1.7% | **0%** |
| hallucinated numbers | 0 | **0** |

**Ingest cost** (3,134 chunks): API teacher 1.85 h and USD 0.82; on-device
student 4.49 h and USD 0, with no document leaving the machine.

## Repository map

```
src/contextual_chunks/
  paths.py           canonical paths (single source of truth)
  dataset/           OCR chunking, prompts, audit, teacher labelling, HF export
  ingest/            raw-document acquisition from zakupki.gov.ru (provenance)
  train/             QLoRA SFT and the MLX merge/export
  retrieval/         Qwen3 embedding and reranking, BM25, RRF, local Qdrant
  queries/           query-set construction, ablations, stratified analysis
  latency/           model and ingest benchmarks, break-even model
  gguf/              llama.cpp conversion, quantized evaluation, parity gates
  paper/             table and figure generation for the report
artifacts/           every JSON the paper's tables and figures read
  queries.jsonl         the frozen 234-query set with gold chunks
  student_context.jsonl on-device contexts for the whole corpus (G/H arms)
paper/               LaTeX sources, generated tables and figures
notebooks/           the Colab training run
tests/               query-set unit tests
Makefile             make help — every stage
```

## Published artefacts

- dataset — [`p4ulbr4dl3y/ru-contextual-chunks`](https://huggingface.co/datasets/p4ulbr4dl3y/ru-contextual-chunks)
  (3,134 labeled chunks, train 2,701 / validation 433 split by `window_hash`);
- model — [`p4ulbr4dl3y/qwen3-0.6b-contextual-chunks`](https://huggingface.co/p4ulbr4dl3y/qwen3-0.6b-contextual-chunks)
  (merged bf16 weights, the LoRA adapter under `lora/`, `Q8_0` GGUF under `gguf/`);
- checkpoints — [`p4ulbr4dl3y/qwen3-0.6b-contextual-chunks-checkpoints`](https://huggingface.co/p4ulbr4dl3y/qwen3-0.6b-contextual-chunks-checkpoints).

## Quickstart

```bash
make install          # uv venv + uv pip install -e ".[retrieval,mlx,paper,dev]"
make help             # list every target
make test             # query-set unit tests, no network
```

The dataset pipeline reads OCR output from `data/raw/dataset_raw` and writes the
intermediate stages; the teacher step calls an API and needs credentials.

```bash
make chunks prompts reorder audit label dedup export   # full dataset pipeline
make train-smoke                                       # LoRA SFT smoke test
make export-mlx convert-mlx check-models               # models for the harness
make index eval-queries                                # build the index, run the query set
make bench-models-mlx                                  # MLX throughput benchmarks
make gguf-convert gguf-eval                            # llama.cpp GGUF + quant ablation
make paper                                             # tables, figures, PDF
```

## Pipeline

| Stage | Command | Output |
|---|---|---|
| OCR and chunking | `make chunks` | `data/interim/chunks_unlabeled.jsonl` |
| Prompt building | `make prompts` | `dataset_ready_for_labeling.jsonl` |
| Cache ordering | `make reorder` | `dataset_optimized_by_doc.jsonl` |
| Invariants | `make audit` | exit 1 on any violation |
| Teacher labelling | `make label` | `train_sft_dataset.jsonl` |
| Deduplication | `make dedup` | deduplicated prompts |
| HF export | `make export` | `data/processed/hf_export/` |

The teacher is `cline-pass/deepseek-v4.1-flash` with reasoning disabled,
`temperature=0`. Labelling is resume-safe and doc-contiguous so that every chunk
after the first in a window hits the prefix cache: 79.3% of prompt tokens came
from cache, which brought the whole corpus to USD 0.82.

`make audit` verifies nine invariants on every stage, including that each chunk
is an exact substring of the document window in its own prompt, that no prompt is
duplicated, and that identifiers agree across stages.

## Training

`contextual_chunks.train.sft` fine-tunes `Qwen/Qwen3-0.6B` with QLoRA (4-bit
base, rank 16, alpha 16, 1.665% of parameters), response-only loss, a group split
by `window_hash`, maximum sequence length 16,384 tokens:

```bash
.venv/bin/python -m contextual_chunks.train.sft --model Qwen/Qwen3-0.6B --no-grad-ckpt
```

Two epochs (676 steps) took the mean word-F1 from 0.189 to 0.537. The published
run, with its outputs, is `notebooks/sft_final.ipynb`.

## Citation

```bibtex
@misc{surovtsev2026chunks,
  title  = {Reranker-Free Edge Retrieval: Distilling Contextual Chunk
            Enrichment into a 0.6B Model},
  author = {Surovtsev, Egor},
  year   = {2026},
  url    = {https://github.com/p4ulbr4dl3y/contextual-chunks}
}
```

## License

Code is released under the MIT License ([`LICENSE`](LICENSE)). The dataset and
the paper are released under CC BY 4.0.
