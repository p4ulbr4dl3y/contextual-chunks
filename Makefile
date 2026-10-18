# contextual-chunks - reranker-free contextual retrieval on the edge
#
# Every stage is a module of the `contextual_chunks` package. Paths come from
# `src/contextual_chunks/paths.py`, so modules run from any directory.
# `make install` sets up .venv with the extras the targets below expect.
PY ?= .venv/bin/python
GGUF_DIR ?= $(HOME)/.cache/contextual-chunks/gguf

.PHONY: help install pipeline chunks prompts reorder audit label dedup export \
        train-smoke export-mlx convert-mlx check-models test \
        index eval-queries eval-student index-student index-full smoke-retrieval \
        student-context queries audit-queries bench-models bench-models-mlx ablate \
        breakeven rerank-pool breakdown bm25-backends eval-strat60 paired-stats jaccard \
        gguf-convert gguf-eval gguf-ingest gguf-metric-check gguf-verify gguf-parity \
        paper

help: ## show the targets
	@grep -E '^[a-z][a-z-]*:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

install: ## create .venv and install the package with the extras the targets need
	uv venv
	uv pip install -e ".[retrieval,mlx,paper,dev]"

# --- dataset --------------------------------------------------------------

chunks: ## raw (data/raw/dataset_raw) -> data/interim/chunks_unlabeled.jsonl
	$(PY) -m contextual_chunks.dataset.chunks

prompts: ## chunks -> data/interim/dataset_ready_for_labeling.jsonl (prompt + metrics)
	$(PY) -m contextual_chunks.dataset.validate

reorder: ## prompts -> data/interim/dataset_optimized_by_doc.jsonl (prefix-cache locality)
	$(PY) -m contextual_chunks.dataset.reorder

label: ## teacher -> data/processed/train_sft_dataset.jsonl + data/interim/labeled_dataset_metadata.jsonl
	$(PY) -m contextual_chunks.dataset.label

dedup: ## drop duplicate prompts (metadata + SFT)
	$(PY) -m contextual_chunks.dataset.dedup

audit: ## invariants of every stage, id agreement across stages (exit 1 on violation)
	$(PY) -m contextual_chunks.dataset.audit --all

export: ## train_sft -> data/processed/hf_export/ (parquet + card)
	$(PY) -m contextual_chunks.dataset.export_hf

pipeline: chunks prompts reorder audit label dedup export ## run the whole dataset pipeline

# --- training and model export -------------------------------------------

train-smoke: ## local trainer smoke test (without gradient checkpointing)
	$(PY) -m contextual_chunks.train.sft --model Qwen/Qwen3-0.6B --no-grad-ckpt

export-mlx: ## LoRA -> merge into Qwen3-0.6B -> MLX bf16 (~/.cache/contextual-chunks/mlx)
	$(PY) -m contextual_chunks.train.export_mlx --verify --cleanup

convert-mlx: ## HF -> MLX cache in fp16 (once; speeds up start-up and inference)
	$(PY) -m contextual_chunks.retrieval.mlx_convert

# --- gates ---------------------------------------------------------------

check-models: ## gate: the Qwen3-Embedding/Reranker wrappers match the model cards
	$(PY) -m contextual_chunks.retrieval.check_models

test: ## unit tests of the query set (no network)
	$(PY) -m pytest tests -q

# --- retrieval harness ---------------------------------------------------

index: ## build the indexes of the paper arms (A/B => 2 indexes: flat and context+chunk)
	$(PY) -m contextual_chunks.retrieval.build_index --arms A B

index-full: ## full-corpus indexes for arms A/B
	$(PY) -m contextual_chunks.retrieval.build_index --arms A B --tag full

index-student: ## indexes for G/H (student context instead of the teacher one)
	$(PY) -m contextual_chunks.retrieval.build_index --arms G H --tag full

eval-queries: ## run the query set over the arms (needs: make index)
	$(PY) -m contextual_chunks.retrieval.eval_queries --arms A B C D --tag full

eval-student: ## teacher (A/B/E/F) vs student (G/H) on the query set
	$(PY) -m contextual_chunks.retrieval.eval_queries --arms A B E F --tag full
	$(PY) -m contextual_chunks.retrieval.eval_queries --arms G H --tag full \
		--context-source student --out artifacts/results/retrieval_eval_student.md
	$(PY) -m contextual_chunks.queries.context_sources

eval-strat60: ## stratified sample of 60 queries (15 per type), all 8 arms
	$(PY) -m contextual_chunks.retrieval.eval_queries --arms A B E F G H --tag full --stratified 60 --out artifacts/results/retrieval_eval_strat60_sub.md
	$(PY) -m contextual_chunks.retrieval.eval_queries --arms C D --tag full --stratified 60 --out artifacts/results/retrieval_eval_strat60_rr.md
	$(PY) -m contextual_chunks.queries.merge_reports --parts artifacts/results/retrieval_eval_strat60_sub.json artifacts/results/retrieval_eval_strat60_rr.json --out artifacts/results/retrieval_eval_strat60.json

smoke-retrieval: ## harness smoke test: indexes on a subsample + arms A/B/C/D
	$(PY) -m contextual_chunks.retrieval.smoke_search --limit 300 --arms A B C D

student-context: ## generate the student context over the whole corpus (MLX bf16, ~4 h)
	$(PY) -m contextual_chunks.latency.gen_student_context --log-every 200

ablate: ## retrieval ablations without rebuilding the index (MRL, instruction, RRF, BM25)
	$(PY) -m contextual_chunks.queries.ablate --arm F
	$(PY) -m contextual_chunks.queries.ablate --arm G --out-suffix student

rerank-pool: ## rerank pool ablation (16/32/64) on 60 queries
	$(PY) -m contextual_chunks.retrieval.eval_queries --arms C --tag full --limit 60 --rerank-pool 16 --out artifacts/results/retrieval_eval_rerank_pool16.md
	$(PY) -m contextual_chunks.retrieval.eval_queries --arms C --tag full --limit 60 --rerank-pool 32 --out artifacts/results/retrieval_eval_rerank_pool32.md
	$(PY) -m contextual_chunks.retrieval.eval_queries --arms C --tag full --limit 60 --rerank-pool 64 --out artifacts/results/retrieval_eval_rerank_pool64.md

breakdown: ## metrics by stratum (query type / category / table-ness), F vs G
	$(PY) -m contextual_chunks.queries.breakdown

bm25-backends: ## BM25 backends: rank_bm25 vs fastembed Qdrant/bm25
	$(PY) -m contextual_chunks.retrieval.build_index --arms A B --tag bm25fe --bm25 fastembed
	$(PY) -m contextual_chunks.retrieval.eval_queries --arms A B --tag bm25fe --bm25 fastembed --out artifacts/results/retrieval_eval_bm25_fastembed.md
	$(PY) -m contextual_chunks.queries.bm25_backends

paired-stats: ## paired bootstrap over per-query metrics (F vs G, F vs C)
	$(PY) -m contextual_chunks.queries.paired_stats --a artifacts/results/retrieval_eval_full.json --pairs F:G G:F --out artifacts/results/paired_stats_full.json
	$(PY) -m contextual_chunks.queries.paired_stats --a artifacts/results/retrieval_eval_strat60.json --pairs F:C E:B B:D G:F --out artifacts/results/paired_stats_strat60.json

jaccard: ## median Jaccard of neighbouring chunks (Section 3 of the paper)
	$(PY) -m contextual_chunks.queries.jaccard

# --- query set -----------------------------------------------------------

queries: ## build the query set with gold chunks (LLM + judge + filters)
	$(PY) -m contextual_chunks.queries.build --docs all --per-doc 4 --stage all --concurrency 6

audit-queries: ## query-set invariants + BM25-only calibration
	$(PY) -m contextual_chunks.queries.audit --baseline

# --- benchmarks ----------------------------------------------------------

bench-models: ## inference benchmark (torch: CPU or MPS) -> artifacts/results/latency_models.md
	$(PY) -m contextual_chunks.latency.bench_models --backend torch

bench-models-mlx: ## inference benchmark on the Apple GPU through MLX fp16
	$(PY) -m contextual_chunks.latency.bench_models --backend mlx

breakeven: ## ingest amortization model: T(D, Q) = D*t_ingest + Q*t_query
	$(PY) -m contextual_chunks.latency.breakeven

# --- GGUF / llama.cpp (the second edge runtime next to MLX) ---------------
# The converter comes from llama.cpp pinned by tag b10180 (the brew package version).

GGUF_QUANTS ?= q8_0 q6_k q5_k_m q4_k_m
GGUF_INGEST_QUANT ?= q8_0

gguf-convert: ## student (MLX bf16) -> GGUF f16 -> quantizations in ~/.cache/contextual-chunks/gguf
	$(PY) -m contextual_chunks.gguf.convert

gguf-eval: ## word-F1 of the quantizations on 60 validation rows -> artifacts/results/gguf_quant.md
	@for q in $(GGUF_QUANTS); do \
		f=$(GGUF_DIR)/qwen3-0.6b-contextual-chunks-$$q.gguf; \
		if [ ! -f "$$f" ]; then echo "no $$f - skipping (run make gguf-convert)"; continue; fi; \
		$(PY) -m contextual_chunks.gguf.eval --gguf $$f --label $$q || exit 1; \
	done
	$(PY) -m contextual_chunks.gguf.eval --summarize

gguf-ingest: ## corpus ingest measurement (window by window, -np 1, --cache-prompt)
	$(PY) -m contextual_chunks.gguf.eval \
		--gguf $(GGUF_DIR)/qwen3-0.6b-contextual-chunks-$(GGUF_INGEST_QUANT).gguf \
		--label $(GGUF_INGEST_QUANT)-ingest --order ingest
	$(PY) -m contextual_chunks.gguf.eval --summarize

gguf-metric-check: ## self-check of the word-F1 metric (no network, no models)
	$(PY) -m contextual_chunks.gguf.eval --metric-check

gguf-verify: ## artefact gates: template, prompt path, tokens, prefix reuse
	$(PY) -m contextual_chunks.gguf.verify

gguf-parity: ## token-by-token comparison of the quantizations against the bf16 reference
	$(PY) -m contextual_chunks.gguf.parity

# --- paper ---------------------------------------------------------------
# Tables and figures are generated from artifacts/results, then tectonic builds
# the PDF. If the TeX Live bundle is not cached, point BUNDLE at a local server:
#   make paper BUNDLE=http://127.0.0.1:8793/bundle.tar
BUNDLE ?=

paper: ## build paper/main.pdf (tables + figures + tectonic)
	$(PY) -m contextual_chunks.paper.tables
	$(PY) -m contextual_chunks.paper.figures
	cd paper && tectonic $(if $(BUNDLE),-b $(BUNDLE),) main.tex
	@echo "-> paper/main.pdf"
