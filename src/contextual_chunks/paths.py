"""Canonical project paths -- the single source of truth.

Every stage takes its defaults from here, so the modules can be run from any
working directory:

    .venv/bin/python -m contextual_chunks.dataset.chunks

Local caches (`CACHE`, `INDEX`) also live here; they are in .gitignore, while
model caches (MLX, GGUF) live outside the repository, in
`~/.cache/contextual-chunks` (overridden by the `CONTEXTUAL_CHUNKS_CACHE`
environment variable).
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- dataset: inputs and stages ---
DATA = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA, "raw", "dataset_raw")
DOWNLOADED_DIR = os.path.join(DATA, "raw", "downloaded_docs")
INTERIM = os.path.join(DATA, "interim")
PROCESSED = os.path.join(DATA, "processed")
PROVENANCE = os.path.join(DATA, "provenance")
LEGACY = os.path.join(DATA, "legacy")

CHUNKS = os.path.join(INTERIM, "chunks_unlabeled.jsonl")
READY = os.path.join(INTERIM, "dataset_ready_for_labeling.jsonl")
OPTIMIZED = os.path.join(INTERIM, "dataset_optimized_by_doc.jsonl")
LABELED = os.path.join(INTERIM, "labeled_dataset_metadata.jsonl")
SFT = os.path.join(PROCESSED, "train_sft_dataset.jsonl")
HF_EXPORT = os.path.join(PROCESSED, "hf_export")

STAGES = [CHUNKS, READY, OPTIMIZED, LABELED, SFT]

# --- experiment artifacts behind the paper's tables and figures ---
ARTIFACTS = os.path.join(ROOT, "artifacts")
RESULTS = os.path.join(ARTIFACTS, "results")
METRICS = os.path.join(ARTIFACTS, "metrics")
QUERIES = os.path.join(ARTIFACTS, "queries.jsonl")
STUDENT_CONTEXT = os.path.join(ARTIFACTS, "student_context.jsonl")

# --- local caches (in .gitignore) ---
CACHE = os.path.join(ROOT, ".cache")
INDEX = os.path.join(ROOT, "index")

# --- paper and notebooks ---
PAPER = os.path.join(ROOT, "paper")
NOTEBOOKS = os.path.join(ROOT, "notebooks")

# --- model caches outside the repository ---
HOME_CACHE = os.environ.get(
    "CONTEXTUAL_CHUNKS_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "contextual-chunks"),
)
MLX_CACHE = os.path.join(HOME_CACHE, "mlx")
GGUF_CACHE = os.path.join(HOME_CACHE, "gguf")
