# GGUF quantizations of the student (llama.cpp / Metal)

Metric: word-F1 vs the teacher context on 60 validation rows, greedy, `max_new_tokens=160`, prompt with `enable_thinking=False`.
bf16 SFT (MLX) reference, the same rows, recomputed from `per_row_comparison.jsonl` (the median is the mean of two middle values; previously in the training artifacts it was taken as `scores[n//2]` and was biased upward, now fixed).

| quantization | size, MB | word-F1 mean | median | p10 | empty | halluc. numbers |
|---|---:|---:|---:|---:|---:|---:|
| `f16` | 1143 | 0.526 | 0.549 | 0.323 | 0.000 | 0 |
| `q8_0` | 610 | 0.521 | 0.515 | 0.286 | 0.000 | 0 |
| `q6_k` | 472 | 0.502 | 0.500 | 0.286 | 0.000 | 0 |
| `q5_k_m` | 424 | 0.500 | 0.513 | 0.267 | 0.000 | 0 |
| `q4_k_m` | 378 | 0.484 | 0.482 | 0.263 | 0.000 | 0 |

bf16 SFT (MLX, the same 60 rows from `per_row_comparison.jsonl`): mean 0.537 / median 0.534.

## Corpus ingest: prompt prefix reuse

The prompt is `<document>` + the whole window + `</document>` + `<chunk>` + instruction (the window takes ~82% of the prompt), so neighboring chunks of one window share a prefix, and llama.cpp (`--cache-prompt`, `-np 1`) recomputes only the tail. Over the corpus the shared prefix is 81.4% (p50 per chunk 93.7%), so we must prefill 2.50M tokens of the first chunks of windows and 2.02M tokens of tails. The measurement is an ingest-order run: window by window, as in a real corpus run.

| quantization | slots | reused, % | window prefill, tok/s | tail prefill, 
tok/s | decode, tok/s | corpus ingest, h |
|---|---:|---:|---:|---:|---:|---:|
| `q8_0-ingest` | 1 | 46.7% | 604 | 475 | 43 | 4.08 |

Runs `f16`, `q6_k`, `q5_k_m`, `q4_k_m` were made in the first configuration (server without `-np 1`); their `est_ingest_hours` field is an estimate of "the whole corpus from scratch", that is an upper bound; word-F1 does not depend on the server configuration.

## Significance of differences from f16 (paired bootstrap, n=60)

| comparison | Δ word-F1 | 95% CI |
|---|---:|---|
| `q8_0` − `f16` | -0.0051 | [-0.0193, +0.0097] |
| `q6_k` − `f16` | -0.0244 | [-0.0528, +0.0020] |
| `q5_k_m` − `f16` | -0.0269 | [-0.0615, +0.0057] |
| `q4_k_m` − `f16` | -0.0426 | [-0.0683, -0.0161] |

A CI that covers zero means the loss from quantization is indistinguishable from zero on this slice.
