"""Teacher labelling: contextual-retrieval contexts for every chunk.

Teacher: `cline-pass/deepseek-v4.1-flash` (ClinePass gateway, OpenRouter behind
it) with thinking disabled -- measured $0.0001 / request, 1.8 s, ~100 output
tokens, versus $0.0015 / 28 s / 1447 reasoning tokens for qwen3.7-plus, and it
wins a blind A/B against it (see MODEL_BATTLE.md).

Cache-friendliness: rows must arrive grouped by document (run
`reorder_dataset_for_cache.py` first).  Documents are processed sequentially so
every chunk after the first of a window is a prefix-cache hit (~6.4x cheaper);
concurrency is across documents only.

Resume-safe: already labelled ids are skipped, outputs are appended to.

    .venv/bin/python -m contextual_chunks.dataset.label --limit 20          # smoke test
    .venv/bin/python -m contextual_chunks.dataset.label                     # full run
    .venv/bin/python -m contextual_chunks.dataset.label --fresh             # ignore existing outputs
"""

import os
import sys
import json
import time
import asyncio
import argparse
import collections

from contextual_chunks import paths

import httpx

AGENT_DIR = os.environ.get("PI_AGENT_DIR", os.path.expanduser("~/.pi/agent"))
MODELS_JSON = os.path.join(AGENT_DIR, "models.json")
AUTH_JSON = os.path.join(AGENT_DIR, "auth.json")

PROVIDER = "clinepass"
MODEL_NAME = os.environ.get("TEACHER_MODEL", "cline-pass/deepseek-v4.1-flash")
MAX_TOKENS = int(os.environ.get("TEACHER_MAX_TOKENS", "300"))
CONCURRENCY = int(os.environ.get("TEACHER_CONCURRENCY", "4"))


def load_credentials():
    base_url = os.environ.get("CLINEPASS_BASE_URL")
    api_key = os.environ.get("CLINEPASS_API_KEY")
    headers = {}
    if os.path.exists(MODELS_JSON):
        cfg = json.load(open(MODELS_JSON))
        prov = cfg.get("providers", {}).get(PROVIDER, {})
        base_url = base_url or prov.get("baseUrl")
        api_key = api_key or prov.get("apiKey")
        headers = dict(prov.get("headers") or {})
    if not api_key and os.path.exists(AUTH_JSON):
        auth = json.load(open(AUTH_JSON))
        entry = auth.get(PROVIDER)
        if isinstance(entry, dict):
            api_key = entry.get("key")
    if not base_url or not api_key:
        sys.exit(f"no credentials for {PROVIDER}: set CLINEPASS_BASE_URL/CLINEPASS_API_KEY "
                 f"or fill {MODELS_JSON} + {AUTH_JSON}")
    return base_url, api_key, headers


USAGE_TOTALS = collections.Counter()


async def call_teacher(client, prompt, retries=5):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "reasoning": {"enabled": False},   # -> reasoning_tokens == 0
    }
    last = "unknown"
    for attempt in range(retries):
        t0 = time.time()
        try:
            resp = await client.post("/chat/completions", json=payload, timeout=120.0)
            if resp.status_code == 200:
                data = resp.json().get("data", resp.json())
                choice = data["choices"][0]
                usage = data.get("usage", {}) or {}
                details = usage.get("completion_tokens_details") or {}
                meta = {
                    # NB: `prompt_tokens`/`chunk_tokens` are the pipeline's own
                    # (tokenizer-measured) fields -- API usage is namespaced.
                    "usage_prompt_tokens": usage.get("prompt_tokens"),
                    "usage_cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "usage_completion_tokens": usage.get("completion_tokens"),
                    "usage_reasoning_tokens": details.get("reasoning_tokens"),
                    "cost_usd": usage.get("cost"),
                    "finish_reason": choice.get("finish_reason"),
                    "latency_sec": round(time.time() - t0, 2),
                }
                return choice["message"]["content"].strip(), meta
            last = f"HTTP{resp.status_code}"
            if resp.status_code == 429:
                await asyncio.sleep(4 * (attempt + 1))
            elif resp.status_code >= 500:
                await asyncio.sleep(2 * (attempt + 1))
            else:
                return None, {"error": f"{last}: {resp.text[:200]}"}
        except Exception as exc:                      # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(2 * (attempt + 1))
    return None, {"error": last}


async def process_document(doc_id, chunks, client, lock, sft_f, meta_f, state):
    for rec in chunks:
        context, meta = await call_teacher(client, rec["prompt"])
        async with lock:
            state["done"] += 1
            i, n = state["done"], state["total"]
            if not context:
                state["failed"] += 1
                print(f"[{i}/{n}] FAIL  {doc_id[:32]} chunk {rec.get('doc_chunk_index', 0)+1}"
                      f"  {meta.get('error','')[:90]}", flush=True)
                continue
            if meta.get("finish_reason") == "length":
                state["truncated"] += 1
            if meta.get("usage_reasoning_tokens"):
                state["reasoning_not_off"] += 1
            for k in ("usage_prompt_tokens", "usage_cached_tokens",
                      "usage_completion_tokens", "usage_reasoning_tokens"):
                USAGE_TOTALS[k] += meta.get(k) or 0
            USAGE_TOTALS["cost_usd"] += meta.get("cost_usd") or 0.0

            sft_f.write(json.dumps({
                "id": rec["id"],
                "doc_hash": rec.get("doc_hash"),
                "category": rec.get("category"),
                "messages": [
                    {"role": "user", "content": rec["prompt"]},
                    {"role": "assistant", "content": context},
                ],
            }, ensure_ascii=False) + "\n")
            sft_f.flush()

            entry = dict(rec)
            entry["generated_context"] = context
            entry.update(meta)
            meta_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            meta_f.flush()

            cached = meta.get("usage_cached_tokens") or 0
            pt = meta.get("usage_prompt_tokens") or 1
            print(f"[{i}/{n}] {doc_id[:28]} chunk {rec.get('doc_chunk_index',0)+1:>3}"
                  f" | cache {cached}/{pt} | {meta['latency_sec']}s | ${(meta.get('cost_usd') or 0):.6f}"
                  f" | {context[:60]}...", flush=True)
        await asyncio.sleep(0.05)


async def label(input_file, sft_file, meta_file, start, limit, concurrency, fresh):
    rows = []
    for idx, line in enumerate(open(input_file, encoding="utf-8")):
        if idx < start:
            continue
        if limit and len(rows) >= limit:
            break
        rows.append(json.loads(line))

    if fresh:
        for path in (sft_file, meta_file):
            if os.path.exists(path):
                os.remove(path)
                print(f"[fresh] removed {path}")

    done_ids = set()
    for path in (meta_file,):
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:                      # noqa: BLE001
                    pass

    todo = [r for r in rows if r["id"] not in done_ids]
    docs = collections.OrderedDict()
    for rec in todo:
        docs.setdefault(rec["doc_id"], []).append(rec)

    print(f"=== TEACHER LABELLING ===")
    print(f"model:        {MODEL_NAME}  (reasoning off, max_tokens={MAX_TOKENS})")
    print(f"rows in slice:{len(rows)}   already labelled: {len(rows) - len(todo)}   todo: {len(todo)}")
    print(f"documents:    {len(docs)}   concurrency: {concurrency}")
    if not todo:
        print("nothing to do.")
        return

    base_url, api_key, headers = load_credentials()
    headers = {**headers, "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    state = {"done": 0, "total": len(todo), "failed": 0, "truncated": 0, "reasoning_not_off": 0}

    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)
    mode = "a" if done_ids else "w"
    t0 = time.time()
    async with httpx.AsyncClient(base_url=base_url, headers=headers) as client:
        with open(sft_file, mode, encoding="utf-8") as sft_f, \
             open(meta_file, mode, encoding="utf-8") as meta_f:
            async def worker(doc_id, chunks):
                async with sem:
                    await process_document(doc_id, chunks, client, lock, sft_f, meta_f, state)

            await asyncio.gather(*(worker(d, c) for d, c in docs.items()))

    elapsed = time.time() - t0
    pt = USAGE_TOTALS["usage_prompt_tokens"]
    print(f"\n=== DONE in {elapsed/60:.1f} min ===")
    print(f"labelled:          {state['done'] - state['failed']}/{state['total']}")
    print(f"failed:            {state['failed']}")
    print(f"finish_reason=length (truncated): {state['truncated']}")
    print(f"reasoning_tokens != 0 (should be 0): {state['reasoning_not_off']}")
    print(f"prompt tokens:     {pt}  (cached {USAGE_TOTALS['usage_cached_tokens']} = "
          f"{USAGE_TOTALS['usage_cached_tokens'] / pt * 100:.1f}%)" if pt else "")
    print(f"completion tokens: {USAGE_TOTALS['usage_completion_tokens']}")
    print(f"cost:              ${USAGE_TOTALS['cost_usd']:.4f}  "
          f"(${USAGE_TOTALS['cost_usd'] / max(state['done'] - state['failed'], 1) * 1000:.2f} / 1k rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=paths.OPTIMIZED)
    ap.add_argument("--sft", default=paths.SFT)
    ap.add_argument("--meta", default=paths.LABELED)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--fresh", action="store_true", help="delete existing outputs first")
    a = ap.parse_args()
    asyncio.run(label(a.inp, a.sft, a.meta, a.start, a.limit, a.concurrency, a.fresh))


if __name__ == "__main__":
    main()
