"""Teacher client for the experiments (ClinePass gateway, same credentials as the pipeline).

Difference from `src/contextual_chunks/dataset/label.py`: the model is chosen per call
(two generators are needed so the set has no style shift), there is JSON parsing
and a shared cost counter.

    t = Teacher()
    async with t:
        out, meta = await t.ask_json("deepseek", prompt)
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

AGENT_DIR = os.environ.get("PI_AGENT_DIR", os.path.expanduser("~/.pi/agent"))
MODELS_JSON = os.path.join(AGENT_DIR, "models.json")
AUTH_JSON = os.path.join(AGENT_DIR, "auth.json")
PROVIDER = "clinepass"

# aliases: two generators (one paid + one free from the provider), against style shift
MODELS = {
    "deepseek": os.environ.get("GEN_MODEL_A", "cline-pass/deepseek-v4.1-flash"),
    "mimo": os.environ.get("GEN_MODEL_B", "cline-pass/mimo-v2.5-pro"),
    "qwen": os.environ.get("GEN_MODEL_C", "cline-pass/qwen3.7-plus"),
}
# `deepseek-v4.1-flash` with `reasoning: off` returns EMPTY content on a complex prompt
# (HTTP500 "empty response content"): the model wants to think, and thinking was forbidden.
# With reasoning enabled it answers normally -> 764 completion / 709 reasoning, $0.00048.
# `mimo-v2.5-pro` answers without reasoning and is free ($0), but quotes identifiers more often.
REASONING = {"deepseek": True, "mimo": False, "qwen": False}
MAX_TOKENS = {"deepseek": 2000, "mimo": 600, "qwen": 600}
FALLBACK = {"deepseek": "mimo", "mimo": "qwen", "qwen": "mimo"}
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_credentials() -> tuple[str, str, dict]:
    base_url = os.environ.get("CLINEPASS_BASE_URL")
    api_key = os.environ.get("CLINEPASS_API_KEY")
    headers: dict = {}
    if os.path.exists(MODELS_JSON):
        prov = json.load(open(MODELS_JSON)).get("providers", {}).get(PROVIDER, {})
        base_url = base_url or prov.get("baseUrl")
        api_key = api_key or prov.get("apiKey")
        headers = dict(prov.get("headers") or {})
    if not api_key and os.path.exists(AUTH_JSON):
        entry = json.load(open(AUTH_JSON)).get(PROVIDER)
        if isinstance(entry, dict):
            api_key = entry.get("key")
    if not base_url or not api_key:
        sys.exit(f"no credentials for {PROVIDER}: fill in {MODELS_JSON} / {AUTH_JSON}")
    return base_url, api_key, headers


def parse_json(text: str) -> dict | None:
    """Tolerant parsing: ```json ...```, extra text around, broken quotes."""
    if not text:
        return None
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\n?|```$", "", body).strip()
    try:
        out = json.loads(body)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        pass
    m = JSON_RE.search(body)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


class Teacher:
    def __init__(self, concurrency: int = 4, max_tokens: int = 500) -> None:
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.calls = 0
        self.cost = 0.0
        self.usage: collections.Counter = collections.Counter()
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Teacher:
        base_url, api_key, headers = load_credentials()
        headers = {"Authorization": f"Bearer {api_key}", **headers}
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=180.0)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def ask(self, model: str, prompt: str, retries: int = 5) -> tuple[str | None, dict]:
        payload = {"model": MODELS.get(model, model), "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0,
                   "max_tokens": MAX_TOKENS.get(model, self.max_tokens),
                   "reasoning": {"enabled": REASONING.get(model, False)}}
        last = "unknown"
        for attempt in range(retries):
            t0 = time.time()
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                if resp.status_code == 200:
                    data = resp.json().get("data", resp.json())
                    choice = data["choices"][0]
                    usage = data.get("usage", {}) or {}
                    self.calls += 1
                    self.cost += float(usage.get("cost") or 0.0)
                    self.usage["prompt_tokens"] += usage.get("prompt_tokens") or 0
                    self.usage["cached_tokens"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                    self.usage["completion_tokens"] += usage.get("completion_tokens") or 0
                    meta = {"model": payload["model"], "cost_usd": usage.get("cost"),
                            "finish_reason": choice.get("finish_reason"),
                            "latency_sec": round(time.time() - t0, 2)}
                    return choice["message"]["content"].strip(), meta
                last = f"HTTP{resp.status_code}: {resp.text[:160]}"
                if resp.status_code in (429,) or resp.status_code >= 500:
                    await asyncio.sleep(3 * (attempt + 1))
                else:
                    return None, {"error": last}
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(2 * (attempt + 1))
        return None, {"error": last}

    async def ask_json(self, model: str, prompt: str, tries: int = 3,
                       fallback: bool = True) -> tuple[dict | None, dict]:
        """Like `ask`, but requires valid JSON; on failure tries the second model."""
        models = [model] + ([FALLBACK[model]] if fallback and model in FALLBACK else [])
        meta: dict = {}
        for m in models:
            for _ in range(tries):
                text, meta = await self.ask(m, prompt)
                if text is None:
                    break                      # empty response, this model cannot, move to the next
                out = parse_json(text)
                if out is not None:
                    return out, {**meta, "used": m, "raw": text}
                meta = {**meta, "error": f"not JSON: {text[:120]}"}
        return None, meta

    async def map_docs(self, jobs: list[tuple[str, Callable[[], Awaitable[Any]]]]) -> list[Any]:
        """Concurrency across documents, order within a document is sequential."""
        sem = asyncio.Semaphore(self.concurrency)
        results: list[Any] = [None] * len(jobs)

        async def run(i: int, fn: Callable[[], Awaitable[Any]]) -> None:
            async with sem:
                results[i] = await fn()

        await asyncio.gather(*(run(i, fn) for i, (_, fn) in enumerate(jobs)))
        return results
