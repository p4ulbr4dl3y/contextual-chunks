"""Gates of the student GGUF artifact: does llama.cpp see exactly the prompt from training.

Why this is separate from `eval_gguf.py`. That one sends **ready text** to
`llama-server` in raw-completion mode, so the template from the GGUF is not applied at
all. So its results say nothing about two things production depends on:

* whether the tokenization of the prompt text matches transformers/MLX (the prompt here
  is up to 9k tokens of Cyrillic, and `tokenizer.ggml.pre` is a name the converter
  guesses from the fingerprint; an error in it silently breaks the prompt);
* whether the prompt llama.cpp builds from its chat template matches what HF built
  (`enable_thinking=False`). The student template adds
  `<think>\\n\\n</think>\\n\\n` **only if the kwarg is passed explicitly**, so a server
  without `--chat-template-kwargs` builds a prompt the model never saw.

    .venv/bin/python -m contextual_chunks.gguf.verify
    .venv/bin/python -m contextual_chunks.gguf.verify --rows 5 --no-sweep

Exit code: 0 means all gates passed, 1 means a failure (every failure is printed).
Result: `artifacts/results/gguf_verify.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import urllib.request
from pathlib import Path

from contextual_chunks import paths
from contextual_chunks.gguf.eval import (
    MLX_MODEL,
    RESULTS,
    load_rows,
    render_prompts,
    start_server,
)

ROOT = Path(paths.ROOT)
GGUF_DEFAULT = Path(paths.GGUF_CACHE) / "qwen3-0.6b-contextual-chunks-q8_0.gguf"
SERVER_DEFAULT = "/opt/homebrew/bin/llama-server"
SNAPSHOT = Path(paths.MLX_CACHE) / "qwen3-0.6b-contextual-chunks-bfloat16"
PRE_CANDIDATES = ("qwen2", "qwen35", "llama-bpe", "gpt-4o", "default")
SHIPPED_PRE = "qwen2"  # this is the name the converter wrote; matches the student regex
THINK_BLOCK = "<think>\n\n</think>\n\n"

def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

def get_json(port: int, path: str, body: dict | None = None, timeout: float = 600.0):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())

def tokenize(port: int, text: str) -> list[int]:
    return [int(t) for t in get_json(port, "/tokenize", {"content": text, "add_special": False})["tokens"]]

def render_server(port: int, prompt_raw: str, kwargs: dict | None = None) -> str:
    body: dict = {"messages": [{"role": "user", "content": prompt_raw}]}
    if kwargs is not None:
        body["chat_template_kwargs"] = kwargs
    return get_json(port, "/apply-template", body)["prompt"]

def tag(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, default=GGUF_DEFAULT)
    parser.add_argument("--server", default=SERVER_DEFAULT)
    parser.add_argument("--rows", type=int, default=0, help="how many rows to take (0 means all 60)")
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--expect-quant", default="Q8_0")
    parser.add_argument("--no-sweep", action="store_true", help="do not sweep tokenizer.ggml.pre")
    args = parser.parse_args()

    if not args.gguf.exists():
        print(f"no file: {args.gguf}")
        return 1

    rows = render_prompts(load_rows(args.rows))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MLX_MODEL))
    prompts = [r["prompt"] for r in rows]
    expected = [tok.encode(p, add_special_tokens=False) for p in prompts]
    ctx = max(len(e) for e in expected) + 256
    print(f"rows {len(rows)}, longest prompt {max(len(e) for e in expected)} tokens, ctx {ctx}")
    print(f"GGUF: {args.gguf.name} ({args.gguf.stat().st_size / 1024**2:.1f} MB)\n")

    failures: list[str] = []
    report: dict = {"gguf": str(args.gguf), "rows": len(rows), "gates": {}}
    port = free_port()
    logfile = args.gguf.parent / "logs" / f"verify_{args.gguf.stem}.log"
    proc = start_server(args.server, args.gguf, port, ctx, args.ngl, logfile)
    try:
        props = get_json(port, "/props")

        # G1 · template in the GGUF == snapshot template (byte for byte)
        local = (SNAPSHOT / "chat_template.jinja").read_text(encoding="utf-8")
        same_template = props.get("chat_template") == local
        report["gates"]["chat_template_identical"] = {
            "ok": same_template, "snapshot_sha": tag(local), "gguf_sha": tag(props.get("chat_template") or ""),
            "chars": len(props.get("chat_template") or ""),
        }
        print(f"[{'OK ' if same_template else 'FAIL'}] G1 GGUF template == snapshot: "
              f"{tag(local)} vs {tag(props.get('chat_template') or '')} ({len(local)} characters)")
        if not same_template:
            failures.append("the template in the GGUF differs from the snapshot")

        # G2 · production-path prompt (chat endpoint) == training prompt
        probes = rows[:3]
        gguf_prompts = [render_server(port, r["prompt_raw"], {"enable_thinking": False}) for r in probes]
        prompt_ok = all(g == r["prompt"] for g, r in zip(gguf_prompts, probes, strict=True))
        report["gates"]["apply_template_matches_training"] = {
            "ok": prompt_ok, "checked": len(probes),
            "gguf_sha": tag(gguf_prompts[0]), "hf_sha": tag(probes[0]["prompt"]),
        }
        print(f"[{'OK ' if prompt_ok else 'FAIL'}] G2 /apply-template(enable_thinking=false) == HF render: "
              f"{tag(gguf_prompts[0])} vs {tag(probes[0]['prompt'])} ({len(probes)} rows)")
        if not prompt_ok:
            failures.append("the server chat template gives a different prompt than training")

        # G3 · negative control: without kwargs the prompt is DIFFERENT (and has no empty think block)
        bare = [render_server(port, r["prompt_raw"]) for r in probes]
        control_ok = all(b != r["prompt"] and THINK_BLOCK not in b for b, r in zip(bare, probes, strict=True))
        report["gates"]["negative_control_without_kwargs"] = {
            "ok": control_ok, "differs": [b != r["prompt"] for b, r in zip(bare, probes, strict=True)],
            "think_block_present": [THINK_BLOCK in b for b in bare],
        }
        print(f"[{'OK ' if control_ok else 'FAIL'}] G3 control: without kwarg the prompt differs "
              f"and carries no empty think (think found: {[THINK_BLOCK in b for b in bare]})")
        if not control_ok:
            failures.append("the negative control does not distinguish kwargs")

        # G4 · tokens: llama.cpp == HF on all real prompts
        bad_rows, bad_tokens, first = 0, 0, None
        for index, (prompt, exp) in enumerate(zip(prompts, expected, strict=True)):
            got = tokenize(port, prompt)
            if got != exp:
                bad_rows += 1
                bad_tokens += sum(1 for a, b in zip(exp, got, strict=False) if a != b)
                if first is None:
                    pos = next((n for n, (a, b) in enumerate(zip(exp, got, strict=False)) if a != b), 0)
                    first = {"row": index, "pos": pos, "hf": exp[pos], "llama": got[pos]}
        report["gates"]["tokenizer_parity"] = {
            "ok": bad_rows == 0, "rows": len(prompts), "bad_rows": bad_rows,
            "bad_tokens": bad_tokens, "first_divergence": first,
        }
        print(f"[{'OK ' if bad_rows == 0 else 'FAIL'}] G4 tokens: {len(prompts) - bad_rows}/{len(prompts)} "
              f"prompts match HF ({bad_tokens} divergent tokens)")
        if first:
            print(f"        first: row {first['row']} position {first['pos']} "
                  f"hf={first['hf']} llama={first['llama']}")
        if bad_rows:
            failures.append(f"tokenization diverges on {bad_rows} rows")

        # G5 · file type (a swapped quantization must not pass unnoticed)
        ftype = props.get("model_ftype")
        quant_ok = ftype == args.expect_quant
        report["gates"]["file_type"] = {"ok": quant_ok, "got": ftype, "expected": args.expect_quant}
        print(f"[{'OK ' if quant_ok else 'FAIL'}] G5 file type: {ftype} (expected {args.expect_quant})")
        if not quant_ok:
            failures.append(f"file type {ftype} != {args.expect_quant}")
        report["build_info"] = props.get("build_info")
    finally:
        proc.terminate()
        proc.wait(timeout=30)

    # G6 · sensitivity: what else can corrupt the tokenization
    if not args.no_sweep:
        sweep: dict[str, int] = {}
        for pre in PRE_CANDIDATES:
            port = free_port()
            proc = start_server(args.server, args.gguf, port, ctx, 0, logfile,
                                extra=["--override-kv", f"tokenizer.ggml.pre=str:{pre}"])
            try:
                sweep[pre] = sum(1 for prompt, exp in zip(prompts, expected, strict=True)
                                 if tokenize(port, prompt) != exp)
            finally:
                proc.terminate()
                proc.wait(timeout=30)
        report["pre_sweep_bad_rows"] = sweep
        shipped = sweep.get(SHIPPED_PRE)
        print(f"\n[{'OK ' if shipped == 0 else 'FAIL'}] G6 pre sweep: rows diverging out of "
              f"{len(prompts)}: " + ", ".join(f"{k}={v}" for k, v in sweep.items()))
        if shipped != 0:
            failures.append(f"actual pre={SHIPPED_PRE} diverges on {shipped} rows")

    out = RESULTS / "gguf_verify.json"
    report["failures"] = failures
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{out.relative_to(ROOT)}")
    if failures:
        print("FAILURE: " + "; ".join(failures))
        return 1
    print("all gates passed")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
