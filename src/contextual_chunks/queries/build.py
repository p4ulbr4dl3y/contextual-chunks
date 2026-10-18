"""Build the query set with a gold chunk for retrieval metrics (recall@k/MRR/nDCG).

Four stages (usable separately, state is a single jsonl):

  1. `gen`    - the teacher invents a query from the target chunk (it sees the document
                start and neighbouring fragments, so that multi-hop is possible);
  2. `para`   - paraphrase WITHOUT access to the chunk: kills verbatim lexical leakage
                (the main source of rejects: otherwise BM25 solves everything and the
                arm difference disappears);
  3. `judge`  - question + gold chunk + 3 confuser neighbours: does exactly one answer?
  4. `filter` - deterministic checks: df=1 tokens, shared n-grams, length,
                presence of answer in the chunk, dedup.

Acceptance of the set is calibration against the baseline (see `audit_queries.py
--baseline`): BM25-only recall@10 must land in 0.3-0.6. Above 0.9 means the set is
lexical, below 0.1 it is unsolvable.

    .venv/bin/python -m contextual_chunks.queries.build --docs 2 --per-doc 3 \
        --out artifacts/queries.jsonl          # smoke
    .venv/bin/python -m contextual_chunks.queries.build --docs all --per-doc 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path


from contextual_chunks import paths
from contextual_chunks.queries.teacher import MODELS, Teacher
from contextual_chunks.retrieval.corpus import Chunk, load
from contextual_chunks.retrieval.sparse import words
from snowballstemmer import stemmer  # noqa: E402

OUT_DEFAULT = Path(paths.QUERIES)
GENERATORS = [g for g in os.environ.get("GEN_ALIASES", "mimo,qwen").split(",") if g]
RU = stemmer("russian")

TYPES = ["semantic", "multihop", "table", "lexical"]
TYPE_HINT = {
    "semantic": "пользователь описывает нужное своими словами, не зная формулировок документа",
    "multihop": "ответ требует связать целевой фрагмент с другим местом документа "
                "(условие+следствие, ссылка, срок+предмет)",
    "table": "вопрос про данные: числа, сроки, суммы, характеристики, реквизиты",
    "lexical": "вопрос содержит один-два значимых отраслевых термина, но не точную формулировку документа",
}
GEN_PROMPT = """Ты готовишь тестовые поисковые запросы к базе официальных документов (закупки, договоры, техдокументация, охрана труда).

Ниже начало документа, один целевой фрагмент и соседние фрагменты (чтобы понимать устройство документа).

<document_start>
{head}
</document_start>

<target_fragment>
{chunk}
</target_fragment>

<neighbouring_fragments>
{neighbours}
</neighbouring_fragments>

Придумай ОДИН запрос, который мог бы ввести пользователь и ответ на который есть в целевом фрагменте.
Характер запроса: {type_hint}

В базе много однотипных документов (контракты, ТЗ, инструкции), поэтому запрос должен позволять отличить именно этот документ: назови своими словами предмет закупки, отрасль или вид работ («поставка электрооборудования», «ремонт медицинской техники»). Без этого запрос «какой адрес доставки указан в соглашении» найдёт десятки документов.

Требования:
1. 8–25 слов, живая формулировка (как ищут люди), без «что сказано в пункте».
2. Не копируй дословно редкие термины, номера, названия организаций и точные формулировки из фрагмента — описывай их своими словами. Обычные отраслевые слова использовать можно.
3. Запрос — про СОДЕРЖАНИЕ (условия, требования, порядок, сроки, характеристики, состав), а не про номер/код/дату документа. Плохо: «какой номер у соглашения». Хорошо: «какие условия оплаты предусмотрены для поставки оборудования».
4. В поле answer — короткая цитата из целевого фрагмента (2–10 слов), которая отвечает на запрос.
5. Если для ответа нужно ещё и другое место документа — поставь needs_other_chunks=true.

Верни только JSON:
{{"question": "...", "answer": "...", "needs_other_chunks": false}}"""

PARA_PROMPT = """Перепиши поисковый запрос так, чтобы он не повторял дословных формулировок документа. Сохрани ВСЕ конкретные детали — и предметную (что за товар, работы или услуги), и смысловую (что именно нужно узнать) — и объём 10–20 слов. Убери только редкие термины, номера, коды, названия организаций и даты. НЕ обобщай: «К какому договору относится приложение?» или «какой адрес указан в соглашении» — плохо, потеряна предметная конкретика.

Запрос: "{question}"
{retry}
Верни только JSON: {{"question": "..."}}"""

JUDGE_PROMPT = """Пользователь искал: "{question}"

Ниже четыре фрагмента из базы документов. Определи, в каком из них есть ответ на запрос.

{options}

Если ответ есть только в одном фрагменте — укажи его букву; если в нескольких — "multiple"; если ни в одном — "none".

Верни только JSON: {{"answer": "A", "why": "кратко"}}"""


# ---------------------------------------------------------------- utilities
def stem_ws(text: str) -> list[str]:
    return [RU.stemWord(w) for w in words(text)]


def df_table(chunks: list[Chunk]) -> Counter:
    df: Counter = Counter()
    for c in chunks:
        df.update(set(stem_ws(c.chunk)))
    return df


def max_ngram(a: list[str], b: list[str], n: int = 6) -> int:
    """Maximum length of a shared n-gram (sliding window over the set b)."""
    bset = {tuple(b[i:i + k]) for k in range(2, n + 1) for i in range(len(b) - k + 1)}
    best = 0
    for k in range(2, n + 1):
        for i in range(len(a) - k + 1):
            if tuple(a[i:i + k]) in bset:
                best = max(best, k)
    return best


def is_identifier(w: str) -> bool:
    """Identifier token: contract number, INN, article number. Domain vocabulary
    ("road surface", df=1, but an ordinary word) is NOT considered leakage."""
    if len(w) >= 20:
        return True
    return sum(ch.isdigit() for ch in w) >= 2


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def doc_head(doc_chunks: list[Chunk], chars: int = 4000) -> str:
    out = ""
    for c in doc_chunks[:3]:
        out += c.chunk + "\n"
        if len(out) >= chars:
            break
    return out[:chars]


def neighbours_text(doc_chunks: list[Chunk], pos: int, window: int = 2, chars: int = 1200) -> str:
    parts = []
    for j in range(max(0, pos - window), min(len(doc_chunks), pos + window + 1)):
        if j != pos:
            parts.append(f"[фрагмент {j + 1} из {len(doc_chunks)}]\n{doc_chunks[j].chunk[:chars]}")
    return "\n\n".join(parts) or "(соседних фрагментов нет)"


def digit_ratio(c: Chunk) -> float:
    ws = words(c.chunk)
    return sum(w.isdigit() for w in ws) / max(1, len(ws))


def ok_for_type(c: Chunk, t: str) -> bool:
    """`semantic`/`multihop` are not taken from tables or "numeric" chunks: there the
    answer is an identifier, and the question degenerates into "what is the number of
    ..." (those go to the table type)."""
    if len(words(c.chunk)) < 25:
        return False
    if t == "table":
        return c.is_table or digit_ratio(c) > 0.25
    if t in ("semantic", "multihop"):
        # requisites (INN, addresses, phones) produce questions like "what is the number of ..."
        return not c.is_table and not c.is_requisites and digit_ratio(c) < 0.25
    return True


def pick_targets(doc_chunks: list[Chunk], per_doc: int, rng: random.Random) -> list[tuple[int, str]]:
    """(position in the document, type): one chunk per type, no repeats."""
    picked: list[tuple[int, str]] = []
    used: set[int] = set()
    types = (TYPES * (per_doc // len(TYPES) + 1))[:per_doc]
    for t in types:
        pool = [i for i, c in enumerate(doc_chunks) if i not in used and ok_for_type(c, t)]
        if not pool:
            pool = [i for i in range(len(doc_chunks)) if i not in used]
        if not pool:
            break
        i = rng.choice(pool)
        used.add(i)
        picked.append((i, t))
    return picked


def pick_docs(all_chunks: list[Chunk], n: int | None, seed: int) -> list[str]:
    by_doc: dict[str, list[Chunk]] = {}
    for c in all_chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    buckets: dict[str, list[str]] = {}
    for doc_id, cs in by_doc.items():
        if len(cs) >= 4:
            buckets.setdefault(cs[0].doc_id.split("_")[0], []).append(doc_id)
    rng = random.Random(seed)
    for v in buckets.values():
        v.sort()
        rng.shuffle(v)
    # round-robin over categories: the smoke run must not consist of one category
    order, cats = [], sorted(buckets)
    while any(buckets[c] for c in cats):
        for c in cats:
            if buckets[c]:
                order.append(buckets[c].pop())
    return order[:n] if n else order


# ---------------------------------------------------------------- store
class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[r["qid"]] = r

    def add(self, row: dict) -> None:
        self.rows[row["qid"]] = row

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in sorted(self.rows.values(), key=lambda r: r["qid"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    def todo(self, field: str) -> list[dict]:
        return [r for r in self.rows.values() if not r.get(field) and not r.get("reject_gen")]


# ---------------------------------------------------------------- stages
async def stage_gen(teacher: Teacher, chunks: list[Chunk], docs: list[str], per_doc: int,
                    store: Store, seed: int) -> None:
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    async def one(doc_id: str) -> None:
        doc_chunks = by_doc[doc_id]
        rng = random.Random(f"{seed}:{doc_id}")
        for pos, t in pick_targets(doc_chunks, per_doc, rng):
            c = doc_chunks[pos]
            qid = f"{c.idx}-{t}"
            if qid in store.rows and store.rows[qid].get("q_raw"):
                continue
            # two styles: the free mimo + qwen3.7-plus. deepseek-v4.1-flash returns
            # empty content on these prompts (verified), kept as an option via env
            gen = GENERATORS[len(store.rows) % len(GENERATORS)]
            prompt = GEN_PROMPT.format(head=doc_head(doc_chunks), chunk=c.chunk,
                                       neighbours=neighbours_text(doc_chunks, pos),
                                       type_hint=TYPE_HINT[t])
            out, meta = await teacher.ask_json(gen, prompt)
            row = {"qid": qid, "doc_id": doc_id, "category": c.doc_id.split("_")[0],
                   "gold_idx": c.idx, "gold_doc_chunk_index": pos, "type": t, "generator": gen,
                   "gold_tokens": c.chunk_tokens, "gen_meta": meta}
            if out and out.get("question"):
                row.update({"q_raw": out["question"].strip(),
                            "answer": (out.get("answer") or "").strip(),
                            "needs_other_chunks": bool(out.get("needs_other_chunks"))})
            else:
                row["reject_gen"] = meta.get("error", "no question")
            store.add(row)
            print(f"  gen  {row['qid']:<14} {t:<8} {gen:<8} {(row.get('q_raw') or row['reject_gen'])[:80]}",
                  flush=True)
        store.save()

    await teacher.map_docs([(d, (lambda d=d: one(d))) for d in docs])


async def stage_para(teacher: Teacher, store: Store) -> None:
    rows = [r for r in store.todo("q") if r.get("q_raw")]
    sem = asyncio.Semaphore(teacher.concurrency)

    async def one(r: dict) -> None:
        async with sem:
            # paraphrase and judging are simple tasks: the free mimo does them
            retry = ""
            for attempt in range(3):
                out, meta = await teacher.ask_json(
                    "mimo", PARA_PROMPT.format(question=r["q_raw"], retry=retry))
                r["para_meta"] = meta
                if not out or not out.get("question"):
                    r["reject_gen"] = meta.get("error", "no paraphrase")
                    break
                q = out["question"].strip()
                if 8 <= len(words(q)) <= 25:
                    r["q"] = q
                    break
                retry = (f"\nПрошлый вариант был {len(words(q))} слов — сделай 10–20 слов, "
                         f"сохранив детали. Он был таким: \"{q}\"\n")
                r["q"] = q                      # keep the last one, the filter decides
            print(f"  para {r['qid']:<14} {(r.get('q') or r.get('reject_gen',''))[:80]}", flush=True)

    await asyncio.gather(*(one(r) for r in rows))
    store.save()


def confusers(doc_chunks: list[Chunk], pos: int, chunks: list[Chunk], rng: random.Random,
              n: int = 3, min_gap: int = 6) -> list[Chunk]:
    """Confusers for the judge: 1 far chunk from the same document + the rest from others.

    Window neighbours do not work: chunks are overlapping pieces (31% of adjacent
    pairs have Jaccard>0.5), they also contain the answer and the judge would honestly
    answer "multiple".
    """
    out: list[Chunk] = []
    far = [i for i in range(len(doc_chunks)) if abs(i - pos) >= min_gap]
    if far:
        out.append(doc_chunks[rng.choice(far)])
    guard = 0
    while len(out) < n and guard < 200:
        guard += 1
        c = chunks[rng.randrange(len(chunks))]
        if c.doc_id != doc_chunks[0].doc_id:
            out.append(c)
    return out[:n]


async def stage_judge(teacher: Teacher, chunks: list[Chunk], store: Store) -> None:
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    rows = [r for r in store.todo("judge") if r.get("q")]
    sem = asyncio.Semaphore(teacher.concurrency)

    async def one(r: dict) -> None:
        doc_chunks = by_doc[r["doc_id"]]
        pos = r["gold_doc_chunk_index"]
        gold = doc_chunks[pos]
        rng = random.Random(f"judge:{r['qid']}")
        cand = [("gold", gold)] + [("other", c) for c in confusers(doc_chunks, pos, chunks, rng)]
        rng.shuffle(cand)
        letters = "ABCD"
        options = "\n\n".join(f"[{letters[i]}]\n{c.chunk[:1500]}" for i, (_, c) in enumerate(cand))
        gold_letter = letters[[k for k, (kind, _) in enumerate(cand) if kind == "gold"][0]]
        async with sem:
            out, meta = await teacher.ask_json("mimo", JUDGE_PROMPT.format(
                question=r["q"], options=options))
        r["judge_meta"] = meta
        r["judge_letter"] = gold_letter
        if out and out.get("answer"):
            r["judge"] = str(out["answer"]).strip().upper()[:8]
            r["judge_why"] = (out.get("why") or "")[:120]
        else:
            r["reject_gen"] = meta.get("error", "no judge answer")
        print(f"  judge {r['qid']:<14} gold={gold_letter} said={r.get('judge')} "
              f"{(r.get('judge_why') or '')[:50]}", flush=True)

    await asyncio.gather(*(one(r) for r in rows))
    store.save()


def stage_filter(chunks: list[Chunk], store: Store) -> Counter:
    df = df_table(chunks)
    by_idx = {c.idx: c for c in chunks}
    reasons: Counter = Counter()
    accepted: list[set[str]] = []
    for r in sorted(store.rows.values(), key=lambda r: r["qid"]):
        if r.get("reject_gen"):
            r["reject"] = "gen"
        elif not r.get("q"):
            r["reject"] = "no_query"
        else:
            q_ws, q_st = words(r["q"]), stem_ws(r["q"])
            gold = by_idx[r["gold_idx"]]
            g_st = set(stem_ws(gold.chunk))
            rare = [w for w in q_st if df[w] <= 2]
            rare_frac = len(rare) / max(1, len(q_st))
            # leakage = a unique identifier from the gold chunk (number/code/article)
            leak = [w for w in rare if is_identifier(w)]
            ngram = max_ngram(q_st, stem_ws(gold.chunk))
            ans_ws = [w for w in stem_ws(r.get("answer") or "") if len(w) >= 4]
            ans_hit = (sum(w in g_st for w in ans_ws) / len(ans_ws)) if ans_ws else 1.0
            if len(q_ws) < 8 or len(q_ws) > 25:
                r["reject"] = f"len:{len(q_ws)}"
            elif leak or rare_frac > 0.6:
                r["reject"] = f"leak:{','.join((leak or rare)[:3])}"
            elif ngram >= 5:
                r["reject"] = f"copy:{ngram}"
            elif ans_hit < 0.6:
                r["reject"] = "answer_missing"
            elif r.get("judge") != r.get("judge_letter"):
                r["reject"] = f"judge:{r.get('judge')}"
            elif any(jaccard(set(q_st), prev) > 0.6 for prev in accepted):
                r["reject"] = "dup"
            else:
                r["reject"] = None
                accepted.append(set(q_st))
            r["filters"] = {"words": len(q_ws), "leak": leak[:5], "max_ngram": ngram,
                            "answer_hit": round(ans_hit, 2), "rare_frac": round(rare_frac, 2),
                            "rare_words": rare[:5]}
        if r["reject"]:
            reasons[r["reject"].split(":")[0]] += 1
    store.save()
    return reasons


# ---------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="2", help="how many documents (a number or all)")
    ap.add_argument("--per-doc", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--stage", default="all", choices=["all", "gen", "para", "judge", "filter"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    chunks = load()
    docs = pick_docs(chunks, None if args.docs == "all" else int(args.docs), args.seed)
    store = Store(args.out)
    print(f"corpus {len(chunks)} chunks / {len({c.doc_id for c in chunks})} docs; "
          f"docs in the run: {len(docs)}; output {args.out}; stage {args.stage}", flush=True)

    async def run() -> None:
        async with Teacher(concurrency=args.concurrency) as teacher:
            t0 = time.time()
            if args.stage in ("all", "gen"):
                await stage_gen(teacher, chunks, docs, args.per_doc, store, args.seed)
            if args.stage in ("all", "para"):
                await stage_para(teacher, store)
            if args.stage in ("all", "judge"):
                await stage_judge(teacher, chunks, store)
            print(f"\ncalls {teacher.calls}, cost ${teacher.cost:.4f}, "
                  f"{time.time() - t0:.0f}s, tokens {dict(teacher.usage)}", flush=True)

    asyncio.run(run())
    if args.stage in ("all", "filter"):
        reasons = stage_filter(chunks, store)
        rows = list(store.rows.values())
        good = [r for r in rows if not r.get("reject") and r.get("q")]
        print(f"\ntotal candidates {len(rows)}, accepted {len(good)}, rejected {dict(reasons)}")
        for r in good[:10]:
            print(f"  OK  {r['qid']:<14} {r['type']:<8} {r['generator']:<8} {r['q'][:90]}")
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
