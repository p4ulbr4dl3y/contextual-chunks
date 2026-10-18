"""Network-free tests of the query set: JSON parsing, filters, target picking, determinism.

    .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from contextual_chunks.queries.build import (
    Store,
    df_table,
    doc_head,
    jaccard,
    max_ngram,
    neighbours_text,
    pick_docs,
    pick_targets,
    stage_filter,
    stem_ws,
    words,
)
from contextual_chunks.queries.teacher import parse_json
from contextual_chunks.retrieval.corpus import load

OK = Counter()


@pytest.fixture(scope="module")
def chunks():
    return load()


def check(name: str, cond: bool, extra: str = "") -> None:
    OK["pass" if cond else "fail"] += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")
    assert cond, name

def test_parse_json() -> None:
    print("\n[parse_json]")
    check("clean json", parse_json('{"a": 1}') == {"a": 1})
    check("inside ```json", parse_json('```json\n{"a": "б"}\n```') == {"a": "б"})
    check("with text around", parse_json('Вот ответ:\n{"answer": "A"}\nГотово') == {"answer": "A"})
    check("garbage -> None", parse_json("никакого json") is None)
    check("empty -> None", parse_json("") is None)
    check("array -> None", parse_json("[1, 2]") is None)

def test_text_utils() -> None:
    print("\n[text utils]")
    check("words", words("Срок поставки — 31.08.2026!") == ["срок", "поставки", "31", "08", "2026"])
    check("stem conflates forms", stem_ws("поставки") == stem_ws("поставка"))
    check("jaccard 1.0", jaccard({"a", "b"}, {"a", "b"}) == 1.0)
    check("jaccard 0", jaccard({"a"}, {"b"}) == 0.0)
    a, b = words("срок поставки товара до конца августа"), words("срок поставки товара до конца августа 2026")
    check("max_ngram catches a copy", max_ngram(a, b) >= 6, f"={max_ngram(a, b)}")
    check("max_ngram on different text = 0",
          max_ngram(words("поставка кабеля в срок"), words("охрана труда на объекте")) == 0)
    check("a single shared preposition is not a copy",
          max_ngram(words("поставка кабеля в срок"), words("охрана труда в цехе")) < 5)

def test_corpus_and_df(chunks) -> None:
    print("\n[corpus and df]")
    df = df_table(chunks)
    check("corpus is non-empty", len(chunks) > 3000, f"{len(chunks)} chunks")
    check("df exists", len(df) > 10000, f"{len(df)} unique stems")
    check("df is monotone in frequency", df.most_common(1)[0][1] >= 100)
    singles = [w for w, n in df.items() if n == 1]
    check("rare tokens exist (df=1)", len(singles) > 100, f"{len(singles)} items")
    print(f"        df=1 examples: {singles[:6]}")

def test_pick_targets(chunks) -> None:
    print("\n[pick_targets: determinism and types]")
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    doc = max(by_doc, key=lambda d: len(by_doc[d]))
    doc_chunks = by_doc[doc]
    t1 = pick_targets(doc_chunks, 3, random.Random("x"))
    t2 = pick_targets(doc_chunks, 3, random.Random("x"))
    check("determinism by seed", t1 == t2, str(t1))
    check("no repeated chunks", len({i for i, _ in t1}) == len(t1))
    check("3 types", len({t for _, t in t1}) == 3, str([t for _, t in t1]))
    check("indices in bounds", all(0 <= i < len(doc_chunks) for i, _ in t1))
    small = doc_chunks[:2]
    check("small doc does not crash", len(pick_targets(small, 3, random.Random("y"))) <= 2)

def test_type_routing(chunks) -> None:
    print("\n[ok_for_type: chunk routing]")
    from contextual_chunks.queries.build import digit_ratio, ok_for_type
    tables = [c for c in chunks if c.is_table]
    plain = [c for c in chunks if not c.is_table and digit_ratio(c) < 0.25]
    numeric = [c for c in chunks if digit_ratio(c) > 0.3]
    check("tables exist", len(tables) > 10, f"{len(tables)}")
    check("numeric chunks exist", len(numeric) > 10, f"{len(numeric)}")
    check("table accepts a table", ok_for_type(tables[0], "table"))
    check("table accepts a numeric chunk", ok_for_type(numeric[0], "table"))
    check("semantic rejects a table", not ok_for_type(tables[0], "semantic"))
    check("semantic rejects a numeric chunk", not ok_for_type(numeric[0], "semantic"))
    check("semantic accepts a text chunk", ok_for_type(plain[0], "semantic"))
    check("multihop rejects a numeric chunk", not ok_for_type(numeric[0], "multihop"))
    check("short chunk filtered out", not ok_for_type(next(c for c in chunks if len(words(c.chunk)) < 25), "semantic"))
    req = [c for c in chunks if c.is_requisites and not c.is_table and digit_ratio(c) < 0.25]
    check("requisites exist", len(req) > 10, f"{len(req)}")
    if req:
        check("semantic rejects requisites", not ok_for_type(req[0], "semantic"))
    check("corpus stats know requisites", "requisites" in __import__("contextual_chunks.retrieval.corpus", fromlist=["x"]).stats(chunks))

def test_pick_docs(chunks) -> None:
    print("\n[pick_docs: stratification]")
    two = pick_docs(chunks, 2, 7)
    check("2 docs", len(two) == 2)
    check("distinct docs", two[0] != two[1])
    ten = pick_docs(chunks, 10, 7)
    cats = {d.split("_")[0] for d in ten}
    check("categories are shuffled", len(cats) >= 3, str(sorted(cats)))
    check("determinism", pick_docs(chunks, 10, 7) == ten)
    by_doc_n: dict[str, int] = {}
    for c in chunks:
        by_doc_n[c.doc_id] = by_doc_n.get(c.doc_id, 0) + 1
    big = sum(1 for n in by_doc_n.values() if n >= 4)
    check("all = all docs with >=4 chunks", len(pick_docs(chunks, None, 7)) == big,
          f"{big} of {len(by_doc_n)}")

def test_prompts_assembly(chunks) -> None:
    print("\n[prompt assembly]")
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    doc_chunks = by_doc[max(by_doc, key=lambda d: len(by_doc[d]))]
    head = doc_head(doc_chunks)
    check("head is truncated", 0 < len(head) <= 4000, f"{len(head)} characters")
    nb = neighbours_text(doc_chunks, 5, window=2)
    check("neighbours exist", "фрагмент" in nb and str(len(doc_chunks)) in nb)
    check("neighbours without the target", doc_chunks[5].chunk[:40] not in nb)
    check("document boundary", "нет)" in neighbours_text(doc_chunks[:1], 0))

def test_filter(chunks, tmp_path: Path) -> None:
    print("\n[stage_filter: all branches]")
    from contextual_chunks.queries.build import is_identifier
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    doc_id = max(by_doc, key=lambda d: len(by_doc[d]))
    doc_chunks = by_doc[doc_id]
    gold = doc_chunks[10]
    other_doc = next(d for d in by_doc if d != doc_id)
    foreign = by_doc[other_doc][3]                      # we take the answer from a FOREIGN document
    df = df_table(chunks)
    # leak = identifier: a long digit string with low df
    rare_word = next(w for w in words(" ".join(c.chunk for c in chunks[:400]))
                     if is_identifier(w) and df[stem_ws(w)[0]] <= 2)
    check("identifier recognized", is_identifier(rare_word), rare_word)
    check("an ordinary word is not an identifier", not is_identifier("покрытие"))
    check("amount/number = identifier", is_identifier("135125"))
    check("a single digit is not an identifier", not is_identifier("3"))
    check("a part number with digits = identifier", is_identifier("ввг3х2"))
    check("a word with one digit is not an identifier", not is_identifier("гост2"))
    g_ans = [w for w in words(gold.chunk) if len(w) >= 5][:3]
    f_ans = [w for w in words(foreign.chunk) if len(w) >= 5][:3]
    check("rare token is rare", df[stem_ws(rare_word)[0]] <= 2, f"df={df[stem_ws(rare_word)[0]]}")
    check("a foreign-doc answer is not in the gold",
          sum(w in " ".join(words(gold.chunk)) for w in f_ans) == 0, str(f_ans))

    store = Store(tmp_path / "t2.jsonl")
    base = {"doc_id": doc_id, "category": "it", "gold_idx": gold.idx,
            "gold_doc_chunk_index": 10, "type": "semantic", "generator": "deepseek",
            "judge_letter": "B", "judge": "B"}
    q_ok = "какой срок поставки указан в договоре на поставку товаров заказчику"
    store.add({**base, "qid": "a_ok", "q": q_ok, "answer": " ".join(g_ans)})
    store.add({**base, "qid": "b_leak", "q": f"{q_ok} {rare_word}", "answer": " ".join(g_ans)})
    store.add({**base, "qid": "c_copy", "q": " ".join(words(gold.chunk)[:12]), "answer": " ".join(g_ans)})
    store.add({**base, "qid": "d_short", "q": "какой срок поставки", "answer": " ".join(g_ans)})
    store.add({**base, "qid": "e_noans", "q": q_ok, "answer": " ".join(f_ans)})
    store.add({**base, "qid": "f_badjudge", "judge": "C", "q": q_ok, "answer": " ".join(g_ans)})
    store.add({**base, "qid": "g_genfail", "reject_gen": "HTTP500"})
    store.add({**base, "qid": "z_dup", "q": q_ok, "answer": " ".join(g_ans)})
    reasons = stage_filter(chunks, store)
    got = {r["qid"]: r["reject"] for r in store.rows.values()}
    print("       ", {k: (v or "OK") for k, v in sorted(got.items())})
    check("ok accepted", got["a_ok"] is None, str(got["a_ok"]))
    check("leak (identifier) rejected", (got["b_leak"] or "").startswith("leak"), str(got["b_leak"]))
    check("copy rejected", (got["c_copy"] or "").startswith("copy"), str(got["c_copy"]))
    check("len rejected", (got["d_short"] or "").startswith("len"), str(got["d_short"]))
    check("answer_missing", got["e_noans"] == "answer_missing", str(got["e_noans"]))
    check("judge mismatch", (got["f_badjudge"] or "").startswith("judge"), str(got["f_badjudge"]))
    check("gen-fail", got["g_genfail"] == "gen", str(got["g_genfail"]))
    check("dup (same question)", got["z_dup"] == "dup", str(got["z_dup"]))
    check("exactly one accepted", sum(1 for v in got.values() if v is None) == 1)
    check("reason counter", set(reasons) >= {"leak", "copy", "len", "answer_missing", "judge", "gen", "dup"},
          str(dict(reasons)))

def test_store_roundtrip(tmp_path: Path) -> None:
    print("\n[Store: write/read/dedup]")
    p = tmp_path / "s.jsonl"
    s = Store(p)
    s.add({"qid": "b", "q": "второй"})
    s.add({"qid": "a", "q": "первый"})
    s.add({"qid": "a", "q": "первый v2"})
    s.save()
    back = Store(p)
    check("dedup by qid", len(back.rows) == 2)
    check("latest version", back.rows["a"]["q"] == "первый v2")
    check("sorting in the file", p.read_text().splitlines()[0].startswith('{"qid": "a"'))
    back.add({"qid": "c", "q_raw": "ещё не перефразирован"})
    check("todo yields unprocessed rows", [r["qid"] for r in back.todo("q")] == ["c"])
    check("todo skips reject_gen", back.todo("judge") == []
          or all(not r.get("reject_gen") for r in back.todo("judge")))

