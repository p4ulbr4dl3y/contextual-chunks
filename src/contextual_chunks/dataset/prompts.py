import re

import docx
from bs4 import BeautifulSoup

from contextual_chunks.dataset.tokenizer import count_tokens, split_by_tokens


def parse_docx(path):
    try:
        doc = docx.Document(path)
        lines = []
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                lines.append(text)

        # Tables to markdown
        for table in doc.tables:
            table_lines = []
            for row_idx, row in enumerate(table.rows):
                row_cells = [c.text.replace("\n", " ").strip() for c in row.cells]
                table_lines.append("| " + " | ".join(row_cells) + " |")
                if row_idx == 0:
                    table_lines.append("|" + "|".join(["---"] * len(row_cells)) + "|")
            if table_lines:
                lines.append("\n" + "\n".join(table_lines) + "\n")

        return "\n\n".join(lines)
    except Exception:
        return ""

def parse_html(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            html = f.read()
        soup = BeautifulSoup(html, "lxml")

        # Format tables to markdown
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                if len(rows) > 1:
                    header_len = len(rows[0].split("|")) - 2
                    sep = "|" + "|".join(["---"] * header_len) + "|"
                    md_table = "\n\n" + rows[0] + "\n" + sep + "\n" + "\n".join(rows[1:]) + "\n\n"
                else:
                    md_table = "\n\n" + "\n".join(rows) + "\n\n"
                table.replace_with(md_table)

        text = soup.get_text(separator="\n")
        # Clean multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception:
        return ""

def parse_xml(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            xml = f.read()
        soup = BeautifulSoup(xml, "xml")
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
    except Exception:
        return ""

def extract_document_text(path):
    with open(path, "rb") as f:
        head = f.read(1024)

    if head.startswith(b"%PDF"):
        return "" # Scans skip for ground truth
    elif b"word/" in head or head.startswith(b"PK\x03\x04"):
        return parse_docx(path)
    elif b"<!DOCTYPE html" in head or b"<html" in head or b"<HTML" in head:
        return parse_html(path)
    elif head.startswith(b"<?xml") or b"<xml" in head:
        return parse_xml(path)
    return ""

# ---------------------------------------------------------------------------
# Text normalization + table-aware, token-aware chunking
#
# Old behaviour (bug): a markdown table is ONE paragraph (rows are separated by
# a single "\n"), so the `p_words > max_words` branch ran `" ".join(words)` and
# destroyed every newline -> 25% of the dataset became one-line pipe soup with
# rows cut mid-row.  Tables are now split BY ROWS, the header block is repeated
# in every chunk, and no word-wrapping is ever applied to a table.
# ---------------------------------------------------------------------------

_SEP_RE = re.compile(r"^\|[\s\-:|]+\|$")
_NBSP_RE = re.compile("[\u00a0\u202f\u2009\u2007\u2060]")
_INLINE_WS_RE = re.compile(r"[ \t\x0b\x0c]+")

def normalize_text(text):
    """NBSP -> space, strip trailing blanks, collapse 3+ newlines to 2.

    Newlines inside a block are preserved: they are the only row delimiter of
    markdown tables.
    """
    if not text:
        return ""
    text = _NBSP_RE.sub(" ", text)
    text = _INLINE_WS_RE.sub(" ", text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

def looks_like_table(block):
    """A table = >=2 lines starting with `|` (separator line optional)."""
    return sum(1 for ln in block.split("\n") if ln.lstrip().startswith("|")) >= 2

def table_parts(block):
    """Split a table block into (header_lines, body_rows).

    header_lines = everything up to and including the `|---|---|` separator
    (keeps multi-level headers intact).  Without a separator the header is
    empty and all lines are treated as body rows.
    """
    lines = [ln for ln in block.split("\n") if ln.strip()]
    for i, ln in enumerate(lines):
        if _SEP_RE.match(ln.strip()):
            return lines[:i + 1], lines[i + 1:]
    return [], lines

def is_junk_table(block):
    """Layout-only tables (`|  |  |`) carry no information."""
    _, body = table_parts(block)
    if not body:
        return True
    cells = [c.strip() for row in body for c in row.strip("|").split("|")]
    return not any(len(c) >= 2 for c in cells)

def _plain_chunks(paragraphs, max_tokens, overlap_tokens):
    chunks, cur, cur_tokens = [], [], 0
    for p in paragraphs:
        t = count_tokens(p)
        if cur and cur_tokens + t > max_tokens:
            chunks.append("\n\n".join(cur))
            if overlap_tokens and count_tokens(cur[-1]) <= overlap_tokens:
                cur, cur_tokens = [cur[-1]], count_tokens(cur[-1])
            else:
                cur, cur_tokens = [], 0
        cur.append(p)
        cur_tokens += t
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks

def chunk_text(text, max_tokens=1200, overlap_tokens=0, max_table_rows=8,
               max_table_tokens=1500, min_tokens=30, repeat_table_header=False):
    """Chunk a document window into pieces of <= `max_tokens` tokens.

    Tables are chunked by ROWS (never word-wrapped), so the markdown structure
    survives.  With `repeat_table_header=True` every table chunk is prefixed by
    the table header block (self-contained retrieval artifact, but then the
    chunk is no longer a contiguous slice of the document -- the caller must
    record `chunk_span`).  Default False: every chunk is an exact slice of the
    window, which keeps the `chunk in document` invariant trivially checkable.
    """
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    chunks, plain = [], []

    def flush_plain():
        if plain:
            chunks.extend(_plain_chunks(list(plain), max_tokens, overlap_tokens))
            plain.clear()

    for block in blocks:
        if looks_like_table(block):
            flush_plain()
            if is_junk_table(block):
                continue
            header, body = table_parts(block)
            prefix = header if repeat_table_header else []
            def join(rows, prefix=prefix):
                return "\n".join(prefix + rows) if prefix else "\n".join(rows)
            header_tokens = count_tokens("\n".join(prefix))
            budget = max(max_table_tokens - header_tokens, 1)
            cur_rows, cur_tokens = [], 0
            for row in body:
                row_tokens = count_tokens(row)
                if row_tokens > budget:
                    # pathological row: hard-split (still row-oriented)
                    if cur_rows:
                        chunks.append(join(cur_rows))
                        cur_rows, cur_tokens = [], 0
                    for piece in split_by_tokens(row, budget):
                        chunks.append(join([piece]))
                    continue
                if cur_rows and (len(cur_rows) >= max_table_rows
                                 or cur_tokens + row_tokens > budget):
                    chunks.append(join(cur_rows))
                    cur_rows, cur_tokens = [], 0
                cur_rows.append(row)
                cur_tokens += row_tokens
            if cur_rows:
                chunks.append(join(cur_rows))
        else:
            plain.extend(split_by_tokens(block, max_tokens))
    flush_plain()

    return [c for c in chunks if count_tokens(c) >= min_tokens]
