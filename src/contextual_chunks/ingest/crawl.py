"""Crawl one contract per category from zakupki.gov.ru.

The site renders results with JavaScript and guards downloads with Cloudflare,
so the crawl drives a real browser session through `agent-browser`: open the
search page, collect contract-card links, then fetch the attached files inside
the page and save them base64-decoded.

This is provenance, not a reproducible step: it produced the raw documents of
the five categories (`it_hardware`, `pharma_tablets`, `food_supply`,
`construction`, `security_services`), and page markup changes will break it.

    uvx --from agent-browser agent-browser open https://zakupki.gov.ru/   # once, by hand
    .venv/bin/python -m contextual_chunks.ingest.crawl
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess

from contextual_chunks import paths

BASE = "https://zakupki.gov.ru"
CATEGORIES = [
    {"query": "серверное оборудование", "label": "it_hardware"},
    {"query": "лекарственные препараты", "label": "pharma_tablets"},
    {"query": "продовольственные товары", "label": "food_supply"},
    {"query": "строительные материалы", "label": "construction"},
    {"query": "услуги охраны", "label": "security_services"},
]

CARD_LINKS_JS = """
Array.from(document.querySelectorAll("a[href*='contractCard/document-info']"))
  .map(a => a.getAttribute("href"))
  .slice(0, 4)
"""

CARD_LINKS_FALLBACK_JS = """
Array.from(document.querySelectorAll("a[href*='contractCard/common-info']"))
  .map(a => a.getAttribute("href").replace("common-info", "document-info"))
  .slice(0, 4)
"""

CARD_FILES_JS = """
Array.from(document.querySelectorAll("a[href*='download']"))
  .map(a => ({
    name: (a.innerText || a.getAttribute('title') || '').trim().replace(/\\s+/g, ' '),
    url: a.getAttribute('href')
  }))
  .filter(x => x.name.length > 0 && x.url && x.url.startsWith('https://'))
  .slice(0, 5)
"""

DOWNLOAD_JS = """
(async () => {{
    try {{
        const res = await fetch('{url}');
        const blob = await res.blob();
        return new Promise((resolve) => {{
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result.split(',')[1]);
            reader.readAsDataURL(blob);
        }});
    }} catch (e) {{
        return null;
    }}
}})()
"""


def eval_js(js: str):
    """Run JS in the browser session and decode the JSON result when possible."""
    proc = subprocess.run(["agent-browser", "eval", js], capture_output=True, text=True)
    raw = proc.stdout.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def open_page(url: str, wait_ms: int) -> None:
    subprocess.run(["agent-browser", "open", url], capture_output=True)
    subprocess.run(["agent-browser", "wait", str(wait_ms)], capture_output=True)


def safe_filename(name: str) -> str:
    stem = re.sub(r"[^a-zA-Zа-яА-Я0-9_.-]", "_", name)[:50]
    if not stem.endswith((".pdf", ".docx", ".doc")):
        stem += ".bin"
    return stem


def main() -> int:
    os.makedirs(paths.RAW_DIR, exist_ok=True)
    print("crawling one contract per category")
    downloaded = 0
    for cat in CATEGORIES:
        print(f"\n---> {cat['label']} (query: {cat['query']!r})")
        search_url = (
            f"{BASE}/epz/contract/search/results.html?"
            "morphology=on&search-filter=%D0%94%D0%B0%D1%82%D0%B5+%D1%80%D0%B0%D0%B7%D0%BC%D0%B5%D1%89%D0%B5%D0%BD%D0%B8%D1%8F&"
            f"searchString={cat['query']}&contractStageList_3=on&fz44=on"
        )
        open_page(search_url, 4000)

        cards = eval_js(CARD_LINKS_JS)
        if not isinstance(cards, list) or not cards:
            print("  no contract cards, falling back to common-info links")
            cards = eval_js(CARD_LINKS_FALLBACK_JS)
        print(f"  cards found: {len(cards) if isinstance(cards, list) else 0}")
        if not isinstance(cards, list):
            continue

        for rel in cards:
            open_page(f"{BASE}{rel}", 3000)
            files = eval_js(CARD_FILES_JS)
            if not isinstance(files, list):
                continue
            for entry in files:
                save_name = f"{cat['label']}_{downloaded + 1}_{safe_filename(entry['name'])}"
                dest = os.path.join(paths.RAW_DIR, save_name)
                b64 = eval_js(DOWNLOAD_JS.format(url=entry["url"]))
                if not isinstance(b64, str) or len(b64) <= 100:
                    continue
                try:
                    data = base64.b64decode(b64)
                except Exception:  # noqa: BLE001 - the browser can return an error page
                    continue
                with open(dest, "wb") as fh:
                    fh.write(data)
                downloaded += 1
                print(f"  [+] {save_name} ({len(data) / 1024:.1f} KB)")
    print(f"\ndocuments downloaded: {downloaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
