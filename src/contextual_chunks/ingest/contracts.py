"""Download the five source documents of the contract category.

The files sit behind a Cloudflare check on zakupki.gov.ru, so the download goes
through an already authenticated browser session: `agent-browser` fetches the
document inside the page and returns it base64-encoded, which keeps the session
cookies and skips the certificate issues.

This is provenance, not a reproducible step: the files are the raw input of the
`construction` category and the URLs below are the exact ones used.

    uvx --from agent-browser agent-browser open https://zakupki.gov.ru/   # once, by hand
    .venv/bin/python -m contextual_chunks.ingest.contracts
"""

from __future__ import annotations

import base64
import json
import os
import subprocess

from contextual_chunks import paths

# Links on the contract card, in the order they appear on the page.
TARGETS = [
    {
        "filename": "01_kontrakt_osnovnoy.pdf",
        "url": "https://zakupki.gov.ru/44fz/filestore/public/1.0/download/rgk2/file.html?uid=BAAC664EF0794640A27427E06AFA0B7C",
    },
    {
        "filename": "02_pechatnaya_forma.pdf",
        "url": "https://zakupki.gov.ru/44fz/filestore/public/1.0/download/rgk2/file.html?uid=5A805CA29897426A836726CC2B40209E",
    },
    {
        "filename": "03_elektronny_kontrakt.pdf",
        "url": "https://zakupki.gov.ru/44fz/filestore/public/1.0/download/rgk2/file.html?uid=12EF103A660042F4998C01D304A4D4B8",
    },
    {
        "filename": "04_platezhnoe_poruchenie.pdf",
        "url": "https://zakupki.gov.ru/44fz/filestore/public/1.0/download/rgk2/file.html?uid=01A0804228FB7AFEA59E030705ECECBA",
    },
    {
        "filename": "05_dokument_priemki_ryba.pdf",
        "url": "https://zakupki.gov.ru/44fz/filestore/public/1.0/download/rgk2/file.html?uid=01A0AE5D51D271148772E6BB2347629B",
    },
]


def fetch_b64(url: str) -> str | None:
    """Fetch `url` inside the browser session; return base64 content."""
    js = f"""
    (async () => {{
        const res = await fetch('{url}');
        const blob = await res.blob();
        return new Promise((resolve) => {{
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result.split(',')[1]);
            reader.readAsDataURL(blob);
        }});
    }})()
    """
    proc = subprocess.run(["agent-browser", "eval", js], capture_output=True, text=True)
    raw = proc.stdout.strip()
    if raw.startswith('"') and raw.endswith('"'):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw or None


def main() -> int:
    os.makedirs(paths.DOWNLOADED_DIR, exist_ok=True)
    print("downloading contract documents through the browser session")
    for target in TARGETS:
        dest = os.path.join(paths.DOWNLOADED_DIR, target["filename"])
        print(f"\n{target['filename']} ...")
        raw = fetch_b64(target["url"])
        if not raw:
            print("  no data returned")
            continue
        try:
            data = base64.b64decode(raw)
        except Exception as exc:  # noqa: BLE001 - the browser can return an error page
            print(f"  not base64: {exc}; head={raw[:100]!r}")
            continue
        with open(dest, "wb") as fh:
            fh.write(data)
        print(f"  saved {dest} ({len(data) / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
