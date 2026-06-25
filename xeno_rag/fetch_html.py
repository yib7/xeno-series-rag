"""Resumable fetch of *rendered* HTML via the MediaWiki ``action=parse`` API.

Unlike content fetch (``action=query`` batches 50 titles/call), ``action=parse`` renders one page per
call — that's the price of getting the Lua-decoded stat tables that only exist in the HTML. Pages are
grouped into gzipped JSONL batches for resume granularity; the checkpoint advances per batch, so a
crash only re-fetches the current batch. Etiquette (User-Agent, maxlag, throttle) lives in WikiClient.
"""

import gzip
import json
import os
from typing import Iterator

from .api_client import WikiClient
from .fetch_content import batched, save_checkpoint, load_checkpoint, _read_titles


def fetch_one(client, title: str) -> dict:
    """Fetch one page's rendered HTML (+ wikitext, kept as a parse fallback). Never raises: a missing
    page or parse error is recorded so the batch — and the whole run — keeps going."""
    try:
        data = client.get({
            "action": "parse", "page": title,
            "prop": "text|wikitext", "redirects": 1,
        })
    except Exception as exc:  # noqa: BLE001 - record + continue, don't abort a 34k run
        return {"title": title, "error": f"request:{exc}"}
    if not isinstance(data, dict) or "parse" not in data:
        code = (data or {}).get("error", {}).get("code", "no-parse")
        return {"title": title, "error": code}
    p = data["parse"]
    return {
        "pageid": p.get("pageid"),
        "title": p.get("title", title),
        "html": p.get("text", ""),
        "wikitext": p.get("wikitext", ""),
    }


def _write_batch_gz(index: int, records, html_dir: str) -> None:
    os.makedirs(html_dir, exist_ok=True)
    path = os.path.join(html_dir, f"html_{index:05d}.jsonl.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def iter_html_records(html_dir: str) -> Iterator[dict]:
    import glob
    for path in sorted(glob.glob(os.path.join(html_dir, "html_*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def fetch_all(client, titles, cfg: dict, start_batch: int = 0, log=print) -> None:
    batch_size = cfg.get("html_batch_size", 100)
    html_dir = cfg["paths"]["html"]
    checkpoint = cfg["paths"]["html_checkpoint"]
    batches = list(batched(titles, batch_size))
    for i, batch in enumerate(batches):
        if i < start_batch:
            continue
        records, errors = [], 0
        for t in batch:
            rec = fetch_one(client, t["title"])
            if "error" in rec:
                errors += 1
            records.append(rec)
        _write_batch_gz(i, records, html_dir)
        save_checkpoint(i, checkpoint)
        if log:
            log(f"batch {i+1}/{len(batches)} done ({len(records)} pages, {errors} errors)", flush=True)


def run(cfg: dict, client=None, titles=None, log=print) -> None:
    """Fetch rendered HTML, resuming from the checkpoint. Defaults to the targeted stat-page list
    (``paths.html_titles``) — only those pages have Lua-decoded tables that wikitext can't see — and
    falls back to the full title list if no targeted list is configured."""
    if client is None:
        client = WikiClient(cfg)
    if titles is None:
        path = cfg["paths"].get("html_titles") or cfg["paths"]["titles"]
        titles = _read_titles(path)
    start_batch = load_checkpoint(cfg["paths"]["html_checkpoint"]) + 1
    fetch_all(client, titles, cfg, start_batch=start_batch, log=log)
