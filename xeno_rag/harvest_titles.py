"""Harvest all ns=0 article titles via list=allpages, paginated by apcontinue."""

import json
import os
from typing import Iterator

from .api_client import WikiClient


def harvest_titles(client, cfg: dict, nonredirects: bool = True) -> Iterator[dict]:
    """Yield {title, pageid} for every ns=0 page, following apcontinue pagination."""
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": 0,
        "aplimit": "max",
    }
    if nonredirects:
        params["apfilterredir"] = "nonredirects"

    while True:
        data = client.get(params)
        for p in data["query"]["allpages"]:
            yield {"title": p["title"], "pageid": p["pageid"]}
        if "continue" in data:
            params.update(data["continue"])
        else:
            break


def write_titles(records, path: str) -> int:
    """Write an iterable of {title, pageid} dicts as JSONL. Returns the count written.

    Mirrors bm25_index.build's atomic-write pattern: writes to a temp file (path + ".tmp") and
    os.replace()s it into place only once every record has been written successfully. Harvest has no
    checkpoint, so if the paginated fetch raises partway (e.g. api_client retries exhausted), the
    temp file is discarded and the pre-existing path is left byte-for-byte untouched -- not silently
    truncated, which would under-scope every downstream step.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    n = 0
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    os.replace(tmp_path, path)
    return n


def run(cfg: dict, client=None) -> int:
    """Harvest all titles and write them to cfg['paths']['titles']. Returns the count."""
    if client is None:
        client = WikiClient(cfg)
    return write_titles(harvest_titles(client, cfg), cfg["paths"]["titles"])
