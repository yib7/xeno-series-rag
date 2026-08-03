"""Batched, resumable content fetch.

Each batch of titles is written to its own JSONL file so a crash never corrupts a big file. After
each batch the checkpoint advances; on restart, completed batches are skipped.
"""

import json
import os
from collections.abc import Iterator

from .api_client import WikiClient


def batched(iterable, n: int = 50) -> Iterator[list]:
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def save_checkpoint(batch_index: int, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"last_completed_batch": batch_index}, f)


def load_checkpoint(path: str) -> int:
    """Return the last completed batch index, or -1 if no checkpoint exists."""
    if not os.path.isfile(path):
        return -1
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("last_completed_batch", -1)


def _write_batch(index: int, pages, pages_dir: str) -> None:
    os.makedirs(pages_dir, exist_ok=True)
    path = os.path.join(pages_dir, f"pages_{index:05d}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in pages)


def fetch_all(client, titles, cfg: dict, start_batch: int = 0) -> None:
    batch_size = cfg.get("batch_size", 50)
    pages_dir = cfg["paths"]["pages"]
    checkpoint = cfg["paths"]["checkpoint"]
    for i, batch in enumerate(batched(titles, batch_size)):
        if i < start_batch:
            continue
        titles_param = "|".join(t["title"] for t in batch)
        data = client.get({
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "titles": titles_param,
        })
        _write_batch(i, data["query"]["pages"], pages_dir)
        save_checkpoint(i, checkpoint)


def _read_titles(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(cfg: dict, client=None, titles=None) -> None:
    """Fetch all content, resuming from the checkpoint."""
    if client is None:
        client = WikiClient(cfg)
    if titles is None:
        titles = _read_titles(cfg["paths"]["titles"])
    start_batch = load_checkpoint(cfg["paths"]["checkpoint"]) + 1
    fetch_all(client, titles, cfg, start_batch=start_batch)
