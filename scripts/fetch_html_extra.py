"""Fetch rendered HTML for the coded-infobox pages the original ``titles_stats`` set missed.

The first HTML pull only covered combat/character stat pages (~7.6k). An audit found ~21.8k more
pages (collectibles, weapons/armor, missions, locations, quests, arts, skills, materials, ...) whose
infoboxes still carry undecoded Lua codes (``panel:0`` -> "Lucky Panel? No", ``Rare:4`` -> "Prime").
Their decoded values exist only in rendered HTML, exactly like the stat pages. This script fetches
that HTML so ``parse_html.run_hybrid`` can replace the raw-code infoboxes with decoded factblocks.

Design:
  * Reads titles from ``data/raw/titles_stats_additional.jsonl`` (one {"title": ...} per line).
  * Writes into the SAME ``paths.html`` dir but with an offset filename block (``html_01000+``) so it
    never overwrites the original ``html_00000..00075`` batches. ``iter_html_records`` globs them all.
  * Resumable via its own checkpoint (``data/raw/html_extra_checkpoint.json``); a crash re-runs only
    the current batch.
  * Serial requests at ``html_request_delay_seconds`` (API etiquette: serial + maxlag, not the
    crawler Crawl-delay). ``maxlag`` in WikiClient still backs off under server load.

Run (coded-infobox pass, defaults):  .venv/Scripts/python.exe -m scripts.fetch_html_extra
Run (table-gap second pass):
    .venv/Scripts/python.exe -m scripts.fetch_html_extra \
        data/raw/titles_stats_tables.jsonl data/raw/html_tables_checkpoint.json 2000

Each pass uses its OWN title list, checkpoint, and filename offset block so they never collide and
each is independently resumable. Run passes SERIALLY (never concurrently): MediaWiki API etiquette
is serial requests; two loops at once would be parallel hits on the wiki.
"""

import json
import os
import sys

import yaml

from xeno_rag.api_client import WikiClient
from xeno_rag.fetch_content import batched, load_checkpoint, save_checkpoint
from xeno_rag.fetch_html import _write_batch_gz, fetch_one

# Defaults = the coded-infobox pass; argv overrides for the table-gap second pass.
TITLES = "data/raw/titles_stats_additional.jsonl"
CHECKPOINT = "data/raw/html_extra_checkpoint.json"
FILE_OFFSET = 1000  # html_01000.jsonl.gz onward; originals are html_00000..00075


def _prevent_sleep():
    """Keep the system awake for this process's lifetime so an idle machine-sleep can't suspend/kill a
    multi-hour fetch (the first run died when the machine slept). Windows-only; ES_CONTINUOUS is
    auto-cleared when the process exits, so nothing to undo. No-op elsewhere."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("[keep-awake] system sleep inhibited for this process", flush=True)
    except Exception as e:  # noqa: BLE001 - keep-awake is best-effort, never block the fetch
        print(f"[keep-awake] could not inhibit sleep: {e}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # wiki titles outside the ANSI codepage
    _prevent_sleep()
    titles_path = sys.argv[1] if len(sys.argv) > 1 else TITLES
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else CHECKPOINT
    file_offset = int(sys.argv[3]) if len(sys.argv) > 3 else FILE_OFFSET

    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    client = WikiClient(cfg)
    client.delay = cfg.get("html_request_delay_seconds", 1.0)  # faster serial cadence

    with open(titles_path, encoding="utf-8") as f:
        titles = [json.loads(ln)["title"] for ln in f if ln.strip()]
    html_dir = cfg["paths"]["html"]
    os.makedirs(html_dir, exist_ok=True)
    batch_size = cfg.get("html_batch_size", 100)
    batches = list(batched(({"title": t} for t in titles), batch_size))
    start = load_checkpoint(checkpoint) + 1

    print(f"{titles_path} | {len(titles)} titles | {len(batches)} batches | "
          f"delay={client.delay}s | offset={file_offset} | resume @ batch {start}", flush=True)
    for i, batch in enumerate(batches):
        if i < start:
            continue
        records, errors = [], 0
        for t in batch:
            rec = fetch_one(client, t["title"])
            if "error" in rec:
                errors += 1
            records.append(rec)
        _write_batch_gz(file_offset + i, records, html_dir)
        save_checkpoint(i, checkpoint)
        done = i - start + 1
        print(f"batch {i + 1}/{len(batches)} -> html_{file_offset + i:05d}.jsonl.gz "
              f"({len(records)} pages, {errors} errors) | {done} batches this run", flush=True)
        if errors:
            print("  failed pages stay in the batch as error records; run "
                  "`python -m xeno_rag.pipeline retry_timeouts` afterwards to re-fetch transient ones",
                  flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
