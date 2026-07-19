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

import requests

from .api_client import WikiClient
from .fetch_content import batched, save_checkpoint, load_checkpoint, _read_titles

# Reserved recovery block for retry_timeouts: clear of the main pull (0..), the extra pass
# (1000..), and the tables pass (2000..). A resumable main fetch only ever writes low indices, so
# it can never overwrite a retry batch parked here.
RETRY_FILE_OFFSET = 9000


def fetch_one(client, title: str) -> dict:
    """Fetch one page's rendered HTML (+ wikitext, kept as a parse fallback). Never raises: a missing
    page or parse error is recorded so the batch — and the whole run — keeps going.

    Failures are categorized so the resume logic can react: a ``requests.Timeout`` is transient
    (server slow / network blip) and tagged ``timeout:...`` so ``retry_timeouts`` can re-attempt it, whereas
    any other exception is a permanent-until-fixed ``request:...`` error. Keeping them distinct stops a
    flaky network window from being silently indistinguishable from genuinely missing pages."""
    try:
        data = client.get({
            "action": "parse", "page": title,
            "prop": "text|wikitext", "redirects": 1,
        })
    except requests.Timeout as exc:  # transient — retryable on a later run
        return {"title": title, "error": f"timeout:{exc}"}
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


def _next_retry_index(html_dir: str) -> int:
    """First unused index inside the reserved ``RETRY_FILE_OFFSET`` block: ``RETRY_FILE_OFFSET`` if
    no retry batch exists yet, else one past the highest existing retry batch. Staying inside this
    block (instead of highest-overall + 1) is what keeps a retry pass from ever landing on an index
    a resumed, still-incomplete main fetch is going to write next."""
    import glob
    import re
    highest = RETRY_FILE_OFFSET - 1
    for path in glob.glob(os.path.join(html_dir, "html_*.jsonl.gz")):
        m = re.match(r"html_(\d+)\.jsonl\.gz$", os.path.basename(path))
        if m:
            idx = int(m.group(1))
            if idx >= RETRY_FILE_OFFSET:
                highest = max(highest, idx)
    return highest + 1


def _warn_if_main_pull_incomplete(cfg: dict, log) -> None:
    """Advisory-only: warn when the main pull's checkpoint looks behind the expected batch count,
    i.e. it may still be mid-flight. Not load-bearing -- the RETRY_FILE_OFFSET block already makes
    a retry pass safe regardless -- so any failure to resolve the title list (missing stat-page
    file, no ``paths.titles`` configured, etc.) just skips the warning rather than raising.

    Mirrors ``run()``'s title resolution to compute the expected batch count, then compares it
    against ``load_checkpoint``. Stays quiet whenever there's no reason to warn."""
    if not log:
        return
    checkpoint_path = cfg["paths"].get("html_checkpoint")
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return  # main pull hasn't started (or was never checkpointed) -- nothing to warn about
    try:
        stat_list = cfg["paths"].get("html_titles")
        path = stat_list or cfg["paths"]["titles"]
        if stat_list and not os.path.isfile(stat_list):
            return
        titles = _read_titles(path)
        batch_size = cfg.get("html_batch_size", 100)
        expected_batches = -(-len(titles) // batch_size)  # ceil division
        last_completed = load_checkpoint(checkpoint_path)
    except Exception:
        return  # advisory check only -- never let it block or crash the retry pass
    if last_completed + 1 < expected_batches:
        log(
            f"retry_timeouts: WARNING - main pull checkpoint is at batch {last_completed} of "
            f"{expected_batches} expected; the main pull may still be mid-flight. Recovery "
            f"batches are written to a reserved block (index >= {RETRY_FILE_OFFSET}) so a "
            "resumed main fetch will not overwrite them.",
            flush=True,
        )


def collect_timeout_titles(html_dir: str) -> list:
    """Titles whose LATEST record is a ``timeout:``-tagged failure.

    Batches are scanned in order and later records win, so a title already recovered by a previous
    retry pass (a newer successful record in a higher-numbered batch) is not re-fetched again —
    the pass is idempotent."""
    latest = {}
    for rec in iter_html_records(html_dir):
        latest[rec["title"]] = rec
    return sorted(
        t for t, rec in latest.items()
        if str(rec.get("error", "")).startswith("timeout:")
    )


def retry_timeouts(cfg: dict, client=None, log=print) -> int:
    """Re-attempt every page whose latest fetch failed with a transient ``timeout:`` error.

    This is the retry pass the ``fetch_one`` docstring promises: it scans the written HTML batches,
    collects the titles still marked ``timeout:``, and re-fetches them into NEW batches appended
    inside the reserved ``RETRY_FILE_OFFSET`` block (parse_html keys articles by title and an
    error-only record parses to nothing, so a recovered page supersedes its failure record and a
    still-failing retry cannot clobber an earlier success). The main fetch checkpoint is left
    untouched — it indexes the original title-list batches, which this pass does not revisit.
    Parking recovery batches in the reserved block (rather than right after the highest existing
    batch) is what stops them from being overwritten if the main pull later resumes and rewrites
    that same index (audit finding P1-2).

    Deliberately NOT part of `pipeline all`: it hits the live API, so a human runs it explicitly
    (``python -m xeno_rag.pipeline retry_timeouts``) after a flaky pull. Returns the number of
    titles re-attempted."""
    html_dir = cfg["paths"]["html"]
    titles = collect_timeout_titles(html_dir)
    if not titles:
        if log:
            log("retry_timeouts: no timeout-tagged failures found")
        return 0
    _warn_if_main_pull_incomplete(cfg, log)
    if client is None:
        client = WikiClient(cfg)
    batch_size = cfg.get("html_batch_size", 100)
    index = _next_retry_index(html_dir)
    still_failing = 0
    for offset, group in enumerate(batched(({"title": t} for t in titles), batch_size)):
        records = [fetch_one(client, t["title"]) for t in group]
        still_failing += sum(1 for r in records if "error" in r)
        _write_batch_gz(index + offset, records, html_dir)
        if log:
            log(f"retry batch {index + offset} written ({len(records)} pages)", flush=True)
    if log:
        log(f"retry_timeouts: re-attempted {len(titles)} pages, {still_failing} still failing",
            flush=True)
    return len(titles)


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
    falls back to the full title list if no targeted list is configured.

    The stat-page list has no generator in this repo (the shipped one was curated by hand against
    the wiki's data-template categories), so a configured-but-missing file is an operator decision
    point, not a bug to paper over: fail with the options spelled out rather than a bare
    FileNotFoundError deep in ``open()``."""
    if client is None:
        client = WikiClient(cfg)
    if titles is None:
        stat_list = cfg["paths"].get("html_titles")
        path = stat_list or cfg["paths"]["titles"]
        if stat_list and not os.path.isfile(stat_list):
            raise FileNotFoundError(
                f"Stat-page title list not found: {stat_list}. No pipeline step generates it (the "
                f"shipped list was curated by hand against the wiki's data-template categories). "
                f'Either provide the file (one {{"title": ...}} JSON object per line), or remove '
                f"`paths.html_titles` from config.yaml to fall back to the full harvested title "
                f"list ({cfg['paths']['titles']}) — note action=parse renders one page per call, "
                f"so fetching ALL ~36k titles is a ~19h pull."
            )
        titles = _read_titles(path)
    start_batch = load_checkpoint(cfg["paths"]["html_checkpoint"]) + 1
    fetch_all(client, titles, cfg, start_batch=start_batch, log=log)
