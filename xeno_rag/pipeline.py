"""End-to-end corpus build driver.

Runs the live pull + local processing + embed, in order. Each step is independently runnable and
resumable (fetch resumes from its checkpoint; embed skips chunks already in the store).

WARNING: `fetch` (and `all`) hit the live MediaWiki API for the full ~36k-article corpus. Rendered
HTML comes from `action=parse`, which renders ONE page per call (the Lua-decoded stat tables exist
only in the HTML), so a full pull is ~19h at the configured throttle. It is resumable. Run
deliberately. Usage:

    python -m xeno_rag.pipeline all       # full build (HTML fetch -> parse -> chunk -> embed)
    python -m xeno_rag.pipeline fetch     # just the (resumable) rendered-HTML pull
"""

import argparse
import logging

from . import harvest_titles, fetch_html, parse_html, chunk, embed_index, bm25_index
from .config import load_config

log = logging.getLogger(__name__)

STEPS = ["harvest", "fetch", "parse", "chunk", "embed", "bm25"]
# `rebuild` skips harvest (titles already exist): re-fetch stat-page HTML, re-merge, re-chunk, a
# FRESH embed (drop + rebuild) because re-parsed text changes under existing chunk ids, then the BM25
# lexical index (built from the embedded collection, so its tags match).
META = {"all": STEPS, "rebuild": ["fetch", "parse", "chunk", "embed_fresh", "bm25"]}


def run_step(name: str, cfg: dict):
    if name == "harvest":
        n = harvest_titles.run(cfg)
        log.info("harvest: %s titles", n)
        return n
    if name == "fetch":
        fetch_html.run(cfg, log=lambda m, **k: log.info("fetch: %s", m))
        log.info("fetch: complete")
        return None
    if name == "parse":
        stats = parse_html.run_hybrid(cfg)   # HTML for stat pages, wikitext for the rest, merged
        log.info("parse: %s", stats)
        return stats
    if name == "chunk":
        n = chunk.run(cfg)
        log.info("chunk: %s chunks", n)
        return n
    if name in ("embed", "embed_fresh"):
        if name == "embed_fresh":
            embed_index.drop_collection(cfg)
            log.info("embed: dropped collection for a fresh rebuild")
        n = embed_index.run(cfg)
        log.info("embed: collection count %s", n)
        return n
    if name == "bm25":
        n = bm25_index.run(cfg)
        log.info("bm25: lexical index built over %s chunks", n)
        return n
    raise ValueError(f"unknown step: {name}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build the Xeno RAG corpus.")
    # embed_fresh is also exposed directly (not just inside `rebuild`) so a drop+rebuild can be
    # (re)run on its own — e.g. resuming after fetch/parse/chunk already completed.
    parser.add_argument("step", choices=STEPS + ["embed_fresh"] + list(META))
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = load_config()
    steps = META.get(args.step, [args.step])
    for step in steps:
        log.info("=== step: %s ===", step)
        run_step(step, cfg)


if __name__ == "__main__":
    main()
