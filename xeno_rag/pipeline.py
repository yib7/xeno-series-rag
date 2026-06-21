"""End-to-end corpus build driver.

Runs the live pull + local processing + embed, in order. Each step is independently runnable and
resumable (fetch resumes from its checkpoint; embed skips chunks already in the store).

WARNING: `fetch` (and `all`) hit the live MediaWiki API for the full ~36k-article corpus, throttled
per config (~1hr). Run deliberately. Usage:

    python -m xeno_rag.pipeline all       # full build
    python -m xeno_rag.pipeline fetch     # just the (resumable) content pull
"""

import argparse
import logging

from . import harvest_titles, fetch_content, parse_wikitext, chunk, embed_index
from .config import load_config

log = logging.getLogger(__name__)

STEPS = ["harvest", "fetch", "parse", "chunk", "embed"]


def run_step(name: str, cfg: dict):
    if name == "harvest":
        n = harvest_titles.run(cfg)
        log.info("harvest: %s titles", n)
        return n
    if name == "fetch":
        fetch_content.run(cfg)
        log.info("fetch: complete")
        return None
    if name == "parse":
        stats = parse_wikitext.run(cfg)
        log.info("parse: %s", stats)
        return stats
    if name == "chunk":
        n = chunk.run(cfg)
        log.info("chunk: %s chunks", n)
        return n
    if name == "embed":
        n = embed_index.run(cfg)
        log.info("embed: collection count %s", n)
        return n
    raise ValueError(f"unknown step: {name}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build the Xeno RAG corpus.")
    parser.add_argument("step", choices=STEPS + ["all"])
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = load_config()
    steps = STEPS if args.step == "all" else [args.step]
    for step in steps:
        log.info("=== step: %s ===", step)
        run_step(step, cfg)


if __name__ == "__main__":
    main()
