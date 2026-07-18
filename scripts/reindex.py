"""Rebuild (or resume) the ChromaDB index from chunks.jsonl.

Resumable: chunk ids already present are skipped, so a crash/stop doesn't lose progress — just
re-run. NaN-safe: the Embedder sanitizes degenerate vectors (see _l2_normalize), so one bad chunk
can't abort the whole build. Pass --fresh to drop the collection first (needed when chunk *text*
changed under existing ids, e.g. after a re-parse). Local + free (CPU embedding).

    python scripts/reindex.py            # resume / top up
    python scripts/reindex.py --fresh    # drop + full rebuild

The actual build is delegated to ``xeno_rag.embed_index.build_index`` (same flush loop the rest of
the pipeline uses), so this script inherits its skip-with-log guard for a chunk with a missing
pageid instead of re-implementing (and drifting from) its own copy.
"""
import sys

import chromadb
import yaml
from chromadb.config import Settings

from xeno_rag import embed_index
from xeno_rag.embed_index import Embedder


def main(cfg=None, client=None, embedder=None, argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if cfg is None:
        cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    if client is None:
        client = chromadb.PersistentClient(
            path=cfg["paths"]["vectorstore"], settings=Settings(anonymized_telemetry=False)
        )
    if embedder is None:
        embedder = Embedder(cfg)

    if "--fresh" in argv:
        embed_index.drop_collection(cfg, client)

    start_count = embed_index._collection(cfg, client).count()
    backend = getattr(embedder, "backend", "n/a")
    print(f"backend={backend}  starting at {start_count} chunks", flush=True)

    total = embed_index.build_index(
        embed_index._iter_chunks(cfg["paths"]["chunks"]), cfg, embedder=embedder, client=client
    )

    print(f"DONE: {total} chunks total", flush=True)
    return total


if __name__ == "__main__":
    main()
