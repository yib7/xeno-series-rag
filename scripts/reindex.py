"""Rebuild (or resume) the ChromaDB index from chunks.jsonl.

Resumable: chunk ids already present are skipped, so a crash/stop doesn't lose progress — just
re-run. NaN-safe: the Embedder sanitizes degenerate vectors (see _l2_normalize), so one bad chunk
can't abort the whole build. Pass --fresh to drop the collection first (needed when chunk *text*
changed under existing ids, e.g. after a re-parse). Local + free (CPU embedding).

    python scripts/reindex.py            # resume / top up
    python scripts/reindex.py --fresh    # drop + full rebuild
"""
import sys
import time

import yaml
import chromadb
from chromadb.config import Settings

from xeno_rag.embed_index import Embedder, _collection, _metadata, _iter_chunks

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
client = chromadb.PersistentClient(path=cfg["paths"]["vectorstore"], settings=Settings(anonymized_telemetry=False))
name = cfg.get("collection_name", "xeno_wiki")

if "--fresh" in sys.argv:
    try:
        client.delete_collection(name)
        print(f"dropped collection {name!r}", flush=True)
    except Exception as exc:
        print(f"no existing collection to drop ({exc})", flush=True)

col = _collection(cfg, client)
embedder = Embedder(cfg)
start_count = col.count()
print(f"backend={embedder.backend}  starting at {start_count} chunks", flush=True)

batch, added, t0 = [], 0, time.time()
BATCH = 256


def flush():
    global added
    if not batch:
        return
    ids = [c["chunk_id"] for c in batch]
    existing = set(col.get(ids=ids)["ids"])
    new = [c for c in batch if c["chunk_id"] not in existing]
    if new:
        col.add(
            ids=[c["chunk_id"] for c in new],
            embeddings=embedder.encode([c["text"] for c in new]),
            metadatas=[_metadata(c) for c in new],
            documents=[c["text"] for c in new],
        )
        added += len(new)
        if added % 5120 < BATCH:
            rate = added / (time.time() - t0)
            print(f"  added {added} new ({rate:.0f}/s, total {col.count()})", flush=True)
    batch.clear()


for chunk in _iter_chunks(cfg["paths"]["chunks"]):
    batch.append(chunk)
    if len(batch) >= BATCH:
        flush()
flush()

print(f"DONE: {col.count()} chunks total (+{added} this run) in {time.time()-t0:.0f}s", flush=True)
