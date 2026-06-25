"""Restore the index 'game' tags to the values captured in a retag snapshot ('old' column).
Fetches each chunk's CURRENT full metadata and changes only 'game', so no other field is lost."""
import json
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from xeno_rag.config import load_config

snap_path = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/retag_snapshot.jsonl")
rows = [json.loads(l) for l in snap_path.read_text(encoding="utf-8").splitlines() if l.strip()]
want = {r["chunk_id"]: r["old"] for r in rows}
print(f"Restoring {len(want)} chunks to their 'old' tag from {snap_path}")

cfg = load_config()
client = chromadb.PersistentClient(path=cfg["paths"]["vectorstore"], settings=Settings(anonymized_telemetry=False))
col = client.get_or_create_collection(name=cfg.get("collection_name", "xeno_wiki"), metadata={"hnsw:space": "cosine"})

ids = list(want)
B = 500
done = 0
for i in range(0, len(ids), B):
    batch = ids[i:i + B]
    got = col.get(ids=batch, include=["metadatas"])
    new_metas = []
    for cid, m in zip(got["ids"], got["metadatas"]):
        nm = dict(m)
        nm["game"] = want[cid]
        new_metas.append(nm)
    col.update(ids=got["ids"], metadatas=new_metas)
    done += len(got["ids"])
    print(f"  restored {done}/{len(ids)}")
print("DONE.")
