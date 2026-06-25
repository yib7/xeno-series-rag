"""Root-cause: under the XC1 filter, which KOS-MOS-named pages come back, and what membership do they
carry? Distinguish the main 'KOS-MOS' bio (must be excluded) from adjacent spinoff/music/episode
pages with no base-game category (ubiquitous by design)."""
import chromadb
from chromadb.config import Settings

from xeno_rag.config import load_config
from xeno_rag.embed_index import Embedder
from xeno_rag.retrieve import retrieve
from xeno_rag.parse_wikitext import _BASE_GAMES

cfg = load_config()
client = chromadb.PersistentClient(path=cfg["paths"]["vectorstore"], settings=Settings(anonymized_telemetry=False))
col = client.get_or_create_collection(name=cfg.get("collection_name", "xeno_wiki"), metadata={"hnsw:space": "cosine"})


def flags(meta):
    return sorted(g for g in _BASE_GAMES if meta.get(f"g_{g}"))


print("=== stored membership flags for KOS-MOS-named pages ===")
got = col.get(where={"title": "KOS-MOS"}, include=["metadatas"], limit=20)
for m in (got["metadatas"] or [])[:3]:
    print(f"  'KOS-MOS' chunk: game={m.get('game')!r} membership={flags(m)}")

for t in ["KOS-MOS (Xeno-pittan)", "KOS-MOS (episode)", "KOS-MOS Activating", "Here she is (KOS-MOS)", "Ai"]:
    g = col.get(where={"title": t}, include=["metadatas"], limit=3)
    ms = g["metadatas"] or []
    if ms:
        print(f"  {t!r}: game={ms[0].get('game')!r} membership={flags(ms[0])}")
    else:
        print(f"  {t!r}: (not found)")

print("\n=== retrieve('Who is KOS-MOS?', game_filter='XC1') -> titles + XC1 membership ===")
emb = Embedder(cfg)
chunks = retrieve("Who is KOS-MOS?", cfg, game_filter="XC1", embedder=emb)
for c in chunks:
    g = col.get(where={"title": c["title"]}, include=["metadatas"], limit=1)
    m = (g["metadatas"] or [{}])[0]
    print(f"  {c['title']!r:34s} membership={flags(m)}")
main_present = any(c["title"] == "KOS-MOS" for c in chunks)
print(f"\nMAIN 'KOS-MOS' bio present under XC1 (must be False): {main_present}")
