"""Diagnostic: for pages that went MISSING from retrieval, what game tag did they get?
Determines whether the failure is a TAGGING bug (filtered out) or a RETRIEVAL-recall bug."""
import sys
import chromadb
from chromadb.config import Settings
from xeno_rag.config import load_config

cfg = load_config()
client = chromadb.PersistentClient(path=cfg["paths"]["vectorstore"], settings=Settings(anonymized_telemetry=False))
col = client.get_or_create_collection(name=cfg.get("collection_name", "xeno_wiki"), metadata={"hnsw:space": "cosine"})

# Exact-title lookups for the canonical pages that should have answered the failing questions.
TITLES = [
    "KOS-MOS", "T-elos", "Mimeosome", "Mimeosomes", "Boost", "Boost (XS1)",
    "Zohar (XS)", "Zohar (XG)", "Zohar", "Interlink", "Interlink (XS)", "Skell",
]
for t in TITLES:
    got = col.get(where={"title": t}, limit=50)
    metas = got.get("metadatas", []) or []
    if not metas:
        print(f"  {t!r:24s} -> NOT FOUND (0 chunks with this exact title)")
        continue
    games = sorted({m.get("game") for m in metas})
    headings = [m.get("heading") for m in metas]
    print(f"  {t!r:24s} -> {len(metas):2d} chunks, game(s)={games}, headings={headings[:8]}")

# Also: fuzzy — any title containing 'imeosome' (case the page is named differently)?
print("\n-- any title containing 'imeosome' --")
big = col.get(where={"game": {"$in": ["XCX", "series"]}}, limit=40000)
hits = [(m.get("title"), m.get("game")) for m in (big.get("metadatas") or []) if "imeosome" in (m.get("title") or "").lower()]
seen = set()
for title, game in hits:
    if title not in seen:
        seen.add(title)
        print(f"   {title!r} game={game}")
print(f"   ({len(seen)} distinct XCX/series titles contain 'imeosome')")

# And: is there a Mimeosome page under ANY tag (maybe filtered out)?
print("\n-- 'imeosome' across ALL tags --")
allm = col.get(limit=200000)
hits2 = {}
for m in (allm.get("metadatas") or []):
    ti = m.get("title") or ""
    if "imeosome" in ti.lower():
        hits2[ti] = m.get("game")
for ti, g in sorted(hits2.items()):
    print(f"   {ti!r} game={g}")
print(f"   ({len(hits2)} distinct titles contain 'imeosome' across all tags)")
