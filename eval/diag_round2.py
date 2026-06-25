"""Round-2 diagnostics: root-cause the two soft spots.

(#2) Solaris "caste-based society": the answer said it could not find the caste tiers. Is the
     content in the corpus (retrieval miss) or absent (coverage gap), and how is it tagged?
(#4) XG "Ether system": Xenosaga 'Ether (XS)' / 'Ether Amp (XS1&2)' pages surfaced under the XG
     filter. What tag do they carry, and is there a genuine XG Ether page being crowded out?
"""
import re
import chromadb
from chromadb.config import Settings
from xeno_rag.config import load_config

cfg = load_config()
client = chromadb.PersistentClient(path=cfg["paths"]["vectorstore"], settings=Settings(anonymized_telemetry=False))
col = client.get_or_create_collection(name=cfg.get("collection_name", "xeno_wiki"), metadata={"hnsw:space": "cosine"})

def load_all():
    """Paginate the whole collection (full fetch overflows ChromaDB's SQL-variable cap)."""
    metas, docs = [], []
    offset, page = 0, 2000
    while True:
        got = col.get(limit=page, offset=offset, include=["metadatas", "documents"])
        m = got.get("metadatas") or []
        if not m:
            break
        metas.extend(m)
        docs.extend(got.get("documents") or [])
        offset += len(m)
        if len(m) < page:
            break
    return metas, docs


metas, docs = load_all()
print(f"collection chunks: {len(metas)}")


def show_titles(substr, note=""):
    substr_l = substr.lower()
    seen = {}
    for m in metas:
        ti = m.get("title") or ""
        if substr_l in ti.lower():
            seen.setdefault(ti, m.get("game"))
    print(f"\n-- titles containing {substr!r} {note} ({len(seen)} distinct) --")
    for ti, g in sorted(seen.items()):
        print(f"   {ti!r:48s} game={g}")


def grep_docs(term, title_filter=None, limit=6):
    """Find chunks whose TEXT mentions term (optionally restricted to a title substring)."""
    term_l = term.lower()
    hits = []
    for m, d in zip(metas, docs):
        ti = m.get("title") or ""
        if title_filter and title_filter.lower() not in ti.lower():
            continue
        if term_l in (d or "").lower():
            hits.append((ti, m.get("game"), d))
    print(f"\n-- chunks mentioning {term!r}" + (f" in title~{title_filter!r}" if title_filter else "") + f" ({len(hits)} hits, showing {min(limit,len(hits))}) --")
    for ti, g, d in hits[:limit]:
        snippet = re.sub(r"\s+", " ", d)[:280]
        print(f"   [{g}] {ti!r}: {snippet}")
    return hits


# ---- #2 Solaris caste/class ----
print("\n" + "=" * 90 + "\n#2  SOLARIS CASTE/CLASS\n" + "=" * 90)
show_titles("Solaris")
show_titles("class", note="(any 'class' page)")
grep_docs("first class", title_filter="Solaris")
grep_docs("third class")
grep_docs("caste")
grep_docs("M-type")
grep_docs("Gazel", limit=3)

# ---- #4 XG Ether ----
print("\n" + "=" * 90 + "\n#4  XG ETHER\n" + "=" * 90)
show_titles("Ether")
grep_docs("ether", title_filter="Ether (XG)", limit=4)
grep_docs("EP", title_filter="Ether", limit=4)
