"""Re-tag the live ChromaDB index to match the corrected derive_game — METADATA ONLY, no re-embed.

The game tag is pure metadata, so fixing it is a `collection.update`; embeddings (the expensive part)
are untouched. Recomputes each page's tag from the SAME wikitext source the hybrid pipeline used
(HTML-record wikitext where we have it, else raw-page wikitext), diffs against the index, and (with
--apply) updates the changed chunks. Always writes a snapshot of prior tags for reversibility.

Usage:
  python eval/retag.py            # dry run: compute diff, print transitions, write snapshot
  python eval/retag.py --apply    # also apply the metadata updates
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import chromadb
from chromadb.config import Settings

from xeno_rag.config import load_config
from xeno_rag.parse_wikitext import derive_game, _iter_raw_pages, _page_wikitext
from xeno_rag.fetch_html import iter_html_records

APPLY = "--apply" in sys.argv
# Optional snapshot path (preserve a prior cycle's snapshot by passing a fresh name).
SNAP_NAME = next((a for a in sys.argv[1:] if a.endswith(".jsonl")), "eval/retag_snapshot.jsonl")
cfg = load_config()

# --- 1. recompute the correct tag per title from raw data (HTML wikitext wins, else raw) ---
print("Recomputing tags from raw corpus...")
html_tag = {}
for rec in iter_html_records(cfg["paths"]["html"]):
    wt = rec.get("wikitext")
    t = rec.get("title")
    if t and wt:
        html_tag[t] = derive_game(t, wt)
print(f"  HTML-set titles with wikitext: {len(html_tag)}")

raw_tag = {}
for page in _iter_raw_pages(cfg["paths"]["pages"]):
    t = page.get("title")
    if t and t not in html_tag:
        raw_tag[t] = derive_game(t, _page_wikitext(page))
print(f"  wikitext-set titles: {len(raw_tag)}")


def new_tag(title):
    if title in html_tag:
        return html_tag[title]
    return raw_tag.get(title)


# --- 2. load index metadata, paged (avoid huge IN-clauses) ---
client = chromadb.PersistentClient(path=cfg["paths"]["vectorstore"], settings=Settings(anonymized_telemetry=False))
col = client.get_or_create_collection(name=cfg.get("collection_name", "xeno_wiki"), metadata={"hnsw:space": "cosine"})
total = col.count()
print(f"Index chunks: {total}")

PAGE = 10000
offset = 0
# chunk_id -> (current_meta dict)
changes = []                       # (chunk_id, full_meta_with_new_game)
snapshot = []                      # (chunk_id, old_game, new_game)
transitions = defaultdict(int)     # (old,new) -> count
changed_titles = defaultdict(lambda: [0, None, None])  # title -> [nchunks, old, new]
missing_titles = set()

while offset < total:
    got = col.get(include=["metadatas"], limit=PAGE, offset=offset)
    ids = got["ids"]
    metas = got["metadatas"]
    for cid, m in zip(ids, metas):
        title = m.get("title")
        old = m.get("game")
        nt = new_tag(title)
        if nt is None:
            missing_titles.add(title)
            continue
        if nt != old:
            nm = dict(m)
            nm["game"] = nt
            changes.append((cid, nm))
            snapshot.append((cid, old, nt))
            transitions[(old, nt)] += 1
            row = changed_titles[title]
            row[0] += 1
            row[1], row[2] = old, nt
    offset += PAGE
    print(f"  scanned {min(offset, total)}/{total}")

print(f"\nChunks needing re-tag: {len(changes)}  across {len(changed_titles)} pages")
if missing_titles:
    print(f"Titles in index but not recomputed (left unchanged): {len(missing_titles)}  e.g. {list(missing_titles)[:5]}")

print("\nTransition matrix (old -> new : chunk count):")
for (old, nt), n in sorted(transitions.items(), key=lambda kv: -kv[1]):
    print(f"  {str(old):8s} -> {str(nt):8s} : {n}")

# sample of changed pages, grouped by transition, base-game -> base-game ones first (highest risk)
def is_base(g):
    return g in {"XG", "XS1", "XS2", "XS3", "XC1", "XC2", "XC3", "XCX"}

print("\nSample changed pages (base-game -> different base-game = inspect closely):")
shown = defaultdict(int)
for title, (n, old, nt) in sorted(changed_titles.items()):
    key = (old, nt)
    if is_base(old) and is_base(nt) and shown[key] < 6:
        shown[key] += 1
        print(f"  [{old}->{nt}] {title!r} ({n} chunks)")

print("\nSample base->series and series->base (recall-improving):")
shown2 = defaultdict(int)
for title, (n, old, nt) in sorted(changed_titles.items()):
    key = (old, nt)
    if (old == "series") != (nt == "series") and shown2[key] < 4:
        shown2[key] += 1
        print(f"  [{old}->{nt}] {title!r} ({n} chunks)")

# --- 3. snapshot + (optionally) apply ---
snap_path = Path(SNAP_NAME)
with snap_path.open("w", encoding="utf-8") as f:
    for cid, old, nt in snapshot:
        f.write(json.dumps({"chunk_id": cid, "old": old, "new": nt}) + "\n")
print(f"\nWrote reversibility snapshot: {snap_path} ({len(snapshot)} rows)")

if not APPLY:
    print("\nDRY RUN — no changes written. Re-run with --apply to update the index.")
    sys.exit(0)

print("\nApplying metadata updates (no re-embed)...")
B = 1000
for i in range(0, len(changes), B):
    batch = changes[i:i + B]
    col.update(ids=[c for c, _ in batch], metadatas=[m for _, m in batch])
    print(f"  updated {min(i + B, len(changes))}/{len(changes)}")
print("DONE. Re-tag applied.")
