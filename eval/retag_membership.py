"""Add multi-tag membership flags (`g_<game>`) to every chunk in the live index — METADATA ONLY.

The single `game` display tag is left untouched (it is baked into the embedded breadcrumb text, so
changing it would need a re-embed). This script recomputes each page's membership SET via
`derive_games` from the same raw wikitext the pipeline used, and writes one boolean flag per member
game so the new membership filter (`g_<game> = True`) can find cross-appearance pages under EACH of
their games (KOS-MOS -> XS1/XS2/XS3/XC2) and ubiquitous pages under all.

Idempotent: rewrites a chunk only when its membership flags differ from what's stored. Always writes
a snapshot of the prior `game` + membership per changed chunk for reversibility.

Usage:
  python eval/retag_membership.py            # dry run: compute, summarize, snapshot
  python eval/retag_membership.py --apply    # also write the metadata updates
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import chromadb
from chromadb.config import Settings

from xeno_rag.config import load_config
from xeno_rag.parse_wikitext import derive_games, membership_flags, _BASE_GAMES, _iter_raw_pages, _page_wikitext
from xeno_rag.fetch_html import iter_html_records

APPLY = "--apply" in sys.argv
SNAP = Path("eval/retag_membership_snapshot.jsonl")
cfg = load_config()

print("Recomputing membership from raw corpus...")
html_member, raw_member = {}, {}
for rec in iter_html_records(cfg["paths"]["html"]):
    t, wt = rec.get("title"), rec.get("wikitext")
    if t and wt:
        html_member[t] = sorted(derive_games(t, wt))
for page in _iter_raw_pages(cfg["paths"]["pages"]):
    t = page.get("title")
    if t and t not in html_member:
        raw_member[t] = sorted(derive_games(t, _page_wikitext(page)))
print(f"  HTML titles: {len(html_member)}  wikitext titles: {len(raw_member)}")


def member_for(title):
    m = html_member.get(title, raw_member.get(title))
    return m if m is not None else None


def desired_flags(member):
    # empty/None membership -> ubiquitous (a flag for every base game)
    return membership_flags(member)


client = chromadb.PersistentClient(path=cfg["paths"]["vectorstore"], settings=Settings(anonymized_telemetry=False))
col = client.get_or_create_collection(name=cfg.get("collection_name", "xeno_wiki"), metadata={"hnsw:space": "cosine"})
total = col.count()
print(f"Index chunks: {total}")

PAGE = 10000
offset = 0
updates = []          # (chunk_id, new_full_metadata)
snapshot = []         # (chunk_id, old_member_list, new_member_list)
size_hist = defaultdict(int)   # membership size -> #chunks
missing = set()

while offset < total:
    got = col.get(include=["metadatas"], limit=PAGE, offset=offset)
    for cid, m in zip(got["ids"], got["metadatas"]):
        title = m.get("title")
        member = member_for(title)
        if member is None:
            missing.add(title)
            member = []                       # ubiquitous fallback (never hide)
        flags = desired_flags(member)
        cur_flags = {k: v for k, v in m.items() if k.startswith("g_")}
        if cur_flags != flags:
            nm = {k: v for k, v in m.items() if not k.startswith("g_")}
            nm.update(flags)
            updates.append((cid, nm))
            old_member = sorted(g for g in _BASE_GAMES if cur_flags.get(f"g_{g}"))
            snapshot.append((cid, old_member, member if member else sorted(_BASE_GAMES)))
        size_hist[len(member) if member else len(_BASE_GAMES)] += 1
    offset += PAGE
    print(f"  scanned {min(offset, total)}/{total}")

multi = sum(n for s, n in size_hist.items() if 1 < s < len(_BASE_GAMES))
print(f"\nChunks needing membership write: {len(updates)}")
print(f"Membership-size histogram (chunks): {dict(sorted(size_hist.items()))}")
print(f"  -> chunks with a genuine multi-game (2..7) membership: {multi}")
if missing:
    print(f"Titles in index but not recomputed (-> ubiquitous): {len(missing)}  e.g. {list(missing)[:4]}")

with SNAP.open("w", encoding="utf-8") as f:
    for cid, old, new in snapshot:
        f.write(json.dumps({"chunk_id": cid, "old": old, "new": new}) + "\n")
print(f"Wrote snapshot: {SNAP} ({len(snapshot)} rows)")

if not APPLY:
    print("\nDRY RUN — no changes written. Re-run with --apply.")
    sys.exit(0)

print("\nApplying membership metadata (no re-embed)...")
B = 1000
for i in range(0, len(updates), B):
    batch = updates[i:i + B]
    col.update(ids=[c for c, _ in batch], metadatas=[mm for _, mm in batch])
    print(f"  updated {min(i + B, len(updates))}/{len(updates)}")
print("DONE. Membership flags applied.")
