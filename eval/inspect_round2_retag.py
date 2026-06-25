"""List every page the round-2 XS re-tag would change, to eyeball for wrongful demotions.

A page going series->XS or XS1->XS loses visibility under XG / Xenoblade filters. That is correct
ONLY if the page is genuinely Xenosaga-only. Print all changed titles grouped by transition so a
human can confirm none is a cross-franchise page that should stay 'series'.
"""
from collections import defaultdict

import chromadb
from chromadb.config import Settings

from xeno_rag.config import load_config
from xeno_rag.parse_wikitext import derive_game, _iter_raw_pages, _page_wikitext
from xeno_rag.fetch_html import iter_html_records

cfg = load_config()
html_tag = {}
for rec in iter_html_records(cfg["paths"]["html"]):
    if rec.get("title") and rec.get("wikitext"):
        html_tag[rec["title"]] = derive_game(rec["title"], rec["wikitext"])
raw_tag = {}
for page in _iter_raw_pages(cfg["paths"]["pages"]):
    t = page.get("title")
    if t and t not in html_tag:
        raw_tag[t] = derive_game(t, _page_wikitext(page))


def new_tag(title):
    return html_tag.get(title, raw_tag.get(title))


client = chromadb.PersistentClient(path=cfg["paths"]["vectorstore"], settings=Settings(anonymized_telemetry=False))
col = client.get_or_create_collection(name=cfg.get("collection_name", "xeno_wiki"), metadata={"hnsw:space": "cosine"})

total = col.count()
offset, PAGE = 0, 10000
changed = defaultdict(lambda: [0, None, None])  # title -> [nchunks, old, new]
while offset < total:
    got = col.get(include=["metadatas"], limit=PAGE, offset=offset)
    for m in got["metadatas"]:
        title, old = m.get("title"), m.get("game")
        nt = new_tag(title)
        if nt is not None and nt != old:
            row = changed[title]
            row[0] += 1
            row[1], row[2] = old, nt
    offset += PAGE

by_trans = defaultdict(list)
for title, (n, old, nt) in changed.items():
    by_trans[(old, nt)].append((title, n))

for trans in sorted(by_trans, key=lambda k: (k[0] or "", k[1] or "")):
    titles = sorted(by_trans[trans])
    print(f"\n=== {trans[0]} -> {trans[1]}  ({len(titles)} pages) ===")
    for title, n in titles:
        print(f"   {title!r}  ({n})")
