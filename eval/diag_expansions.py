"""Are the Xenoblade expansions tracked to their base games?
  Future Connected (XC1 DE epilogue), Torna ~ The Golden Country (XC2), Future Redeemed (XC3).
Show how their pages are tagged (display `game`) and their membership flags in the live index, and —
for Future Connected, which has no alias/phrase — what category signals its pages actually carry."""
import re
from collections import Counter

import chromadb
from chromadb.config import Settings

from xeno_rag.config import load_config
from xeno_rag.parse_wikitext import _BASE_GAMES, _CATEGORY, derive_games, _iter_raw_pages, _page_wikitext
from xeno_rag.fetch_html import iter_html_records

cfg = load_config()
client = chromadb.PersistentClient(path=cfg["paths"]["vectorstore"], settings=Settings(anonymized_telemetry=False))
col = client.get_or_create_collection(name=cfg.get("collection_name", "xeno_wiki"), metadata={"hnsw:space": "cosine"})

# paginated metadata load
metas = []
off, page = 0, 10000
while True:
    got = col.get(include=["metadatas"], limit=page, offset=off)
    m = got.get("metadatas") or []
    if not m:
        break
    metas.extend(m)
    off += len(m)
    if len(m) < page:
        break


def flags(meta):
    return sorted(g for g in _BASE_GAMES if meta.get(f"g_{g}"))


def report(label, title_re):
    rx = re.compile(title_re, re.I)
    by_title = {}
    for m in metas:
        t = m.get("title") or ""
        if rx.search(t):
            by_title.setdefault(t, m)
    print(f"\n=== {label}: {len(by_title)} distinct pages ===")
    game_dist = Counter(m.get("game") for m in by_title.values())
    member_dist = Counter(tuple(flags(m)) for m in by_title.values())
    print(f"  display `game` distribution: {dict(game_dist)}")
    print(f"  membership distribution: { {','.join(k) or 'UBIQUITOUS': v for k, v in member_dist.items()} }")
    for t, m in list(sorted(by_title.items()))[:8]:
        print(f"    {t!r:46s} game={m.get('game')!r:9s} member={flags(m)}")


report("Future Connected '(FC)' suffix", r"\(FC\)")
report("Future Connected (name)", r"future connected")
report("Future Redeemed '(FR)' suffix", r"\(FR\)")
report("Torna '(TTGC)' suffix", r"\(TTGC\)")

# For Future Connected pages: what categories do they actually carry in raw wikitext?
print("\n=== raw categories on a few Future Connected pages ===")
wt = {}
for rec in iter_html_records(cfg["paths"]["html"]):
    if rec.get("title") and rec.get("wikitext"):
        wt[rec["title"]] = rec["wikitext"]
for p in _iter_raw_pages(cfg["paths"]["pages"]):
    t = p.get("title")
    if t and t not in wt:
        wt[t] = _page_wikitext(p)

fc_titles = [t for t in wt if re.search(r"\(FC\)|future connected", t, re.I)][:10]
for t in fc_titles:
    cats = _CATEGORY.findall(wt[t])
    print(f"  {t!r:44s} derive_games={sorted(derive_games(t, wt[t]))}  cats={cats[:5]}")
print(f"\n  (total raw pages matching FC: {sum(1 for t in wt if re.search(r'\\(FC\\)|future connected', t, re.I))})")
