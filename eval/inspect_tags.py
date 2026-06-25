"""Phase-2 evidence: for broken vs working pages, show the signals derive_game sees —
the ordered template heads, the first STRUCTURED template's game, and the category game-codes.
This reveals exactly why KOS-MOS->XC2 etc. and what rule fixes it without flipping working pages."""
import re
from xeno_rag.config import load_config
from xeno_rag.parse_wikitext import (
    derive_game, _TEMPLATE_HEAD, _CATEGORY, _is_structured_name, _name_game,
    _canon_game, _phrase_game, _PAREN,
)
from xeno_rag.fetch_html import iter_html_records
from xeno_rag.parse_wikitext import _iter_raw_pages, _page_wikitext

cfg = load_config()

BROKEN = ["KOS-MOS", "T-elos", "Zohar (XS)", "Zohar", "Interlink (XS)"]
WORKING = ["Shulk", "Rex", "Noah", "Pyra", "Elma", "Fei", "Citan"]
WANT = set(BROKEN + WORKING)

# Collect wikitext by title from HTML records first (stat pages), then raw pages.
wt = {}
for rec in iter_html_records(cfg["paths"]["html"]):
    t = rec.get("title")
    if t in WANT and rec.get("wikitext"):
        wt[t] = rec["wikitext"]
for page in _iter_raw_pages(cfg["paths"]["pages"]):
    t = page.get("title")
    if t in WANT and t not in wt:
        c = _page_wikitext(page)
        if c:
            wt[t] = c
    if len(wt) == len(WANT):
        break


def cat_codes(wikitext):
    codes = []
    for cat in _CATEGORY.findall(wikitext):
        c = None
        for token in _PAREN.findall(cat):
            c = _canon_game(token) or c
        c = c or _phrase_game(cat)
        if c:
            codes.append((cat.strip()[:40], c))
    return codes


def first_structured(names):
    for n in names:
        if _is_structured_name(n):
            g = _name_game(n)
            return n, g
    return None, None


def show(title):
    w = wt.get(title)
    if not w:
        print(f"\n### {title!r}: (wikitext NOT FOUND)")
        return
    names = [n.strip() for n in _TEMPLATE_HEAD.findall(w)]
    structured = [n for n in names if _is_structured_name(n)]
    fs_name, fs_game = first_structured(names)
    cats = cat_codes(w)
    distinct_cat = sorted({c for _, c in cats})
    print(f"\n### {title!r}  -> derive_game = {derive_game(title, w)!r}")
    print(f"    first 6 template heads: {names[:6]}")
    print(f"    structured templates:   {structured[:6]}")
    print(f"    FIRST structured -> name={fs_name!r} game={fs_game!r}  (this is what step 2 returns)")
    print(f"    category game-codes:    {cats[:10]}")
    print(f"    distinct category games: {distinct_cat}")


print("=" * 30, "BROKEN", "=" * 30)
for t in BROKEN:
    show(t)
print("\n" + "=" * 30, "WORKING", "=" * 30)
for t in WORKING:
    show(t)
