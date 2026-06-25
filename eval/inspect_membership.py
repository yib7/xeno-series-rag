"""Evidence for the multi-tag membership design: for sample pages, show the SET of base games their
categories/templates/suffix imply, vs the current single 'game' tag. Confirms 'all game-bearing
categories = true membership' before we rebuild tagging around it."""
import re

from xeno_rag.config import load_config
from xeno_rag.parse_wikitext import (
    _CATEGORY, _PAREN, _canon_game, _phrase_game, _name_game, _is_structured_name,
    _iter_raw_pages, _page_wikitext, derive_game,
)
from xeno_rag.fetch_html import iter_html_records

cfg = load_config()

# index raw wikitext by title (HTML record wins, else raw page) — same source retag.py uses
wt_by_title = {}
for rec in iter_html_records(cfg["paths"]["html"]):
    if rec.get("title") and rec.get("wikitext"):
        wt_by_title[rec["title"]] = rec["wikitext"]
for page in _iter_raw_pages(cfg["paths"]["pages"]):
    t = page.get("title")
    if t and t not in wt_by_title:
        wt_by_title[t] = _page_wikitext(page)


def category_games(wt):
    out = []
    for cat in _CATEGORY.findall(wt):
        c = None
        for token in _PAREN.findall(cat):
            c = _canon_game(token) or c
        c = c or _phrase_game(cat)
        if c:
            out.append(c)
    return out


def template_games(wt):
    from xeno_rag.parse_wikitext import _TEMPLATE_HEAD
    names = [n.strip() for n in _TEMPLATE_HEAD.findall(wt)]
    return {g for n in names if _is_structured_name(n) and (g := _name_game(n))}


SAMPLES = [
    "KOS-MOS", "T-elos", "Shion", "Elma", "Shulk", "Fiora", "Poppi", "chaos",
    "Rex", "Pyra", "Noah", "Mio", "Fei", "Citan Uzuki", "Metal Face", "Zohar",
    "Ether (XS)", "Ether (XC1)", "Mythra", "Nia",
]

for t in SAMPLES:
    wt = wt_by_title.get(t)
    if wt is None:
        print(f"{t!r:22s} -> (not in corpus)")
        continue
    cats = category_games(wt)
    tmpls = template_games(wt)
    member = sorted(set(cats) | tmpls)
    print(f"{t!r:22s} game={derive_game(t, wt)!r:9s} cats(ordered)={cats}  tmpl={sorted(tmpls)}  => MEMBER={member}")
