"""Parse raw wikitext into prose sections + structured infobox data.

Keeps the valuable structured data (infobox template params) that plain-text extraction throws away,
and produces clean prose with links/refs/markup resolved.
"""

import glob
import json
import os
import re

import mwparserfromhell

BASE_WIKI_URL = "https://www.xenoserieswiki.org/wiki/"

# Known game codes used as title suffixes, e.g. "Infinity Blade (XC3) (Noah)".
KNOWN_GAMES = {
    "XG",                       # Xenogears
    "XS1", "XS2", "XS3",        # Xenosaga I/II/III
    "XC1", "XC2", "XC3",        # Xenoblade Chronicles 1/2/3
    "XCX",                      # Xenoblade Chronicles X
    "XCDE", "XC1DE",            # XC1 Definitive Edition
    "XC2T",                     # XC2: Torna
    "XC3FR",                    # XC3: Future Redeemed
}

_PAREN = re.compile(r"\(([^()]+)\)")
_REF_SELF = re.compile(r"<ref[^>]*?/>", re.IGNORECASE)
_REF_PAIR = re.compile(r"<ref[^>]*?>.*?</ref>", re.IGNORECASE | re.DOTALL)


def derive_game(title: str) -> str:
    """Return the game code from a title's parenthetical suffix, or 'series'."""
    for token in _PAREN.findall(title):
        code = token.strip().upper()
        if code in KNOWN_GAMES:
            return code
    return "series"


def title_to_url(title: str) -> str:
    return BASE_WIKI_URL + title.replace(" ", "_")


def _strip_refs(wikitext: str) -> str:
    wikitext = _REF_SELF.sub("", wikitext)
    wikitext = _REF_PAIR.sub("", wikitext)
    return wikitext


def _extract_infoboxes(code):
    boxes = []
    for tmpl in code.filter_templates():
        name = str(tmpl.name).strip()
        if "infobox" in name.lower():
            fields = {}
            for param in tmpl.params:
                key = param.name.strip_code().strip()
                val = param.value.strip_code().strip()
                if key:
                    fields[key] = val
            boxes.append({"template": name, "fields": fields})
    return boxes


def _extract_sections(code):
    sections = []
    for sec in code.get_sections(levels=[2], include_lead=True, include_headings=True, flat=True):
        headings = sec.filter_headings()
        if headings:
            heading = headings[0].title.strip_code().strip()
        else:
            heading = "Introduction"
        text = sec.strip_code().strip()
        # strip_code renders the heading title as the first line; drop it
        if headings and text.startswith(heading):
            text = text[len(heading):].lstrip()
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            sections.append({"heading": heading, "text": text})
    return sections


def _is_disambiguation(code) -> bool:
    for tmpl in code.filter_templates():
        if "disambig" in str(tmpl.name).strip().lower():
            return True
    return False


def parse_article(title: str, pageid, wikitext, cfg: dict):
    """Parse one page. Returns a record dict, or None if it should be dropped."""
    if not wikitext:
        return None
    stripped = wikitext.strip()
    if stripped.lower().startswith("#redirect"):
        return None
    min_bytes = cfg.get("min_wikitext_bytes", 50)
    if len(stripped) < min_bytes:
        return None

    code = mwparserfromhell.parse(_strip_refs(wikitext))
    if _is_disambiguation(code):
        return None

    return {
        "title": title,
        "pageid": pageid,
        "game": derive_game(title),
        "url": title_to_url(title),
        "infoboxes": _extract_infoboxes(code),
        "sections": _extract_sections(code),
    }


def _iter_raw_pages(pages_dir: str):
    for path in sorted(glob.glob(os.path.join(pages_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _page_wikitext(page: dict):
    revs = page.get("revisions")
    if not revs:
        return None
    try:
        return revs[0]["slots"]["main"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def run(cfg: dict, raw_pages=None) -> dict:
    """Parse all raw pages → articles.jsonl. Returns {written, dropped}."""
    if raw_pages is None:
        raw_pages = _iter_raw_pages(cfg["paths"]["pages"])
    out_path = cfg["paths"]["articles"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    written = dropped = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for page in raw_pages:
            art = parse_article(
                page.get("title"), page.get("pageid"), _page_wikitext(page), cfg
            )
            if art is None:
                dropped += 1
                continue
            out.write(json.dumps(art, ensure_ascii=False) + "\n")
            written += 1
    return {"written": written, "dropped": dropped}
