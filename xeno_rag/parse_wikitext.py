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

# Variant / sub-release codes fold into their base game so every chunk's tag matches a UI filter
# option (the game selector only offers the eight base games).
_GAME_ALIASES = {
    "XCDE": "XC1", "XC1DE": "XC1",   # Xenoblade Chronicles: Definitive Edition
    "XC2T": "XC2",                    # Xenoblade Chronicles 2: Torna - The Golden Country
    "XC3FR": "XC3",                   # Xenoblade Chronicles 3: Future Redeemed
}
_BASE_GAMES = {"XG", "XS1", "XS2", "XS3", "XC1", "XC2", "XC3", "XCX"}
_XS_EPISODES = {"XS1", "XS2", "XS3"}  # the Xenosaga trilogy shares one recurring cast (see derive_game)

_CATEGORY = re.compile(r"\[\[\s*Category\s*:\s*([^\]|\n]+)", re.IGNORECASE)
_TEMPLATE_HEAD = re.compile(r"\{\{\s*([^|{}\n]+)")

# Verbose game phrasings (as they appear in category names) -> base code. Ordered most-specific
# first so "Xenoblade Chronicles 3" isn't swallowed by a broader "Xenoblade Chronicles" match.
_PHRASE_GAMES = [
    (re.compile(r"future\s+redeemed", re.I), "XC3"),
    (re.compile(r"xenoblade\s+chronicles\s*(?:3|iii)\b", re.I), "XC3"),
    (re.compile(r"\btorna\b", re.I), "XC2"),
    (re.compile(r"xenoblade\s+chronicles\s*(?:2|ii)\b", re.I), "XC2"),
    (re.compile(r"xenoblade\s+chronicles\s*x\b", re.I), "XCX"),
    (re.compile(r"xenoblade\s+chronicles(?:\s*(?:1|i)\b|\s*:?\s*definitive)", re.I), "XC1"),
    (re.compile(r"xenosaga\s+episode\s*(?:iii|3)\b", re.I), "XS3"),
    (re.compile(r"xenosaga\s+episode\s*(?:ii|2)\b", re.I), "XS2"),
    (re.compile(r"xenosaga\s+episode\s*(?:i|1)\b", re.I), "XS1"),
    (re.compile(r"\bxenogears\b", re.I), "XG"),
]


def _canon_game(token: str):
    """Normalize a raw code token to a base game code, or None if it isn't a game code."""
    code = token.strip().upper()
    code = _GAME_ALIASES.get(code, code)
    return code if code in _BASE_GAMES else None


def _name_game(name: str):
    """The base game code carried by a template name token (``XC1 enemy data`` or
    ``Infobox XC2 blade``), or None."""
    for tok in name.split():
        c = _canon_game(tok)
        if c:
            return c
    return None


def _phrase_game(text: str):
    for pat, code in _PHRASE_GAMES:
        if pat.search(text):
            return code
    return None


def _is_structured_name(name: str) -> bool:
    low = name.lower()
    return "infobox" in low or low.endswith(" data") or low.endswith(" stats")


_XS_WIDE_TOKEN = re.compile(r"XS(?:[1-3](?:&[1-3])+)?$")


def _is_xs_wide(token: str) -> bool:
    """An explicit Xenosaga-*wide* title-suffix token: bare ``XS`` (Xenosaga-generic) or a
    cross-episode combo ``XS1&2`` / ``XS2&3`` / ``XS1&2&3``. A single ``XS1`` is one base episode
    (handled by ``_canon_game``), so it is intentionally NOT matched here."""
    up = token.strip().upper().replace(" ", "")
    return bool(_XS_WIDE_TOKEN.fullmatch(up))


def _is_xeno_generic(name: str) -> bool:
    """A generic / cross-installment Xenosaga marker that names no single base game: bare ``{{XS}}``
    link shortcuts, ``{{ArticleIcon/XS…}}`` banners, ``{{XS1&2}}``-style cross codes. These are
    invisible to ``_name_game`` (the wiki has no single ``XS`` base game — only XS1/XS2/XS3), so a
    Xenosaga page that merely cross-references Xenogears/Xenoblade would otherwise be tagged by that
    *other* game. Detecting the marker lets us keep such a page in 'series' instead."""
    n = name.strip()
    up = n.upper()
    if up == "XS" or up.startswith("XS1&") or up.startswith("XS2&") or up == "XS&":
        return True
    return n.lower().startswith("articleicon/xs")


def derive_game(title: str, wikitext: str = None) -> str:
    """Tag a page with the single Xeno game it belongs to, or 'series' for cross-game pages.

    Priority, most authoritative first:
      1. An explicit ``(XCn)`` suffix in the title — the wiki author disambiguated it by hand.
      2. The page's own structured template (``{{XC3 character infobox}}``, ``{{XC1 enemy data}}``)
         — decisive when the page's coded infobox/data templates name exactly one game.
      3. The page's HOME game from its categories. The wiki lists a page's own-game category first
         and cross-appearance categories after, so for an *entity* page (one with an infobox) the
         first game-bearing category is its home game — a later cameo category must not steal the
         tag. A page with **no** infobox that still spans several game categories is genuine
         cross-game lore and stays 'series'.
      4. Game-prefixed templates as a last resort, but a generic Xenosaga marker blocks a lone
         foreign cross-reference from hijacking the tag.

    A page that belongs to one game no longer leaks into the others, and — crucially — a character
    who debuts in one subseries but cameos in another is no longer hidden from her home filter.
    """
    for token in _PAREN.findall(title):
        c = _canon_game(token)
        if c:
            return c
    # An explicit Xenosaga-wide suffix ('(XS)', '(XS1&2)') is a Xenosaga-only page: tag it the 'XS'
    # umbrella so it shows under every Xenosaga filter but no longer leaks into XG/Xenoblade filters
    # the way the all-franchises 'series' did (e.g. 'Ether (XS)' surfacing under a Xenogears query).
    for token in _PAREN.findall(title):
        if _is_xs_wide(token):
            return "XS"
    if not wikitext:
        return "series"

    names = [n.strip() for n in _TEMPLATE_HEAD.findall(wikitext)]
    # 2. The page's own coded infobox / stat-data template, decisive when it names exactly one game
    #    (so a secondary cameo stat-block can't override the primary subject).
    structured_games = {g for name in names if _is_structured_name(name) and (g := _name_game(name))}
    if len(structured_games) == 1:
        return next(iter(structured_games))

    # 3. Categories, in document order. The first game-bearing one is the page's home game.
    ordered_cat_games = []
    for cat in _CATEGORY.findall(wikitext):
        c = None
        for token in _PAREN.findall(cat):
            c = _canon_game(token) or c
        c = c or _phrase_game(cat)
        if c:
            ordered_cat_games.append(c)
    has_infobox = any("infobox" in name.lower() for name in names)
    distinct_cats = set(ordered_cat_games)
    if has_infobox and ordered_cat_games:
        # A character in two or more Xenosaga *episodes* is a recurring series-wide lead (the
        # Xenosaga trilogy shares one continuous cast — Shion/KOS-MOS/Jr. are leads in all three),
        # so 'series' keeps her under every Xenosaga filter. Xenoblade games have distinct casts, so
        # a Xenoblade character keeps her single home game (the first category) instead of leaking.
        if len(_XS_EPISODES & distinct_cats) >= 2:
            # Purely Xenosaga categories -> the 'XS' umbrella (under every Xenosaga filter, excluded
            # from XG/Xenoblade). A non-Xenosaga cameo category means it genuinely spans franchises.
            return "XS" if distinct_cats <= _XS_EPISODES else "series"
        return ordered_cat_games[0]          # entity page -> its first (home) game category
    if len(distinct_cats) == 1:
        return next(iter(distinct_cats))     # lore page, one game -> that game
    if len(distinct_cats) >= 2:
        # lore page across several games: Xenosaga-only -> 'XS' umbrella, else genuinely cross-game.
        return "XS" if distinct_cats <= _XS_EPISODES else "series"

    # 4. No categories. A lone game-prefixed template tags the page UNLESS a generic Xenosaga marker
    #    is also present (then it is a Xenosaga page merely referencing another game -> 'series').
    tmpl_codes = {c for name in names if (c := _name_game(name))}
    if len(tmpl_codes) == 1 and not any(_is_xeno_generic(name) for name in names):
        return next(iter(tmpl_codes))
    return "series"


def _suffix_games(title: str):
    """Base games named by explicit title suffixes, or ``None`` if the title carries no game suffix.
    ``(XCn)`` -> {that game}; the Xenosaga-wide ``(XS)`` -> all three episodes; a cross-episode combo
    ``(XS1&2)`` -> the episodes it lists. The author hand-scoped the page, so this is authoritative."""
    found = set()
    for token in _PAREN.findall(title):
        c = _canon_game(token)
        if c:
            found.add(c)
            continue
        up = token.strip().upper().replace(" ", "")
        if up == "XS":
            found |= _XS_EPISODES
        elif _XS_WIDE_TOKEN.fullmatch(up):           # XS1&2 / XS2&3 / XS1&2&3
            found |= {"XS" + d for d in re.findall(r"[1-3]", up)}
    return found or None


def derive_games(title: str, wikitext: str = None) -> frozenset:
    """The SET of base games a page belongs to (multi-tag membership). An **empty** set means
    *ubiquitous* — no game signal, so the page is the cross-franchise ``series`` catch-all that a hard
    filter must never hide.

    Unlike :func:`derive_game` (which must pick ONE display label and so collapses cross-appearance
    pages to ``series``/``XS``), this keeps every game a page genuinely appears in: KOS-MOS ->
    {XS1,XS2,XS3,XC2}, Elma -> {XCX,XC2}, Pyra -> {XC2}. It is the union of every game signal — the
    explicit title suffix (authoritative if present), else the page's structured templates plus all
    its game-bearing categories (cameo categories are real appearances, so they are *included*, not
    discarded). This set drives retrieval filtering via per-game membership flags."""
    suffix = _suffix_games(title)
    if suffix:
        return frozenset(suffix)
    if not wikitext:
        return frozenset()

    names = [n.strip() for n in _TEMPLATE_HEAD.findall(wikitext)]
    member = {g for name in names if _is_structured_name(name) and (g := _name_game(name))}
    for cat in _CATEGORY.findall(wikitext):
        c = None
        for token in _PAREN.findall(cat):
            c = _canon_game(token) or c
        c = c or _phrase_game(cat)
        if c:
            member.add(c)
    if not member:
        # No suffix / structured template / category: a lone game-prefixed template tags the page,
        # unless a generic Xenosaga marker shows it is a Xenosaga page citing another game.
        tmpl_codes = {c for name in names if (c := _name_game(name))}
        if len(tmpl_codes) == 1 and not any(_is_xeno_generic(name) for name in names):
            member |= tmpl_codes
    return frozenset(member)


def membership_from_game(game: str):
    """Fallback membership when an explicit set is absent: derive it from the single display tag —
    ``series``/unknown -> every game (ubiquitous), ``XS`` -> the three Xenosaga episodes, a base game
    -> just itself. Lets pre-membership chunks and the display label still filter sensibly."""
    if game == "XS":
        return set(_XS_EPISODES)
    if game in _BASE_GAMES:
        return {game}
    return set(_BASE_GAMES)


def membership_flags(games) -> dict:
    """Per-game boolean metadata for a chunk: ``{"g_XC2": True, ...}``. An empty/false ``games`` means
    ubiquitous -> a flag for every base game (so a hard per-game filter never hides it). ChromaDB
    metadata can't hold a list, so membership is stored as one boolean flag per game."""
    member = set(games) if games else set(_BASE_GAMES)
    return {f"g_{g}": True for g in member}


def filter_membership(game_filter: str):
    """The base game a per-game retrieval filter restricts to (matched against a chunk's ``g_<game>``
    membership flag), or ``None`` for no restriction — the single source of truth shared by the dense
    (``embed_index._where``) and lexical (``bm25_index.search``) filters. ``series``/``XS`` are display
    labels, not base games, so they impose no restriction."""
    if game_filter in _BASE_GAMES:
        return game_filter
    return None


def title_to_url(title: str) -> str:
    return BASE_WIKI_URL + title.replace(" ", "_")


def _strip_refs(wikitext: str) -> str:
    wikitext = _REF_SELF.sub("", wikitext)
    wikitext = _REF_PAIR.sub("", wikitext)
    return wikitext


def _render_game_links(code):
    """Resolve game-namespaced link shortcuts in place: ``{{XC1|Colony 9}}`` and
    ``{{XC1|Colony 9|the colony}}`` are wiki-link templates, but ``strip_code`` deletes any
    template it can't render — turning "Battle of {{XC1|Colony 9}}" into "Battle of ." and an
    infobox ``location`` into "()". Replace each with its display text (the 2nd positional param
    if given, else the 1st = the target page name) so prose and infobox fields keep their words.
    """
    for tmpl in code.filter_templates():
        if str(tmpl.name).strip() not in KNOWN_GAMES:
            continue
        positional = [p for p in tmpl.params if not p.showkey]
        if len(positional) >= 2:
            display = positional[1].value.strip_code().strip()
        elif positional:
            display = positional[0].value.strip_code().strip()
        else:
            display = ""
        try:
            code.replace(tmpl, display)
        except ValueError:
            pass  # node already removed as part of a parent replacement


def _is_structured_template(name: str) -> bool:
    """Infoboxes *and* stat-block "... data" templates ({{XC1 enemy data}}, {{XCX PC art data}}…),
    which carry the numbers questions ask about (level, HP, power) and are otherwise dropped."""
    low = name.lower()
    return "infobox" in low or low.endswith(" data")


def _extract_infoboxes(code):
    boxes = []
    for tmpl in code.filter_templates():
        name = str(tmpl.name).strip()
        if _is_structured_template(name):
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
    _render_game_links(code)  # before extraction so infobox fields + prose both keep link text

    return {
        "title": title,
        "pageid": pageid,
        "game": derive_game(title, wikitext),                # single display/breadcrumb label
        "games": sorted(derive_games(title, wikitext)),      # multi-tag membership for filtering
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
