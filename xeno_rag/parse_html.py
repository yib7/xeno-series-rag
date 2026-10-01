"""Parse rendered MediaWiki HTML (action=parse output) into prose sections + structured facts.

Why HTML and not wikitext: the wiki's Lua modules decode internal numeric codes only when rendering
(``Atr=7`` -> "Light", ``DefWeapon=5005`` -> "Aegis Sword", ``Gender=2`` -> "Female"), so the
human-readable stat tables exist *only* in the HTML. Parsing the rendered HTML also gives prose with
links already resolved, so the wikitext template-stripping gaps ("one of 's Blades") disappear.

Two table shapes are handled:
  - key/value tables (``<th>label</th><td>value</td>`` rows) -> "Label: value." facts, with optional
    column headers (enemy Base/Scaling stats) labelling multi-value rows.
  - grid tables (a header row of ``<th>`` + data rows of ``<td>``) -> one labelled record per row
    (e.g. each of a Blade's Special arts with its damage columns).
"""

import re

from bs4 import BeautifulSoup

from .errors import SetupError
from .fileio import atomic_text_writer
from .parse_wikitext import derive_game, derive_games, title_to_url

_WS = re.compile(r"\s+")
_GRID_ROW_CAP = 40   # don't let a huge drop/skill table explode into one giant chunk
_MAX_LINE = 320      # truncate a runaway concatenated row (e.g. affinity-chart reward dumps)
# Tables that are pure game-internal noise for a natural-language chatbot (coordinate/weather dumps).
_NOISE_HEADINGS = {"spawnpoints", "spawn points", "spawn point"}


def _clip(line: str) -> str:
    return line if len(line) <= _MAX_LINE else line[:_MAX_LINE].rstrip() + "…"

_DROP_SELECTORS = (
    ".navbox, .toc, #toc, .mw-editsection, .reference, sup.reference, "
    ".mw-empty-elt, style, script, .noprint, .thumb, .gallery, figure, .mw-jump-link"
)


# get_text(" ") puts a space on both sides of every inline tag, so a link or <b> in a sentence leaves
# "the Monado , a sword" and "Shulk 's blade". Close those gaps (punctuation and possessives only).
_SPACE_BEFORE_PUNCT = re.compile(r"\s+(?=[,.;:!?)\]]|'s\b)")
_SPACE_AFTER_OPEN = re.compile(r"(?<=[(\[])\s+")


def _norm(s: str) -> str:
    s = _WS.sub(" ", (s or "").replace("\u200b", "")).strip()
    return _SPACE_AFTER_OPEN.sub("", _SPACE_BEFORE_PUNCT.sub("", s))


def _row_cells(tr):
    return [(c.name, _norm(c.get_text(" "))) for c in tr.find_all(["th", "td"], recursive=False)]


def _table_rows(table):
    # Rows can sit directly in the table or inside thead/tbody/tfoot (a grid's header row is usually
    # in a <thead>). Only this table's own rows: recursive=False keeps nested tables out.
    trs = []
    for child in table.find_all(["tr", "thead", "tbody", "tfoot"], recursive=False):
        trs.extend([child] if child.name == "tr" else child.find_all("tr", recursive=False))
    rows = [_row_cells(tr) for tr in trs]
    return [r for r in rows if r]


def _render_kv(rows):
    """Key/value table -> ['Label: value.', ...]. Supports a leading column-header row whose first
    cell is empty (enemy 'Base'/'Scaling' columns) and continuation rows (all-<td>) appended to the
    previous label."""
    colheaders = None
    out = []
    for r in rows:
        tags = [t for t, _ in r]
        vals = [v for _, v in r]
        if all(t == "th" for t in tags) and len(r) >= 2 and vals[0] == "":
            colheaders = vals[1:]
            continue
        if all(t == "th" for t in tags):
            continue  # a sub-header row with no values
        if tags[0] == "th":
            label = vals[0]
            if not label:
                continue
            if colheaders:
                # Pair each cell with its column header by POSITION, skipping empty cells without
                # collapsing the index (a blank Base/Scaling column must not shift later values onto
                # the wrong header).
                parts = []
                for j, (_, v) in enumerate(r[1:]):
                    if not v:
                        continue
                    ch = colheaders[j] if j < len(colheaders) else None
                    parts.append(f"{v} {ch.lower()}" if ch and ch.lower() not in v.lower() else v)
                if parts:
                    out.append(f"{label}: {', '.join(parts)}.")
            else:
                values = [v for _, v in r[1:] if v]
                if values:
                    out.append(f"{label}: {', '.join(values)}.")
        else:  # continuation (all <td>) -> append to previous label
            extra = [v for v in vals if v]
            if extra and out:
                out[-1] = out[-1].rstrip(".") + ", " + ", ".join(extra) + "."
    return out


def _render_grid(header, data_rows):
    out = []
    for r in data_rows[:_GRID_ROW_CAP]:
        vals = [v for _, v in r]
        if not any(vals):
            continue
        name = vals[0]
        pairs = []
        for j in range(1, len(vals)):
            if not vals[j]:
                continue
            h = header[j] if j < len(header) else None
            pairs.append(f"{h}: {vals[j]}" if h else vals[j])
        line = name + (" - " + "; ".join(pairs) if pairs else "")
        if line.strip():
            out.append(_clip(line.rstrip(".")) + ".")
    return out


def _table_lines(table):
    rows = _table_rows(table)
    if not rows:
        return []
    # leading all-<th> rows
    lead = []
    i = 0
    while i < len(rows) and all(t == "th" for t, _ in rows[i]) and len(rows[i]) >= 2:
        lead.append([v for _, v in rows[i]])
        i += 1
    # grid = a real header row (first header cell non-empty) followed by <td> data rows
    if lead and lead[-1][0] != "" and i < len(rows) and any(
        any(t == "td" for t, _ in r) for r in rows[i:]
    ):
        return _render_grid(lead[-1], rows[i:])
    return _render_kv(rows)


def _heading_text(tag):
    headline = tag.find(class_="mw-headline")
    return _norm(headline.get_text() if headline else tag.get_text())


def parse_html_article(title, pageid, html, cfg=None, wikitext=None):
    """Parse one rendered page. Returns a record dict, or None if it has no usable content.

    ``wikitext`` (fetched alongside the HTML) is used only for game tagging: the decoded stats come
    from the HTML, but the game a page belongs to is read from its categories / template prefixes,
    which live in the wikitext (see ``derive_game``)."""
    if not html or not html.strip():
        return None
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one(".mw-parser-output") or soup
    for bad in root.select(_DROP_SELECTORS):
        bad.decompose()

    sections = []  # {heading, text}
    factblocks = []  # {heading, lines}
    cur_heading = "Introduction"
    sec_text = {}  # heading -> list of paragraphs (preserve first-seen order via sections list)
    order = []

    def add_prose(text):
        text = _norm(text)
        if not text:
            return
        if cur_heading not in sec_text:
            sec_text[cur_heading] = []
            order.append(cur_heading)
        sec_text[cur_heading].append(text)

    def walk(node):
        nonlocal cur_heading
        for child in node.children:
            name = getattr(child, "name", None)
            if name is None:
                continue
            if name in ("h2", "h3", "h4"):
                cur_heading = _heading_text(child) or cur_heading
            elif name == "table":
                cls = " ".join(child.get("class", []))
                if "navbox" in cls:
                    continue
                if _norm(cur_heading).lower() in _NOISE_HEADINGS:
                    continue
                lines = _table_lines(child)
                if lines:
                    heading = "infobox" if "infobox" in cls else cur_heading
                    factblocks.append({"heading": heading, "lines": lines})
            elif name in ("p", "ul", "ol", "dl", "blockquote"):
                add_prose(child.get_text(" "))
            else:
                walk(child)

    walk(root)

    for h in order:
        text = _norm("\n".join(sec_text[h]))
        if text:
            sections.append({"heading": h, "text": text})

    if not sections and not factblocks:
        return None
    return {
        "title": title,
        "pageid": pageid,
        "game": derive_game(title, wikitext),                # single display/breadcrumb label
        "games": sorted(derive_games(title, wikitext)),      # multi-tag membership for filtering
        "url": title_to_url(title),
        "sections": sections,
        "factblocks": factblocks,
    }


def _html_articles_by_pageid(cfg):
    """Parse every fetched HTML record into an article, keyed by pageid (wikitext fallback if a
    page's HTML was empty / only an error was recorded).

    Keyed by pageid rather than title: ``action=parse`` resolves redirects and normalizes
    whitespace/underscores, so the HTML record's title can differ from the raw-pull title even
    though both describe the same page (same pageid). Falls back to the title only when a record
    has no pageid (defensive; real pages always have one, and error-only records already parse to
    None above and are skipped)."""
    from .fetch_html import iter_html_records
    from .parse_wikitext import parse_article as _pw
    out = {}
    for rec in iter_html_records(cfg["paths"]["html"]):
        title = rec.get("title")
        pageid = rec.get("pageid")
        art = parse_html_article(title, pageid, rec.get("html"), cfg,
                                 wikitext=rec.get("wikitext"))
        if art is None and rec.get("wikitext"):
            art = _pw(title, pageid, rec.get("wikitext"), cfg)
        if art is not None:
            key = pageid if pageid is not None else title
            out[key] = art
    return out


def run(cfg: dict, html_records=None) -> dict:
    """Parse all fetched HTML records → articles.jsonl. Returns {written, dropped, fallback}.

    Falls back to the wikitext parser when a page's HTML yields nothing (or only an error was
    recorded at fetch time), so a render hiccup never silently loses a page's prose.
    """
    import json as _json
    import os as _os

    from .fetch_html import iter_html_records
    from .parse_wikitext import parse_article as _parse_wikitext

    if html_records is None:
        html_records = iter_html_records(cfg["paths"]["html"])
    out_path = cfg["paths"]["articles"]
    _os.makedirs(_os.path.dirname(_os.path.abspath(out_path)), exist_ok=True)
    written = dropped = fallback = 0
    with atomic_text_writer(out_path) as out:
        for rec in html_records:
            title, pageid = rec.get("title"), rec.get("pageid")
            art = parse_html_article(title, pageid, rec.get("html"), cfg,
                                     wikitext=rec.get("wikitext"))
            if art is None and rec.get("wikitext"):
                art = _parse_wikitext(title, pageid, rec.get("wikitext"), cfg)
                if art is not None:
                    fallback += 1
            if art is None:
                dropped += 1
                continue
            out.write(_json.dumps(art, ensure_ascii=False) + "\n")
            written += 1
    return {"written": written, "dropped": dropped, "fallback": fallback}


def run_hybrid(cfg: dict) -> dict:
    """Build the merged corpus: HTML-parsed articles for the stat pages we fetched, wikitext-parsed
    articles for everything else. One article per page; HTML wins where we have it. Returns counts.

    Matched by pageid, not title: ``action=parse`` can resolve a redirect or normalize whitespace/
    underscores, so an HTML record's title sometimes differs from the raw-pull title of the same
    page. Matching by title would miss the HTML article (the page gets wikitext-parsed, losing the
    Lua-decoded stats) and then write the orphaned HTML article a second time from the leftover
    pass below, duplicating the pageid downstream."""
    import json as _json
    import os as _os
    import sys as _sys

    from .parse_wikitext import _iter_raw_pages, _page_wikitext
    from .parse_wikitext import parse_article as _pw

    html_arts = _html_articles_by_pageid(cfg)
    out_path = cfg["paths"]["articles"]
    _os.makedirs(_os.path.dirname(_os.path.abspath(out_path)), exist_ok=True)
    from_html = from_wikitext = dropped = 0
    consumed_keys = set()
    with atomic_text_writer(out_path) as out:
        for page in _iter_raw_pages(cfg["paths"]["pages"]):
            title = page.get("title")
            pageid = page.get("pageid")
            key = pageid if pageid is not None else title
            art = html_arts.pop(key, None)
            if art is not None:
                consumed_keys.add(key)
                from_html += 1
            else:
                if key in consumed_keys:
                    # Two raw pages share a pageid that already matched an earlier HTML article -
                    # a duplicate-pageid drift signal, not expected on a healthy pull.
                    print(
                        f"run_hybrid: duplicate pageid {pageid!r} (title={title!r}) already "
                        "consumed by an earlier raw page; falling back to wikitext for this one",
                        file=_sys.stderr, flush=True,
                    )
                art = _pw(title, pageid, _page_wikitext(page), cfg)
                if art is not None:
                    from_wikitext += 1
            if art is None:
                dropped += 1
                continue
            out.write(_json.dumps(art, ensure_ascii=False) + "\n")
        # HTML pages with no raw-wikitext counterpart (rare) still get written
        for art in html_arts.values():
            out.write(_json.dumps(art, ensure_ascii=False) + "\n")
            from_html += 1
        if from_html + from_wikitext == 0:
            # Raising inside the writer discards the temp file, so the previous corpus survives.
            raise SetupError(
                f"No pages to parse: {cfg['paths']['pages']} has no raw wikitext pages and "
                f"{cfg['paths']['html']} has no HTML records. Run the fetch steps first; the "
                f"existing {out_path} was left untouched."
            )
    return {"from_html": from_html, "from_wikitext": from_wikitext, "dropped": dropped}
