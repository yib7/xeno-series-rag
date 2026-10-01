"""Tests for the rendered-HTML parser. Fixtures are real action=parse output captured from the wiki
(tests/fixtures/html/*.json), the source of truth the raw-wikitext parser could not see, because
the wiki's Lua modules decode numeric codes (Atr=7 -> "Light") only when rendering HTML."""

import json
import os

from xeno_rag.parse_html import _render_kv, parse_html_article

FX = os.path.join(os.path.dirname(__file__), "fixtures", "html")


def load(slug):
    with open(os.path.join(FX, f"{slug}.json"), encoding="utf-8") as f:
        rec = json.load(f)
    return parse_html_article(rec["title"], rec.get("pageid"), rec["html"], {},
                              wikitext=rec.get("wikitext"))


def all_fact_lines(art):
    return [ln for fb in art["factblocks"] for ln in fb["lines"]]


def prose_text(art):
    return "\n".join(s["text"] for s in art["sections"])


# ---- the bug that started this: Mythra's element ----

def test_mythra_element_is_extracted():
    art = load("mythra_xc2")
    lines = all_fact_lines(art)
    assert any("Element" in ln and "Light" in ln for ln in lines), \
        "decoded 'Element: Light' must be captured from the rendered stats table"


def test_mythra_decoded_and_named_stats():
    art = load("mythra_xc2")
    lines = " | ".join(all_fact_lines(art))
    assert "Weapon class" in lines and "Aegis Sword" in lines   # decoded weapon
    assert "Role" in lines and "ATK" in lines
    assert "Gender" in lines and "Female" in lines              # decoded from "2"


def test_mythra_prose_is_link_resolved():
    """HTML prose has links already resolved, so the {{gp|Rex}} gap ('one of 's Blades') is gone."""
    art = load("mythra_xc2")
    prose = prose_text(art)
    assert "Rex" in prose and "one of 's" not in prose
    assert "{{" not in prose and "}}" not in prose


def test_mythra_metadata():
    art = load("mythra_xc2")
    assert art["game"] == "XC2"
    assert art["url"].endswith("/wiki/Mythra/Gameplay_(XC2)")


def test_html_cross_appearance_carries_multi_game_membership():
    """An HTML-parsed cross-appearance page must carry the true multi-tag ``games`` set from
    ``derive_games`` (KOS-MOS -> {XS1,XS2,XS3,XC2}), spanning BOTH the Xenosaga and Xenoblade
    subseries, not the collapsed single ``game`` label. Without ``games`` on HTML articles, the
    downstream membership flags fall back to the lossy ``membership_from_game`` (P1-2)."""
    art = load("kosmos_crossgame")
    assert art["games"] == sorted({"XS1", "XS2", "XS3", "XC2"})
    # spans two subseries: a Xenosaga episode AND a Xenoblade game
    assert "XS1" in art["games"] and "XC2" in art["games"]
    # the single display label still collapses to 'series' (Xenosaga lead + Xenoblade cameo),
    # so 'games' is strictly richer than 'game' here.
    assert art["game"] == "series"


# ---- column-header alignment with blank cells (P2-4) ----

def test_render_kv_keeps_colheader_alignment_with_blank_middle_cell():
    """A blank middle stat cell must NOT shift later values under the wrong column header.

    colheaders = ['Base', 'Scaling', 'Level'] aligns positionally with the data row's cells after
    the label. With an empty middle cell (Scaling = N/A), the trailing '5' belongs to 'Level': it
    must be labelled 'level', never 'scaling'. The pre-fix code drops the blank before the zip, so
    '5' slides left onto 'Scaling' and mislabels as '5 scaling'."""
    rows = [
        [("th", ""), ("th", "Base"), ("th", "Scaling"), ("th", "Level")],
        [("th", "HP"), ("td", "100"), ("td", ""), ("td", "5")],
    ]
    line = _render_kv(rows)[0]
    assert "100 base" in line
    assert "5 level" in line
    assert "5 scaling" not in line       # the pre-fix mislabel
    assert "scaling" not in line         # the N/A column produces no phantom value


def test_render_kv_keeps_colheader_alignment_with_blank_leading_cell():
    """A blank LEADING stat cell must not drag the following value onto the first column's header."""
    rows = [
        [("th", ""), ("th", "Base"), ("th", "Scaling"), ("th", "Level")],
        [("th", "HP"), ("td", ""), ("td", "200"), ("td", "5")],
    ]
    line = _render_kv(rows)[0]
    assert "200 scaling" in line          # 200 aligns with 'Scaling', not 'Base'
    assert "5 level" in line
    assert "200 base" not in line         # the pre-fix mislabel
    assert "base" not in line             # the blank Base column produces no phantom value


# ---- enemy stats with Base/Scaling columns ----

def test_shield_igna_enemy_stats():
    art = load("shield_igna_xc1")
    lines = all_fact_lines(art)
    joined = " | ".join(lines)
    assert any("Element" in ln and "Normal" in ln for ln in lines)
    assert "Location" in joined and "Satorl Marsh" in joined
    # multi-column stat: HP base 21,400 must be captured (and labelled, not mashed)
    assert any("HP" in ln and "21,400" in ln for ln in lines)


# ---- item/material facts ----

def test_tough_tendon_item_facts():
    art = load("tough_tendon_xcx")
    joined = " | ".join(all_fact_lines(art))
    assert "Rarity" in joined and "Prime" in joined
    assert "Enemy category" in joined and "Theroid" in joined


def test_grid_table_rows_are_labelled():
    """Grid tables (header row + data rows), e.g. Mythra's Specials, keep column labels per row."""
    art = load("mythra_xc2")
    joined = " | ".join(all_fact_lines(art))
    # a Special art with its damage, labelled by column
    assert "Ray of Punishment" in joined
    assert "360" in joined  # Dmg (Lv1) for Ray of Punishment


def test_dropped_when_empty():
    assert parse_html_article("X", 1, "", {}) is None


def test_html_articles_fall_back_to_wikitext_when_html_is_empty(tmp_path):
    """A record whose HTML parses to nothing falls back to the wikitext parser, so a render hiccup
    never silently loses a page's prose; HTML pages keep their decoded stats."""
    import gzip

    from xeno_rag.parse_html import _html_articles_by_pageid
    with open(os.path.join(FX, "mythra_xc2.json"), encoding="utf-8") as f:
        mythra_html = json.load(f)["html"]
    records = [
        {"title": "Mythra/Gameplay (XC2)", "pageid": 1,
         "html": mythra_html,
         "wikitext": ""},
        {"title": "Fallback (XC1)", "pageid": 2, "html": "",
         "wikitext": "{{Infobox XC1 enemy|name=Test}}\nSome prose about the test enemy here for bytes."},
    ]
    hdir = tmp_path / "html"
    hdir.mkdir()
    with gzip.open(hdir / "html_00000.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    arts = _html_articles_by_pageid({"paths": {"html": str(hdir)}, "min_wikitext_bytes": 50})
    assert set(arts) == {1, 2}                          # the HTML-empty page used wikitext
    facts = [ln for fb in arts[1]["factblocks"] for ln in fb["lines"]]
    assert any("Element" in f and "Light" in f for f in facts)


def test_run_hybrid_merges_html_stats_with_wikitext_prose(tmp_path):
    """Hybrid corpus: stat pages we fetched as HTML get decoded facts; other pages keep wikitext prose."""
    import gzip

    from xeno_rag.parse_html import run_hybrid
    hdir = tmp_path / "html"
    hdir.mkdir()
    with open(os.path.join(FX, "mythra_xc2.json"), encoding="utf-8") as f:
        mythra = json.load(f)["html"]
    with gzip.open(hdir / "html_00000.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"title": "Mythra/Gameplay (XC2)", "pageid": 1, "html": mythra, "wikitext": ""}) + "\n")
    pdir = tmp_path / "pages"
    pdir.mkdir()
    raw = [
        {"title": "Mythra/Gameplay (XC2)", "pageid": 1,
         "revisions": [{"slots": {"main": {"content": "{{stub}} Mythra raw wikitext, overridden by HTML, with enough bytes."}}}]},
        {"title": "Sharla (XC1)", "pageid": 2,
         "revisions": [{"slots": {"main": {"content": "'''Sharla''' is a medic in Xenoblade Chronicles, with a healing rifle and prose here."}}}]},
    ]
    (pdir / "pages_00000.jsonl").write_text("\n".join(json.dumps(r) for r in raw), encoding="utf-8")
    cfg = {"paths": {"html": str(hdir), "pages": str(pdir), "articles": str(tmp_path / "articles.jsonl")}}
    stats = run_hybrid(cfg)
    assert stats["from_html"] == 1 and stats["from_wikitext"] == 1
    with open(cfg["paths"]["articles"], encoding="utf-8") as f:
        arts = {a["title"]: a for a in (json.loads(ln) for ln in f)}
    mfacts = [ln for fb in arts["Mythra/Gameplay (XC2)"].get("factblocks", []) for ln in fb["lines"]]
    assert any("Element" in f and "Light" in f for f in mfacts)        # HTML won for the stat page
    assert "Sharla" in arts["Sharla (XC1)"]["sections"][0]["text"]      # wikitext prose for the rest


def test_run_hybrid_matches_by_pageid_on_title_mismatch(tmp_path):
    """P1-1: the HTML action=parse title can be resolved (redirect / whitespace-underscore
    normalization) to something other than the raw-pull title, while both share the same pageid.
    The merge must key on pageid, not title, or the page gets wikitext-parsed (losing the
    Lua-decoded stats the HTML was fetched for) AND the leftover HTML article is ALSO written,
    duplicating the pageid in the corpus."""
    import gzip

    from xeno_rag.parse_html import run_hybrid
    hdir = tmp_path / "html"
    hdir.mkdir()
    html = "<table class='infobox'><tr><th>Element</th><td>Light</td></tr></table>"
    with gzip.open(hdir / "html_00000.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"title": "Foo (normalized)", "pageid": 7, "html": html, "wikitext": ""}) + "\n")
    pdir = tmp_path / "pages"
    pdir.mkdir()
    raw = [
        {"title": "Foo", "pageid": 7,
         "revisions": [{"slots": {"main": {"content": "'''Foo''' is a character with a long enough raw wikitext body for the parser to accept."}}}]},
    ]
    (pdir / "pages_00000.jsonl").write_text("\n".join(json.dumps(r) for r in raw), encoding="utf-8")
    cfg = {"paths": {"html": str(hdir), "pages": str(pdir), "articles": str(tmp_path / "articles.jsonl")}}
    result = run_hybrid(cfg)
    with open(cfg["paths"]["articles"], encoding="utf-8") as f:
        arts = [json.loads(ln) for ln in f]
    foo_arts = [a for a in arts if a["pageid"] == 7]
    assert len(foo_arts) == 1, \
        "raw-pull and HTML-resolved titles differ but share a pageid: must merge to ONE record"
    facts = [ln for fb in foo_arts[0].get("factblocks", []) for ln in fb["lines"]]
    assert any("Element" in f and "Light" in f for f in facts), \
        "the merged record must be the HTML-parsed one (decoded stats), not the wikitext fallback"
    assert result["from_html"] == 1
    assert result["from_wikitext"] == 0


def test_inline_tags_do_not_leave_gaps_before_punctuation_or_possessives():
    html = ("<div class='mw-parser-output'><p>The <a href='/m'>Monado</a>, a sword. "
            "See <a href='/s'>Shulk</a>'s blade (<b>rare</b>).</p></div>")
    art = parse_html_article("Monado", 1, html)
    text = art["sections"][0]["text"]
    assert "Monado, a sword" in text and "Shulk's blade" in text and "(rare)." in text


def test_table_header_row_inside_thead_is_not_lost():
    html = ("<div class='mw-parser-output'><h2>Drops</h2><table>"
            "<thead><tr><th>Item</th><th>Rate</th></tr></thead>"
            "<tbody><tr><td>Potion</td><td>10%</td></tr></tbody></table></div>")
    art = parse_html_article("Foo", 2, html)
    assert art["factblocks"][0]["lines"] == ["Potion - Rate: 10%."]


def test_run_hybrid_with_no_inputs_refuses_and_keeps_the_existing_corpus(tmp_path):
    """Empty/missing page dirs used to overwrite a good articles.jsonl with an empty one, silently."""
    import pytest

    from xeno_rag.errors import SetupError
    from xeno_rag.parse_html import run_hybrid

    (tmp_path / "html").mkdir()
    (tmp_path / "pages").mkdir()
    out = tmp_path / "articles.jsonl"
    out.write_text("good corpus" + chr(10), encoding="utf-8")
    cfg = {"paths": {"html": str(tmp_path / "html"), "pages": str(tmp_path / "pages"), "articles": str(out)}}
    with pytest.raises(SetupError, match="No pages to parse"):
        run_hybrid(cfg)
    assert out.read_text(encoding="utf-8") == "good corpus" + chr(10)


def test_run_hybrid_logs_a_duplicate_pageid_instead_of_printing(tmp_path, caplog, capsys):
    import gzip
    import logging

    from xeno_rag.parse_html import run_hybrid

    pages = tmp_path / "pages"
    hdir = tmp_path / "html"
    pages.mkdir()
    hdir.mkdir()
    wt = "{{Infobox XC1 enemy|name=Test}}" + chr(10) + "Some prose about the test enemy here for bytes."
    raw = [{"title": t, "pageid": 7, "revisions": [{"slots": {"main": {"content": wt}}}]}
           for t in ("Twin A", "Twin B")]
    (pages / "pages_00000.jsonl").write_text(
        chr(10).join(json.dumps(r) for r in raw), encoding="utf-8")
    html = "<div class='mw-parser-output'><p>Twin A is a long enough paragraph of prose.</p></div>"
    with gzip.open(hdir / "html_00000.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"title": "Twin A", "pageid": 7, "html": html, "wikitext": wt}) + chr(10))
    cfg = {"min_wikitext_bytes": 10,
           "paths": {"articles": str(tmp_path / "a.jsonl"), "pages": str(pages), "html": str(hdir)}}
    with caplog.at_level(logging.WARNING, logger="xeno_rag.parse_html"):
        stats = run_hybrid(cfg)
    assert stats["from_html"] == 1 and stats["from_wikitext"] == 1
    assert any("duplicate pageid 7" in r.getMessage() for r in caplog.records)
    assert capsys.readouterr().err == ""
