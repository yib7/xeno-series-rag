"""Tests for the rendered-HTML parser. Fixtures are real action=parse output captured from the wiki
(tests/fixtures/html/*.json) — the source of truth the raw-wikitext parser could not see, because
the wiki's Lua modules decode numeric codes (Atr=7 -> "Light") only when rendering HTML."""

import json
import os


from xeno_rag.parse_html import parse_html_article

FX = os.path.join(os.path.dirname(__file__), "fixtures", "html")


def load(slug):
    rec = json.load(open(os.path.join(FX, f"{slug}.json"), encoding="utf-8"))
    return parse_html_article(rec["title"], rec.get("pageid"), rec["html"], {})


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


def test_run_writes_articles_with_wikitext_fallback(tmp_path):
    """parse_html.run parses HTML records; when HTML is empty it falls back to the wikitext parser."""
    from xeno_rag.parse_html import run as run_parse
    records = [
        {"title": "Mythra/Gameplay (XC2)", "pageid": 1,
         "html": json.load(open(os.path.join(FX, "mythra_xc2.json"), encoding="utf-8"))["html"],
         "wikitext": ""},
        {"title": "Fallback (XC1)", "pageid": 2, "html": "",
         "wikitext": "{{Infobox XC1 enemy|name=Test}}\nSome prose about the test enemy here for bytes."},
    ]
    cfg = {"paths": {"articles": str(tmp_path / "articles.jsonl")}}
    stats = run_parse(cfg, html_records=records)
    assert stats["written"] == 2
    assert stats["fallback"] == 1                       # the HTML-empty page used wikitext
    arts = [json.loads(ln) for ln in open(cfg["paths"]["articles"], encoding="utf-8")]
    mythra = next(a for a in arts if a["title"].startswith("Mythra"))
    facts = [ln for fb in mythra["factblocks"] for ln in fb["lines"]]
    assert any("Element" in f and "Light" in f for f in facts)


def test_run_hybrid_merges_html_stats_with_wikitext_prose(tmp_path):
    """Hybrid corpus: stat pages we fetched as HTML get decoded facts; other pages keep wikitext prose."""
    import gzip
    from xeno_rag.parse_html import run_hybrid
    hdir = tmp_path / "html"
    hdir.mkdir()
    mythra = json.load(open(os.path.join(FX, "mythra_xc2.json"), encoding="utf-8"))["html"]
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
    arts = {a["title"]: a for a in (json.loads(ln) for ln in open(cfg["paths"]["articles"], encoding="utf-8"))}
    mfacts = [ln for fb in arts["Mythra/Gameplay (XC2)"].get("factblocks", []) for ln in fb["lines"]]
    assert any("Element" in f and "Light" in f for f in mfacts)        # HTML won for the stat page
    assert "Sharla" in arts["Sharla (XC1)"]["sections"][0]["text"]      # wikitext prose for the rest
