"""Tests for wikitext parsing into prose + structured infobox data."""

from pathlib import Path

from xeno_rag.parse_wikitext import derive_game, title_to_url, parse_article, run

CFG = {"min_wikitext_bytes": 50, "paths": {}}
FIX = Path(__file__).parent / "fixtures"


def load(name):
    return (FIX / name).read_text(encoding="utf-8")


# --- derive_game ---

def test_derive_game_from_parenthetical_code():
    assert derive_game("Infinity Blade (XC3) (Noah)") == "XC3"
    assert derive_game("Fei Fong Wong (XG)") == "XG"
    assert derive_game("Elma (XCX)") == "XCX"


def test_derive_game_series_when_no_code():
    assert derive_game("Zohar") == "series"
    assert derive_game("Xenoblade Chronicles 3") == "series"


# --- title_to_url ---

def test_title_to_url_spaces_to_underscores():
    assert (
        title_to_url("Infinity Blade (XC3) (Noah)")
        == "https://www.xenoserieswiki.org/wiki/Infinity_Blade_(XC3)_(Noah)"
    )


# --- parse_article: infobox + prose ---

def test_parse_extracts_infobox_fields():
    art = parse_article("Infinity Blade (XC3) (Noah)", 70047, load("art_xc3.wikitext"), CFG)
    assert art["game"] == "XC3"
    assert art["url"].endswith("Infinity_Blade_(XC3)_(Noah)")
    assert len(art["infoboxes"]) == 1
    ib = art["infoboxes"][0]
    assert ib["template"] == "Infobox XC3 art"
    assert ib["fields"]["power"] == "250"
    assert ib["fields"]["type"] == "Talent Art"


def test_parse_resolves_links_and_strips_refs():
    art = parse_article("Infinity Blade (XC3) (Noah)", 70047, load("art_xc3.wikitext"), CFG)
    headings = {s["heading"]: s["text"] for s in art["sections"]}
    # lead section captured, link [[Noah (XC3)|Noah]] resolved to display text
    assert "Noah" in headings["Introduction"]
    assert "power" not in headings["Introduction"]  # infobox not dumped into prose
    # ref content + markup gone, link [[Attack]] resolved
    assert "Mechanics" in headings
    assert "Attack" in headings["Mechanics"]
    assert "ref" not in headings["Mechanics"]
    assert "in-game data" not in headings["Mechanics"]


def test_parse_character_pipe_link_display():
    art = parse_article("Fei Fong Wong (XG)", 1, load("character_xg.wikitext"), CFG)
    assert art["game"] == "XG"
    assert art["infoboxes"][0]["template"] == "Infobox Character"
    story = {s["heading"]: s["text"] for s in art["sections"]}["Story"]
    assert "his village" in story  # [[Lahan|his village]] -> his village


def test_parse_lore_has_no_infobox_and_is_series():
    art = parse_article("Zohar", 2, load("lore_series.wikitext"), CFG)
    assert art["game"] == "series"
    assert art["infoboxes"] == []
    assert any(s["heading"] == "Appearances" for s in art["sections"])


# --- parse_article: drops ---

def test_parse_drops_redirect():
    assert parse_article("Foo", 3, "#REDIRECT [[Bar]]", CFG) is None


def test_parse_drops_short_stub():
    assert parse_article("Foo", 4, "tiny", CFG) is None


def test_parse_drops_disambiguation():
    wt = "{{disambiguation}}\nThis could refer to several things across the series entries here."
    assert parse_article("Vandham", 5, wt, CFG) is None


# --- run ---

def test_run_writes_articles_and_counts_drops(tmp_path):
    out = tmp_path / "articles.jsonl"
    cfg = {"min_wikitext_bytes": 50, "paths": {"articles": str(out)}}
    raw_pages = [
        {"title": "Infinity Blade (XC3) (Noah)", "pageid": 70047,
         "revisions": [{"slots": {"main": {"content": load("art_xc3.wikitext")}}}]},
        {"title": "Foo", "pageid": 3,
         "revisions": [{"slots": {"main": {"content": "#REDIRECT [[Bar]]"}}}]},
    ]
    stats = run(cfg, raw_pages=raw_pages)
    assert stats["written"] == 1
    assert stats["dropped"] == 1
    assert out.read_text(encoding="utf-8").strip().count("\n") == 0  # one line
