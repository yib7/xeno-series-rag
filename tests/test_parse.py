"""Tests for wikitext parsing into prose + structured infobox data."""

from pathlib import Path

from xeno_rag.parse_wikitext import (
    derive_game,
    derive_games,
    filter_membership,
    parse_article,
    title_to_url,
)

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


def test_derive_game_no_wikitext_is_title_only():
    # Backward compatible: without page content, only the title suffix is consulted.
    assert derive_game("Noah", None) == "series"
    assert derive_game("Noah (XC3)", None) == "XC3"


def test_derive_game_from_structured_template_prefix():
    # An XC3 page with no title suffix is still tagged XC3 from its own infobox/data template,
    # so it leaves the catch-all "series" bucket and stops leaking into other games' filters.
    wt = "{{XC3 character infobox|name=Noah}}\nNoah is a protagonist.\n[[Category:XC3 characters]]"
    assert derive_game("Noah", wt) == "XC3"


def test_derive_game_structured_template_code_can_be_mid_name():
    # Some infoboxes carry the code in the middle ("Infobox XC2 blade"), not as the first token.
    wt = "{{Infobox XC2 blade|name=Mythra}}\nMythra is an Aegis."
    assert derive_game("Mythra", wt) == "XC2"


def test_derive_game_from_category_when_template_not_prefixed():
    wt = "{{Infobox character}}\nProse.\n[[Category:Xenoblade Chronicles 2 Blades]]"
    assert derive_game("Pyra", wt) == "XC2"


def test_derive_game_own_infobox_beats_inline_cross_links():
    # The page's own structured template (XC3) is decisive even when prose links other games.
    wt = ("{{XC3 enemy data|lv=50}}\nLike the boss in {{XC1|Colony 9}} and {{XC2|Gormott}}.\n"
          "[[Category:Xenoblade Chronicles 3 Enemies]]")
    assert derive_game("Some Boss", wt) == "XC3"


def test_derive_game_multiple_games_stay_series():
    # A cross-game lore page (multiple games, no single owning template) stays series.
    wt = ("Zohar recurs across the series.\n[[Category:Xenogears]]\n"
          "[[Category:Xenosaga Episode I]]\n[[Category:Xenoblade Chronicles 2]]")
    assert derive_game("Zohar", wt) == "series"


def test_derive_game_variant_codes_fold_to_base():
    # Torna / Future Redeemed / Definitive Edition templates map to their base game.
    assert derive_game("Lila", "{{XC2T blade infobox|name=Lila}}") == "XC2"
    assert derive_game("Matthew", "{{XC3FR character infobox|name=Matthew}}") == "XC3"
    assert derive_game("Dunban (XCDE)", None) == "XC1"


def test_derive_game_title_suffix_overrides_content():
    assert derive_game("Vandham (XC2)",
                       "{{XC1 enemy data}}\n[[Category:Xenoblade Chronicles 1]]") == "XC2"


def test_derive_game_pan_xenosaga_lead_with_foreign_cameo_is_series():
    # A recurring Xenosaga lead in 2+ episodes (KOS-MOS: XS1/XS2/XS3) that ALSO has a non-Xenosaga
    # cameo category (XC2 Blade) spans franchises, so it stays the all-franchises 'series'
    # (visible under the XC2 filter too). Only a *purely* Xenosaga page becomes the 'XS' umbrella.
    wt = ("{{Infobox character}}\n{{XC2|Pyra}} cameos here.\nKOS-MOS is an android.\n"
          "[[Category:Characters (XS1)]]\n[[Category:Characters (XS2)]]\n"
          "[[Category:Characters (XS3)]]\n[[Category:Characters (XC2)]]")
    assert derive_game("KOS-MOS", wt) == "series"


def test_derive_game_explicit_xs_wide_suffix_is_xs():
    # An explicit Xenosaga-wide title suffix the wiki author wrote, '(XS)' (Xenosaga-generic) or a
    # cross-episode '(XS1&2)', is a Xenosaga-only page. It must carry the 'XS' umbrella, NOT the
    # all-franchises 'series' (which leaked these into Xenogears/Xenoblade filters, e.g. 'Ether (XS)'
    # surfacing under a Xenogears query). Episode-specific '(XS1)' still resolves to that base game.
    assert derive_game("Zohar (XS)") == "XS"
    assert derive_game("Ether (XS)") == "XS"
    assert derive_game("Alex (XS1&2)") == "XS"
    assert derive_game("Ether Amp (XS1&2)") == "XS"
    assert derive_game("Margulis (XS1)") == "XS1"   # single episode -> base game, unchanged


def test_derive_game_pan_xenosaga_only_is_xs():
    # A recurring lead whose categories are ALL Xenosaga episodes (no foreign cameo) is Xenosaga-wide,
    # so it gets 'XS': visible under every Xenosaga filter, hidden under XG/XC1/XC2/XC3/XCX.
    wt = ("{{Infobox character}}\nchaos is an android.\n"
          "[[Category:Characters (XS1)]]\n[[Category:Characters (XS2)]]\n"
          "[[Category:Characters (XS3)]]")
    assert derive_game("chaos", wt) == "XS"


def test_derive_game_pan_xenosaga_lore_only_is_xs():
    # A no-infobox lore page spanning multiple Xenosaga episodes only (no other franchise) -> 'XS'.
    wt = ("U-DO recurs across Xenosaga.\n[[Category:Xenosaga Episode I]]\n"
          "[[Category:Xenosaga Episode III]]")
    assert derive_game("U-DO", wt) == "XS"


# --- derive_games: the SET of base games a page belongs to (multi-tag membership) ---

def test_derive_games_cross_appearance_keeps_every_game():
    # KOS-MOS appears in all three Xenosaga episodes AND as an XC2 Blade. A single game tag would collapse this
    # to 'series' (shown everywhere). Membership keeps the EXACT set: XS1/XS2/XS3/XC2, and nothing else,
    # so she does not show under XG/XC1/XC3.
    wt = ("{{Infobox character}}\nKOS-MOS is an android.\n"
          "[[Category:Characters (XS1)]]\n[[Category:Characters (XS2)]]\n"
          "[[Category:Characters (XS3)]]\n[[Category:Characters (XC2)]]")
    assert derive_games("KOS-MOS", wt) == frozenset({"XS1", "XS2", "XS3", "XC2"})


def test_derive_games_home_plus_cameo():
    # Elma debuts in XCX and is also an XC2 Blade -> exactly {XCX, XC2} (not all of 'series', not XCX-only).
    wt = ("{{XCX character infobox}}\n[[Category:Characters (XCX)]]\n[[Category:Characters (XC2)]]")
    assert derive_games("Elma", wt) == frozenset({"XCX", "XC2"})


def test_derive_games_single_game():
    wt = "{{XC2 blade infobox}}\n[[Category:Blades (XC2)]]\nPyra is the Aegis."
    assert derive_games("Pyra", wt) == frozenset({"XC2"})


def test_derive_games_explicit_suffix_scopes_membership():
    # An explicit base-code suffix scopes the page to exactly that game; a Xenosaga-wide suffix with no
    # categories expands to the episodes it names ('(XS)' -> all three, '(XS1&2)' -> XS1+XS2).
    assert derive_games("Infinity Blade (XC3) (Noah)") == frozenset({"XC3"})
    assert derive_games("Ether (XS)") == frozenset({"XS1", "XS2", "XS3"})
    assert derive_games("Ether Amp (XS1&2)") == frozenset({"XS1", "XS2"})


def test_derive_games_ubiquitous_when_no_signal():
    # No game suffix / template / category -> empty set = ubiquitous (the 'series' catch-all: a hard
    # filter must never hide it).
    assert derive_games("Some Meta Page", "Just prose, no game categories.") == frozenset()
    assert derive_games("Noah", None) == frozenset()


# --- filter_membership: a base-game filter narrows to pages whose membership includes that game ---

def test_filter_membership_base_game_and_none():
    assert filter_membership("XC2") == "XC2"
    assert filter_membership("XS1") == "XS1"
    assert filter_membership(None) is None
    assert filter_membership("") is None
    # 'series' / 'XS' are display labels, not base games -> no membership restriction.
    assert filter_membership("series") is None


def test_derive_game_single_episode_character_with_cameo_keeps_home():
    # T-elos appears in ONE Xenosaga episode (XS3) plus an XC2 Blade cameo. One XS episode, so she
    # is not series-wide, her home category (XS3, listed first) wins; the cameo must not steal it.
    wt = ("{{Infobox character}}\nT-elos is a weapon.\n"
          "[[Category:Characters (XS3)]]\n[[Category:Characters (XC2)]]")
    assert derive_game("T-elos", wt) == "XS3"


def test_derive_game_recurring_character_keeps_home_game():
    # Elma debuts in XCX and cameos in XC2. First category is her home game; she must stay XCX and
    # NOT flip to 'series' (which would leak her into XC1/XC3 filters).
    wt = ("{{Infobox character}}\n[[Category:Characters (XCX)]]\n[[Category:Characters (XC2)]]")
    assert derive_game("Elma", wt) == "XCX"


def test_derive_game_xenoblade_character_keeps_home_not_series():
    # Xenoblade games have DISTINCT casts (unlike the Xenosaga trilogy), so a character appearing
    # across several Xenoblade games is an XC1 lead with later cameos: she keeps her home game
    # (first category, XC1) and must NOT collapse to 'series' (which is the leakage we removed).
    wt = ("{{Infobox character}}\n[[Category:Characters (XC1)]]\n"
          "[[Category:Characters (XC2)]]\n[[Category:Characters (XC3)]]")
    assert derive_game("Shulk", wt) == "XC1"


def test_derive_game_xenosaga_marker_blocks_foreign_crossref():
    # A Xenosaga page whose ONLY recognized game-coded template is a lone {{XG}} cross-reference must
    # not be hijacked to 'XG'. Here the explicit '(XS)' title suffix settles it as the Xenosaga
    # umbrella 'XS' (visible under every Xenosaga filter, excluded from the Xenogears filter). The
    # generic {{XS}} marker independently guards the no-suffix case from the foreign {{XG}} crossref.
    wt = "{{for|the Xenogears object|Zohar (XG)}}\n{{XS}} The Zohar of Xenosaga.\n{{XG}} reference."
    assert derive_game("Zohar (XS)", wt) == "XS"


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


# --- parse_article: structured "data" stat-block templates ---

def test_parse_extracts_data_template_stats():
    # The enemy *stat block* lives in a {{XC1 enemy data}} template, not in the infobox.
    # It holds lv/hp/str etc. (the numbers questions ask about), so it must be captured.
    art = parse_article("Metal Face (Colony 9) (part 1)", 61, load("enemy_xc1.wikitext"), CFG)
    by_template = {b["template"]: b["fields"] for b in art["infoboxes"]}
    assert "XC1 enemy data" in by_template, list(by_template)
    stats = by_template["XC1 enemy data"]
    assert stats["lv"] == "10"
    assert stats["hp"] == "124"
    assert stats["str"] == "201"


def test_parse_game_link_template_renders_display_text():
    # {{XC1|Colony 9}} is a game-namespaced link shortcut; strip_code deletes it, leaving
    # "Battle of ." Render it to its display text so prose keeps its locations/links.
    art = parse_article("Metal Face (Colony 9) (part 1)", 61, load("enemy_xc1.wikitext"), CFG)
    intro = {s["heading"]: s["text"] for s in art["sections"]}["Introduction"]
    assert "Battle of Colony 9" in intro
    assert "Residential District" in intro


def test_parse_infobox_resolves_game_links():
    # The infobox "location" field is {{XC1|Colony 9}} ({{XC1|Residential District}}); without
    # rendering it collapses to "()".
    art = parse_article("Metal Face (Colony 9) (part 1)", 61, load("enemy_xc1.wikitext"), CFG)
    ib = {b["template"]: b["fields"] for b in art["infoboxes"]}["Infobox XC1 enemy"]
    assert ib["location"] == "Colony 9 (Residential District)"


def test_parse_keeps_text_under_subsection_headings():
    # A level-2 section's === subsections must be kept; flat=True with levels=[2] would drop them entirely.
    wt = ("Lead sentence about the topic and a little more to clear the stub threshold.\n"
          "== History ==\nHistory body text.\n"
          "=== Early years ===\nSubsection text that must survive indexing.\n"
          "== Gameplay ==\nGameplay body text.\n")
    art = parse_article("Topic", 7, wt, CFG)
    by_heading = {s["heading"]: s["text"] for s in art["sections"]}
    assert set(by_heading) == {"Introduction", "History", "Gameplay"}
    assert "History body text." in by_heading["History"]
    assert "Subsection text that must survive indexing." in by_heading["History"]
    assert "Subsection text" not in by_heading["Gameplay"]


# --- parse_article: drops ---

def test_parse_drops_redirect():
    assert parse_article("Foo", 3, "#REDIRECT [[Bar]]", CFG) is None


def test_parse_drops_short_stub():
    assert parse_article("Foo", 4, "tiny", CFG) is None


def test_parse_drops_disambiguation():
    wt = "{{disambiguation}}\nThis could refer to several things across the series entries here."
    assert parse_article("Vandham", 5, wt, CFG) is None


# --- hybrid run: wikitext-only pages, drop counting ---

def test_run_hybrid_wikitext_only_writes_articles_and_counts_drops(tmp_path):
    import json

    from xeno_rag.parse_html import run_hybrid

    pages = tmp_path / "pages"
    pages.mkdir()
    (tmp_path / "html").mkdir()
    raw_pages = [
        {"title": "Infinity Blade (XC3) (Noah)", "pageid": 70047,
         "revisions": [{"slots": {"main": {"content": load("art_xc3.wikitext")}}}]},
        {"title": "Foo", "pageid": 3,
         "revisions": [{"slots": {"main": {"content": "#REDIRECT [[Bar]]"}}}]},
    ]
    (pages / "pages_00000.jsonl").write_text(
        "\n".join(json.dumps(r) for r in raw_pages), encoding="utf-8")
    out = tmp_path / "articles.jsonl"
    cfg = {"min_wikitext_bytes": 50,
           "paths": {"articles": str(out), "pages": str(pages), "html": str(tmp_path / "html")}}
    stats = run_hybrid(cfg)
    assert stats == {"from_html": 0, "from_wikitext": 1, "dropped": 1}
    assert out.read_text(encoding="utf-8").strip().count("\n") == 0  # one line
