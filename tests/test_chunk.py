"""Tests for section-aware chunking + infobox-as-text rendering."""

import json

from xeno_rag.chunk import chunk_article, run, split_with_overlap

ARTICLE = {
    "title": "Infinity Blade (XC3) (Noah)",
    "pageid": 70047,
    "game": "XC3",
    "url": "https://www.xenoserieswiki.org/wiki/Infinity_Blade_(XC3)_(Noah)",
    "infoboxes": [{
        "template": "Infobox XC3 art",
        "fields": {"power": "250", "recharge": "4 turns", "type": "Talent Art", "user": "Noah"},
    }],
    "sections": [
        {"heading": "Introduction", "text": "Infinity Blade is a Talent Art used by Noah."},
        {"heading": "Mechanics", "text": "Unleashes a powerful slash. Deals damage based on Attack."},
    ],
}

BIG_CFG = {"chunk_max_tokens": 600, "chunk_overlap_tokens": 80}


def test_split_with_overlap_windows():
    words = [f"w{i}" for i in range(10)]
    text = " ".join(words)
    windows = split_with_overlap(text, max_tokens=4, overlap=2)
    assert windows[0].split() == ["w0", "w1", "w2", "w3"]
    # consecutive windows overlap by 2 words
    assert windows[0].split()[-2:] == windows[1].split()[:2]
    # full coverage, last window reaches the end
    assert windows[-1].split()[-1] == "w9"


def test_short_section_is_one_chunk():
    chunks = chunk_article(ARTICLE, BIG_CFG)
    mech = [c for c in chunks if c["heading"] == "Mechanics"]
    assert len(mech) == 1


def test_long_section_splits_into_multiple():
    art = dict(ARTICLE, sections=[{"heading": "Lore", "text": " ".join(f"word{i}" for i in range(60))}],
               infoboxes=[])
    chunks = chunk_article(art, {"chunk_max_tokens": 20, "chunk_overlap_tokens": 5})
    lore = [c for c in chunks if c["heading"] == "Lore"]
    assert len(lore) > 1


def test_infobox_renders_as_sentence():
    chunks = chunk_article(ARTICLE, BIG_CFG)
    ib = [c for c in chunks if c["heading"] == "infobox"]
    assert len(ib) == 1
    text = ib[0]["text"]
    assert "Talent Art" in text
    assert "250" in text
    assert "Power" in text  # field key surfaced, capitalized


def test_data_template_expands_stat_abbreviations():
    # Stat blocks use terse keys (lv/hp/str). Expand the common ones so the chunk reads clearly
    # and so a "what level…" query embeds near "Level: 10" instead of the opaque "Lv: 10".
    art = dict(ARTICLE, infoboxes=[{
        "template": "XC1 enemy data",
        "fields": {"lv": "10", "hp": "124", "str": "201", "agi": "32"},
    }], sections=[])
    text = next(c for c in chunk_article(art, BIG_CFG) if c["heading"] == "infobox")["text"]
    assert "Level: 10" in text
    assert "HP: 124" in text
    assert "Lv: 10" not in text


def test_prose_chunk_has_breadcrumb():
    chunks = chunk_article(ARTICLE, BIG_CFG)
    intro = next(c for c in chunks if c["heading"] == "Introduction")
    assert intro["text"].startswith("[XC3] Infinity Blade (XC3) (Noah) > Introduction:")


def test_every_chunk_nonempty_with_url_and_game():
    chunks = chunk_article(ARTICLE, BIG_CFG)
    assert chunks  # non-empty list
    for c in chunks:
        assert c["text"].strip()
        assert c["url"].startswith("https://")
        assert c["game"] == "XC3"
        assert c["title"] == "Infinity Blade (XC3) (Noah)"


def test_chunk_ids_unique():
    chunks = chunk_article(ARTICLE, BIG_CFG)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_run_writes_chunks(tmp_path):
    out = tmp_path / "chunks.jsonl"
    cfg = dict(BIG_CFG, paths={"chunks": str(out)})
    n = run(cfg, articles=[ARTICLE])
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert n == len(lines)
    assert n >= 3  # 1 infobox + 2 prose
    rec = json.loads(lines[0])
    assert {"chunk_id", "pageid", "title", "game", "heading", "url", "text"} <= set(rec)


def test_html_factblocks_become_breadcrumbed_chunks():
    """Rendered-HTML fact tables (parse_html factblocks) become retrievable 'Label: value' chunks."""
    article = {
        "title": "Mythra/Gameplay (XC2)", "pageid": 22432, "game": "XC2",
        "url": "https://www.xenoserieswiki.org/wiki/Mythra/Gameplay_(XC2)",
        "sections": [],
        "factblocks": [
            {"heading": "Stats", "lines": ["Element: Light.", "Role: ATK.", "Weapon class: Aegis Sword."]},
        ],
    }
    chunks = chunk_article(article, {})
    joined = " ".join(c["text"] for c in chunks)
    assert "Element: Light" in joined
    assert "[XC2] Mythra/Gameplay (XC2) > Stats:" in joined
    assert any(c["heading"] == "Stats" for c in chunks)


def test_overlap_not_smaller_than_window_is_rejected_with_a_clear_message():
    """overlap >= max_tokens advances one word per window (one chunk per word): reject it up front."""
    import pytest

    from xeno_rag.errors import SetupError

    for bad in ({"chunk_max_tokens": 50, "chunk_overlap_tokens": 50},
                {"chunk_max_tokens": 50, "chunk_overlap_tokens": 80},
                {"chunk_max_tokens": 0, "chunk_overlap_tokens": 0},
                {"chunk_max_tokens": 50, "chunk_overlap_tokens": -1}):
        with pytest.raises(SetupError, match="chunk_overlap_tokens"):
            chunk_article(ARTICLE, bad)


def test_oversized_infobox_is_split_and_every_window_keeps_its_breadcrumb():
    big = dict(ARTICLE, infoboxes=[{
        "template": "Infobox XC3 art",
        "fields": {f"field{i}": f"value{i}" for i in range(200)},
    }])
    chunks = [c for c in chunk_article(big, {"chunk_max_tokens": 100, "chunk_overlap_tokens": 10})
              if c["heading"] == "infobox"]
    assert len(chunks) > 1
    assert all(c["text"].startswith("[XC3] Infinity Blade (XC3) (Noah) > infobox: ") for c in chunks)
    assert all(len(c["text"].split()) <= 100 + 8 for c in chunks)       # window plus the breadcrumb


def test_run_keeps_the_old_chunks_file_when_the_input_fails_midway(tmp_path):
    """A missing/failing article source must not truncate the existing chunks file (atomic write)."""
    import pytest

    out = tmp_path / "chunks.jsonl"
    out.write_text("old-content" + chr(10), encoding="utf-8")
    cfg = dict(BIG_CFG, paths={"chunks": str(out), "articles": str(tmp_path / "missing.jsonl")})
    with pytest.raises(FileNotFoundError):
        run(cfg)
    assert out.read_text(encoding="utf-8") == "old-content" + chr(10)
    assert not (tmp_path / "chunks.jsonl.tmp").exists()
