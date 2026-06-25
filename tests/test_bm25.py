"""Tests for the SQLite-FTS5 BM25 lexical index (exact proper-noun / concept recall)."""

import pytest

from xeno_rag.bm25_index import Bm25Index

CHUNKS = [
    {"chunk_id": "1-0", "pageid": 1, "title": "Mimeosome", "game": "XCX", "heading": "Introduction",
     "url": "u1", "text": "A mimeosome is an artificial body used by humanity in New Los Angeles."},
    {"chunk_id": "2-0", "pageid": 2, "title": "SKM-M230ME Claymore", "game": "XCX", "heading": "infobox",
     "url": "u2", "text": "SKM-M230ME Claymore is a Skell weapon with high attack power."},
    {"chunk_id": "3-0", "pageid": 3, "title": "KOS-MOS", "game": "series", "heading": "Introduction",
     "url": "u3", "text": "KOS-MOS is an anti-Gnosis battle android built by Vector Industries."},
    {"chunk_id": "4-0", "pageid": 4, "title": "Shulk", "game": "XC1", "heading": "Introduction",
     "url": "u4", "text": "Shulk wields the Monado in Xenoblade Chronicles."},
]


@pytest.fixture()
def index(tmp_path):
    path = str(tmp_path / "bm25.sqlite3")
    return Bm25Index.build(CHUNKS, path=path)


def test_exact_term_ranks_canonical_chunk_first(index):
    # The recall failure we are fixing: dense retrieval buried the Mimeosome page under weapon SKUs.
    # BM25 on the exact term must surface it first.
    ids = index.search("What are mimeosomes?", n=5)
    assert ids[0] == "1-0"


def test_search_returns_ranked_chunk_ids(index):
    ids = index.search("Skell weapon attack", n=5)
    assert "2-0" in ids
    assert isinstance(ids, list) and all(isinstance(i, str) for i in ids)


def test_game_filter_is_series_inclusive(index):
    # Filtering to XC1 keeps XC1 + series, excludes other base games (XCX here).
    ids = index.search("android weapon Monado", n=10, game_filter="XC1")
    assert "4-0" in ids        # XC1
    assert "3-0" in ids        # series (always allowed)
    assert "2-0" not in ids    # XCX excluded
    assert "1-0" not in ids    # XCX excluded


def test_game_filter_none_searches_everything(index):
    ids = index.search("mimeosome", n=10, game_filter=None)
    assert "1-0" in ids


def test_game_filter_multi_game_membership(tmp_path):
    # A chunk carrying an explicit multi-game membership is found under EACH of its games, not others.
    chunks = [
        {"chunk_id": "k-0", "pageid": 9, "title": "KOS-MOS", "game": "series",
         "games": ["XS1", "XS2", "XS3", "XC2"], "heading": "Introduction", "url": "u",
         "text": "KOS-MOS is an anti-Gnosis android and a Blade."},
        {"chunk_id": "p-0", "pageid": 8, "title": "Poppi", "game": "XC2", "heading": "Introduction",
         "url": "u", "text": "Poppi is an artificial Blade android built by Tora."},
    ]
    idx = Bm25Index.build(chunks, path=str(tmp_path / "m.sqlite3"))
    assert "k-0" in idx.search("android", n=10, game_filter="XC2")   # cameo game
    assert "k-0" in idx.search("android", n=10, game_filter="XS3")   # home game
    assert "k-0" not in idx.search("android", n=10, game_filter="XC1")  # not a member
    assert "p-0" not in idx.search("android", n=10, game_filter="XS3")  # Poppi is XC2-only


def test_reopen_persisted_index(tmp_path):
    path = str(tmp_path / "bm25.sqlite3")
    Bm25Index.build(CHUNKS, path=path)
    reopened = Bm25Index(path=path)          # open the persisted file, no rebuild
    assert reopened.search("mimeosome", n=3)[0] == "1-0"


def test_query_with_fts_special_chars_does_not_crash(index):
    # User questions contain punctuation that is FTS5 syntax ("-", quotes, parens). Must be sanitized.
    assert isinstance(index.search('what is a "mimeosome" (XCX)? - really', n=5), list)
