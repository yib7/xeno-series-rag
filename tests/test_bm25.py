"""Tests for the SQLite-FTS5 BM25 lexical index (exact proper-noun / concept recall)."""

import os
import sys

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


def test_single_character_token_is_searchable(tmp_path):
    # "N" is a real XC3 character; the old `len(t) > 1` filter discarded exactly the rare-exact-name
    # query BM25 exists to fix (P2-9a).
    chunks = [
        {"chunk_id": "n-0", "pageid": 6, "title": "N", "game": "XC3", "heading": "Introduction",
         "url": "u6", "text": "N is a Moebius and Noah's alternate self in Xenoblade Chronicles 3."},
    ]
    idx = Bm25Index.build(chunks, path=str(tmp_path / "n.sqlite3"))
    assert idx.search("Who is N?", n=5) == ["n-0"]


def test_stopwords_dropped_when_content_words_remain():
    # The MATCH expression keeps only content tokens when any exist (P2-9b), still quoted.
    from xeno_rag.bm25_index import _match_query

    q = _match_query("Who is Shulk and where does he live?")
    assert '"shulk"' in q and '"live"' in q
    for stop in ('"who"', '"is"', '"and"', '"where"', '"does"', '"he"'):
        assert stop not in q


def test_all_stopword_query_falls_back_to_keeping_tokens(index):
    # A question made only of stopwords must not collapse to an empty MATCH — the tokens are kept
    # so the search still returns whatever matches.
    from xeno_rag.bm25_index import _match_query

    assert _match_query("who is that") == '"who" OR "is" OR "that"'
    # end-to-end: "is ... in ..." appears in the corpus text, so results are non-empty
    assert index.search("what is in there", n=5)


def test_query_with_fts_special_chars_does_not_crash(index):
    # User questions contain punctuation that is FTS5 syntax ("-", quotes, parens). Must be sanitized.
    assert isinstance(index.search('what is a "mimeosome" (XCX)? - really', n=5), list)


def test_concurrent_search_on_shared_connection_is_safe(index):
    """The web server's threadpool shares one Bm25Index (one sqlite connection); concurrent
    searches must all succeed and return correct results (audit suspicion S2)."""
    import concurrent.futures

    def do_search(i):
        q = "mimeosome" if i % 2 == 0 else "Skell weapon attack"
        want = "1-0" if i % 2 == 0 else "2-0"
        return want in index.search(q, n=5)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(do_search, range(64)))
    assert all(results)


# --- atomic rebuild (build to temp, os.replace into place) ---

NEW_CHUNKS = [
    {"chunk_id": "5-0", "pageid": 5, "title": "Noah", "game": "XC3", "heading": "Introduction",
     "url": "u5", "text": "Noah is an off-seer from the nation of Keves."},
]


def test_rebuild_is_atomic_old_index_present_until_replace(tmp_path, monkeypatch):
    """The destination file must never be missing: the new index is built to a temp file and the
    old one stays readable right up to the os.replace swap."""
    import xeno_rag.bm25_index as bm

    path = str(tmp_path / "bm25.sqlite3")
    Bm25Index.build(CHUNKS, path=path).close()

    real_replace = os.replace
    seen = {}

    def spying_replace(src, dst):
        seen["at_replace_dst_exists"] = os.path.exists(dst)
        seen["src"] = src
        # the OLD index must still be fully readable at this instant
        old = Bm25Index(path=dst)
        seen["old_still_serves"] = old.search("mimeosome", n=3)[0] == "1-0"
        old.close()
        return real_replace(src, dst)

    monkeypatch.setattr(bm.os, "replace", spying_replace)
    idx = Bm25Index.build(NEW_CHUNKS, path=path)
    assert seen["at_replace_dst_exists"], "old index file must exist until the swap"
    assert seen["old_still_serves"]
    assert seen["src"] == path + ".tmp"
    assert idx.search("off-seer Keves", n=3)[0] == "5-0"   # swapped-in index serves new content
    assert not os.path.exists(path + ".tmp")               # temp file consumed by the replace
    idx.close()


def test_rebuild_overwrites_stale_temp_file(tmp_path):
    path = str(tmp_path / "bm25.sqlite3")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write("garbage from an interrupted build")
    idx = Bm25Index.build(CHUNKS, path=path)
    assert idx.search("mimeosome", n=3)[0] == "1-0"
    idx.close()



# --- from_collection count-parity guard (P2-5) ---

GOOD_COLLECTION_CHUNKS = [
    ("1-0", "A mimeosome is an artificial body used by humanity in New Los Angeles.",
     {"game": "XCX", "title": "Mimeosome"}),
    ("2-0", "SKM-M230ME Claymore is a Skell weapon with high attack power.",
     {"game": "XCX", "title": "SKM-M230ME Claymore"}),
    ("3-0", "KOS-MOS is an anti-Gnosis battle android built by Vector Industries.",
     {"game": "series", "title": "KOS-MOS"}),
    ("4-0", "Shulk wields the Monado in Xenoblade Chronicles.",
     {"game": "XC1", "title": "Shulk"}),
]


class FakeCollection:
    """Mimics the slice of the chromadb collection API that `_from_collection_obj` uses:
    `count()` and paginated `get(include, limit, offset)`. `rows` is a list of (id, doc, meta)
    tuples; `pages` optionally overrides what each successive `get()` call returns, to simulate
    an offset-paginated get() whose ordering is unstable across pages."""

    def __init__(self, rows, pages=None):
        self.rows = rows
        self.pages = pages
        self._calls = 0

    def count(self):
        return len(self.rows)

    def get(self, include=None, limit=None, offset=None):
        if self.pages is not None:
            page = self.pages[self._calls]
            self._calls += 1
        else:
            page = self.rows[offset:offset + limit]
        return {
            "ids": [r[0] for r in page],
            "documents": [r[1] for r in page],
            "metadatas": [r[2] for r in page],
        }


def test_from_collection_verifies_count_parity(tmp_path):
    cfg = {"paths": {"bm25": str(tmp_path / "bm25.sqlite3")}}
    col = FakeCollection(GOOD_COLLECTION_CHUNKS)

    idx = Bm25Index._from_collection_obj(col, cfg=cfg, page=2)

    assert idx.count == len(GOOD_COLLECTION_CHUNKS)
    assert idx.search("mimeosome", n=5)[0] == "1-0"  # a normal search still works


def test_from_collection_raises_on_dropped_or_duplicated_chunk(tmp_path):
    cfg = {"paths": {"bm25": str(tmp_path / "bm25.sqlite3")}}
    # Unstable paginated get(): page 2 (offset=2) re-returns row "2-0" instead of advancing to
    # "3-0"/"4-0", so a chunk_id is duplicated and another is dropped entirely - exactly the
    # failure mode an unstable get() ordering would produce across successive offset calls.
    unstable_pages = [
        GOOD_COLLECTION_CHUNKS[0:2],   # offset=0: "1-0", "2-0"
        GOOD_COLLECTION_CHUNKS[1:3],   # offset=2: repeats "2-0", drops "4-0"
    ]
    col = FakeCollection(GOOD_COLLECTION_CHUNKS, pages=unstable_pages)

    with pytest.raises(RuntimeError):
        Bm25Index._from_collection_obj(col, cfg=cfg, page=2)


def test_from_collection_raises_on_pure_row_drop(tmp_path):
    # A paginated get() that drops a row outright (no duplicate chunk_id anywhere) must still be
    # caught: idx.count < total, but the duplicate-chunk_id guard (b) never fires because nothing
    # repeats. This exercises the count-parity guard (a) - the literal P2-5 requirement - in
    # isolation, distinct from test_from_collection_raises_on_dropped_or_duplicated_chunk above
    # (which trips guard (b) instead).
    dropped_pages = [
        GOOD_COLLECTION_CHUNKS[0:2],   # offset=0: "1-0", "2-0"
        GOOD_COLLECTION_CHUNKS[3:4],   # offset=2: skips "3-0" straight to "4-0", nothing duplicated
    ]
    cfg = {"paths": {"bm25": str(tmp_path / "bm25.sqlite3")}}
    col = FakeCollection(GOOD_COLLECTION_CHUNKS, pages=dropped_pages)

    with pytest.raises(RuntimeError, match="rows but the collection has"):
        Bm25Index._from_collection_obj(col, cfg=cfg, page=2)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only open-handle file lock")
def test_rebuild_with_open_reader_raises_actionable_error(tmp_path):
    """On Windows an open sqlite handle blocks os.replace; the operator must get a 'stop the
    server' message, not a raw PermissionError."""
    path = str(tmp_path / "bm25.sqlite3")
    Bm25Index.build(CHUNKS, path=path).close()
    reader = Bm25Index(path=path)                     # simulates the running server's handle
    reader.search("mimeosome", n=1)
    try:
        with pytest.raises(RuntimeError, match="[Ss]top the server"):
            Bm25Index.build(NEW_CHUNKS, path=path)
    finally:
        reader.close()
    # the freshly built index is preserved at the temp path, as the message promises
    assert os.path.exists(path + ".tmp")
