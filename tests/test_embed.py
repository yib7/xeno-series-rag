"""Tests for embedding + ChromaDB indexing against a temp store.

Vectors come from the deterministic ``HashingEmbedder`` (no model, no download). The real
Qwen3-Embedding model is exercised in ``tests/test_embedder_model.py`` (marker ``model``).
"""

import json

import numpy as np
import pytest

from xeno_rag.embed_index import (
    _l2_normalize,
    _where,
    build_index,
    cap_per_page,
    dense_query,
    query,
)
from xeno_rag.embed_index import run as run_embed

from .fakes import HashingEmbedder


class _FakeEmbedder:
    """Embeds any query to a fixed 3-d vector, lets dense_query tests run without the real model."""
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class _FakeCollection:
    """Returns a canned ChromaDB ``query`` payload so array-shape edge cases can be exercised."""
    def __init__(self, payload):
        self._payload = payload

    def query(self, **kw):
        return self._payload


def test_dense_query_raises_on_mismatched_result_arrays(monkeypatch):
    """ChromaDB is contracted to return equal-length id/doc/meta/dist arrays. If it ever returns
    ragged arrays (an API change or a corrupted store), dense_query must fail loudly and clearly
    (a strict zip's ValueError) rather than silently slicing a shorter array by index: the latter
    either IndexErrors or fabricates mismatched rows."""
    from xeno_rag import embed_index

    payload = {
        "ids": [["a", "b"]],           # two ids ...
        "documents": [["doc-a"]],      # ... but only one document (ragged)
        "metadatas": [[{"pageid": 1}]],
        "distances": [[0.1]],
    }
    monkeypatch.setattr(embed_index, "_collection", lambda cfg, client, **kw: _FakeCollection(payload))
    with pytest.raises(ValueError):
        dense_query("q", {"top_k": 8}, embedder=_FakeEmbedder())


def test_dense_query_maps_aligned_arrays(monkeypatch):
    """The normal (aligned) case still maps every row to a result dict, distance included."""
    from xeno_rag import embed_index

    payload = {
        "ids": [["a", "b"]],
        "documents": [["doc-a", "doc-b"]],
        "metadatas": [[{"pageid": 1}, {"pageid": 2}]],
        "distances": [[0.1, 0.2]],
    }
    monkeypatch.setattr(embed_index, "_collection", lambda cfg, client, **kw: _FakeCollection(payload))
    out = dense_query("q", {"top_k": 8}, embedder=_FakeEmbedder())
    assert [r["chunk_id"] for r in out] == ["a", "b"]
    assert [r["text"] for r in out] == ["doc-a", "doc-b"]
    assert [r["distance"] for r in out] == [0.1, 0.2]
    assert out[0]["pageid"] == 1


class _ZeroVectorEmbedder:
    """Embeds every doc/query to a degenerate zero vector, the raw case that makes naive L2
    normalization emit NaN. Reuses the real embedder's safe ``_l2_normalize`` so the sanitized
    (finite) vectors are what actually reach ChromaDB, mirroring the production code path."""
    def encode(self, texts):
        return _l2_normalize(np.zeros((len(list(texts)), 4), dtype=np.float32))

    def embed_query(self, text):
        return _l2_normalize(np.zeros((1, 4), dtype=np.float32))[0]


def test_end_to_end_retrieval_survives_degenerate_zero_vectors(cfg, embedder):
    """Integration: a chunk (and query) that embeds to a degenerate zero vector must not crash the
    index build or the query. `test_l2_normalize_never_produces_nan_or_inf` covers the unit; this
    proves the sanitized vectors actually flow through build_index -> ChromaDB -> query without a
    'must not contain NaN or Infinity' rejection."""
    chunks = [
        {"chunk_id": "z-0", "pageid": 900, "title": "Degenerate", "game": "XC1",
         "heading": "Introduction", "url": "https://w/Degenerate", "text": ""},
        {"chunk_id": "z-1", "pageid": 901, "title": "Also Degenerate", "game": "XC1",
         "heading": "Introduction", "url": "https://w/Also", "text": ""},
    ]
    c2 = {**cfg, "collection_name": "degenerate_test"}
    zero = _ZeroVectorEmbedder()
    count = build_index(chunks, c2, embedder=zero)      # must not raise on NaN/Inf embeddings
    assert count == len(chunks)
    results = query("anything", c2, embedder=zero)      # querying with a zero vector must not crash
    assert isinstance(results, list)                    # returns (some ordering of) the chunks, no error


def test_build_index_skips_chunk_with_none_pageid(caplog):
    """A chunk with pageid=None must be skipped with a logged warning, not crash the whole embed
    run when ChromaDB rejects the None metadata value (audit suspicion S1)."""
    import logging

    class _RecordingCollection:
        def __init__(self):
            self.added = []

        def get(self, ids=None, **kw):
            return {"ids": []}

        def add(self, ids, embeddings, metadatas, documents):
            self.added.extend(ids)

        def count(self):
            return len(self.added)

    class _StubClient:
        def __init__(self, col):
            self._col = col

        def get_or_create_collection(self, name, metadata=None):
            return self._col

    class _StubEmbedder:
        def encode(self, texts):
            return [[0.0, 1.0] for _ in texts]

    col = _RecordingCollection()
    chunks = [
        {"chunk_id": "ok-0", "pageid": 1, "title": "Fine", "game": "XC1",
         "heading": "Introduction", "url": "u", "text": "fine"},
        {"chunk_id": "None-0", "pageid": None, "title": "Broken", "game": "XC1",
         "heading": "Introduction", "url": "u", "text": "broken"},
    ]
    with caplog.at_level(logging.WARNING, logger="xeno_rag.embed_index"):
        count = build_index(chunks, {"collection_name": "s1"}, embedder=_StubEmbedder(),
                            client=_StubClient(col))
    assert count == 1 and col.added == ["ok-0"]          # good chunk indexed, bad one skipped
    assert any("missing pageid" in r.message for r in caplog.records)


def test_where_filters_on_game_membership_flag():
    # Multi-tag membership: a base-game filter selects chunks whose membership boolean for that game
    # is set (a page can belong to several games at once, e.g. KOS-MOS -> XS1/XS2/XS3/XC2).
    assert _where("XS2") == {"g_XS2": True}
    assert _where("XC1") == {"g_XC1": True}


def test_where_none_or_non_base_is_unrestricted():
    assert _where(None) is None
    assert _where("series") is None


def test_l2_normalize_never_produces_nan_or_inf():
    # A degenerate chunk can embed to a zero (or non-finite) raw vector; normalizing it by its
    # ~0 norm yields NaN, which ChromaDB rejects ("Embeddings must not contain NaN or Infinity").
    # Safe normalization must keep every row finite.
    arr = [[3.0, 4.0, 0.0],            # ordinary row -> 0.6, 0.8, 0
           [0.0, 0.0, 0.0],            # zero norm -> stays zero, must NOT become NaN
           [float("nan"), 1.0, 0.0],   # non-finite component -> sanitized
           [float("inf"), 2.0, 0.0]]   # non-finite component -> sanitized
    out = np.array(_l2_normalize(arr))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[0], [0.6, 0.8, 0.0], atol=1e-6)
    assert (out[1] == 0).all()


def test_get_embedder_is_cached_per_model_device(monkeypatch):
    """The embedder (a heavy ~1.2GB model load) must be built once and reused across requests, not
    reloaded on every dense_query, the same caching the BM25/reranker already get."""
    from xeno_rag import embed_index

    embed_index._EMBEDDER_CACHE.clear()
    calls = []

    class FakeEmbedder:
        def __init__(self, cfg):
            calls.append(cfg.get("embed_model"))

    monkeypatch.setattr(embed_index, "Embedder", FakeEmbedder)
    cfg = {"embed_model": "m", "embed_device": "cpu"}
    e1 = embed_index._get_embedder(cfg)
    e2 = embed_index._get_embedder(cfg)
    assert e1 is e2 and len(calls) == 1          # built exactly once, then reused


def test_get_client_is_cached_per_vectorstore_path(monkeypatch):
    """The Chroma PersistentClient is cached per vectorstore path, so the request path doesn't
    re-open the store on every query."""
    from xeno_rag import embed_index

    embed_index._CLIENT_CACHE.clear()
    calls = []

    class FakeClient:
        def __init__(self, path, settings=None):
            calls.append(path)

    monkeypatch.setattr(embed_index.chromadb, "PersistentClient", FakeClient)
    cfg = {"paths": {"vectorstore": "/tmp/vs"}}
    c1 = embed_index._get_client(cfg)
    c2 = embed_index._get_client(cfg)
    assert c1 is c2 and len(calls) == 1


CHUNKS = [
    {"chunk_id": "1-0", "pageid": 1, "title": "Infinity Blade (XC3) (Noah)", "game": "XC3",
     "heading": "infobox", "url": "https://w/Infinity_Blade",
     "text": "[XC3] Infinity Blade > infobox: Infobox XC3 art. Power: 250. Type: Talent Art. User: Noah."},
    {"chunk_id": "1-1", "pageid": 1, "title": "Infinity Blade (XC3) (Noah)", "game": "XC3",
     "heading": "Mechanics", "url": "https://w/Infinity_Blade",
     "text": "[XC3] Infinity Blade > Mechanics: Noah unleashes a powerful slashing talent art."},
    {"chunk_id": "2-0", "pageid": 2, "title": "Rex (XC2)", "game": "XC2",
     "heading": "Introduction", "url": "https://w/Rex",
     "text": "[XC2] Rex > Introduction: Rex is the salvager protagonist of Xenoblade Chronicles 2."},
    {"chunk_id": "3-0", "pageid": 3, "title": "Fei Fong Wong (XG)", "game": "XG",
     "heading": "Introduction", "url": "https://w/Fei",
     "text": "[XG] Fei Fong Wong > Introduction: Fei pilots the gear Weltall in Xenogears."},
]


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    d = tmp_path_factory.mktemp("vectorstore")
    return {
        "embed_model": "Qwen/Qwen3-Embedding-0.6B",
        "embed_device": "cpu",
        "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "embed_tokenizer_kwargs": {"padding_side": "left"},
        "collection_name": "test_xeno",
        "top_k": 3,
        "paths": {"vectorstore": str(d)},
    }


@pytest.fixture(scope="module")
def embedder(cfg):
    return HashingEmbedder(cfg)


@pytest.fixture(scope="module")
def indexed(cfg, embedder):
    return build_index(CHUNKS, cfg, embedder=embedder)


def test_collection_count_matches_chunks(indexed):
    assert indexed == len(CHUNKS)


def test_query_returns_on_topic_chunk(cfg, embedder, indexed):
    results = query("How much power does Infinity Blade have?", cfg, embedder=embedder)
    assert results, "expected at least one result"
    assert "Infinity Blade" in results[0]["text"]
    assert results[0]["game"] == "XC3"


def test_query_carries_metadata(cfg, embedder, indexed):
    results = query("Infinity Blade", cfg, embedder=embedder)
    top = results[0]
    assert top["url"].startswith("https://")
    assert "title" in top and "game" in top and "chunk_id" in top


def test_game_filter_restricts_results(cfg, embedder, indexed):
    results = query("protagonist", cfg, game_filter="XC2", embedder=embedder)
    assert results
    assert all(r["game"] in ("XC2", "series") for r in results)


def test_game_filter_includes_series_pages(cfg, embedder):
    """A game filter must still surface cross-game 'series' pages (recurring bosses, lore).

    Most wiki pages have no '(XCn)' title suffix, so they are tagged 'series'. A hard
    game filter would hide them (e.g. the 'Metal Face' boss for an XC1 question); the
    filter must include the 'series' bucket alongside the selected game.
    """
    chunks = [
        {"chunk_id": "s-0", "pageid": 10, "title": "Metal Face", "game": "series",
         "heading": "Introduction", "url": "https://w/Metal_Face",
         "text": "[series] Metal Face > Introduction: Metal Face is a Faced Mechon boss "
                 "fought in Xenoblade Chronicles."},
        {"chunk_id": "x-0", "pageid": 11, "title": "Unrelated (XC2)", "game": "XC2",
         "heading": "Introduction", "url": "https://w/Unrelated",
         "text": "[XC2] Unrelated > Introduction: an unrelated Xenoblade 2 weapon entry."},
    ]
    c2 = {**cfg, "collection_name": "series_filter_test"}
    build_index(chunks, c2, embedder=embedder)
    results = query("Who is Metal Face?", c2, game_filter="XC1", embedder=embedder)
    titles = [r["title"] for r in results]
    assert "Metal Face" in titles            # series page surfaces under an XC1 filter
    assert "Unrelated (XC2)" not in titles   # a *different* game stays excluded


def test_game_filter_multi_game_membership(cfg, embedder):
    """A cross-appearance page (membership in several games) surfaces under EACH of its games and
    nowhere else: the multi-tag fix for KOS-MOS-class pages (Xenosaga lead + Xenoblade cameo)."""
    chunks = [
        {"chunk_id": "k-0", "pageid": 20, "title": "KOS-MOS", "game": "series",
         "games": ["XS1", "XS2", "XS3", "XC2"], "heading": "Introduction", "url": "https://w/KOS-MOS",
         "text": "[series] KOS-MOS > Introduction: KOS-MOS is an anti-Gnosis android, also an XC2 Blade."},
        {"chunk_id": "s-1", "pageid": 21, "title": "Shion", "game": "XS",
         "games": ["XS1", "XS2", "XS3"], "heading": "Introduction", "url": "https://w/Shion",
         "text": "[XS] Shion > Introduction: Shion Uzuki is the lead engineer of the KOS-MOS Project."},
    ]
    c2 = {**cfg, "collection_name": "membership_test"}
    build_index(chunks, c2, embedder=embedder)
    # KOS-MOS appears under XC2 (a cameo game) ...
    assert "KOS-MOS" in [r["title"] for r in query("KOS-MOS android", c2, game_filter="XC2", embedder=embedder)]
    # ... and under XS1 (a home game), alongside Shion ...
    xs1 = [r["title"] for r in query("KOS-MOS Shion", c2, game_filter="XS1", embedder=embedder)]
    assert "KOS-MOS" in xs1 and "Shion" in xs1
    # ... but NOT under XG or XC1 (not in its membership); Shion (Xenosaga-only) absent from XC2.
    assert "KOS-MOS" not in [r["title"] for r in query("KOS-MOS android", c2, game_filter="XG", embedder=embedder)]
    assert "KOS-MOS" not in [r["title"] for r in query("KOS-MOS android", c2, game_filter="XC1", embedder=embedder)]
    assert "Shion" not in [r["title"] for r in query("Shion engineer", c2, game_filter="XC2", embedder=embedder)]


def test_html_cross_appearance_metadata_flags_from_derive_games():
    """End-to-end (fast, no model): a cross-appearance HTML stat page parsed by parse_html_article
    -> chunked -> _metadata must carry per-game flags sourced from derive_games ({XS1,XS2,XS3,XC2}),
    NOT the lossy membership_from_game fallback that a collapsed 'series' label would trigger (which
    would flag EVERY base game). Proves the P1-2 fix propagates all the way to the stored metadata."""
    import os

    from xeno_rag.chunk import chunk_article
    from xeno_rag.embed_index import _metadata
    from xeno_rag.parse_html import parse_html_article

    fx = os.path.join(os.path.dirname(__file__), "fixtures", "html", "kosmos_crossgame.json")
    with open(fx, encoding="utf-8") as f:
        rec = json.load(f)
    art = parse_html_article(rec["title"], rec.get("pageid"), rec["html"], {},
                             wikitext=rec.get("wikitext"))
    # The display label collapses to 'series'; if _metadata used it via membership_from_game it would
    # (wrongly) flag every base game. The real 'games' set must drive the flags instead.
    assert art["game"] == "series"
    assert art["games"] == sorted({"XS1", "XS2", "XS3", "XC2"})

    chunks = chunk_article(art, {})
    assert chunks, "expected at least one chunk from the cross-appearance page"
    for chunk in chunks:
        meta = _metadata(chunk)
        # member games flagged True ...
        for g in ("XS1", "XS2", "XS3", "XC2"):
            assert meta.get(f"g_{g}") is True, f"expected g_{g}=True on {chunk['chunk_id']}"
        # ... and non-member games absent (not flagged): the lossy fallback would have set these.
        for g in ("XG", "XC1", "XC3", "XCX"):
            assert not meta.get(f"g_{g}"), f"g_{g} must be absent/False (not from membership_from_game)"


def test_cap_per_page_keeps_infobox_identity_chunk():
    """The per-page cap must not evict a stat page's infobox (its Location/Species/Level identity
    card) in favour of boilerplate. Reproduces the 'Territorial Rotbart location not found' bug:
    its infobox chunk (rank 3 on the page) was dropped by max_chunks_per_page=2, so the Bionis' Leg
    location never reached the LLM even though it was retrieved."""
    items = [
        {"chunk_id": "7722-15", "pageid": 7722, "heading": "Introduction",
         "text": "[XC1] Territorial Rotbart > Introduction: a unique monster in Xenoblade Chronicles."},
        {"chunk_id": "7722-08", "pageid": 7722, "heading": "Enemy",
         "text": "[XC1] Territorial Rotbart > Enemy: Nametag: Unique. Size category: Extra-Large."},
        {"chunk_id": "7722-00", "pageid": 7722, "heading": "infobox",
         "text": "[XC1] Territorial Rotbart > infobox: Species: Gogol. Location: Bionis' Leg. Level: 81."},
    ]
    out = cap_per_page(items, k=10, per_page_cap=2)
    ids = [c["chunk_id"] for c in out]
    assert len(out) == 2, "cap of 2 per page is still respected"
    assert "7722-00" in ids, "the infobox identity chunk must survive the cap, not be evicted"
    assert ids[0] == "7722-15", "the top-ranked chunk is still kept; only a lower non-infobox is dropped"


def test_cap_per_page_without_infobox_unchanged():
    """No infobox chunk on the page -> behavior identical to before (first N in rank order)."""
    items = [{"chunk_id": f"5-{i}", "pageid": 5, "heading": f"h{i}", "text": f"sec {i}"} for i in range(4)]
    out = cap_per_page(items, k=10, per_page_cap=2)
    assert [c["chunk_id"] for c in out] == ["5-0", "5-1"]


def test_query_caps_chunks_per_page(cfg, embedder):
    """One page with many similar chunks must not monopolize the top-k: query diversifies by page
    (over-fetch, then cap chunks per page) so the LLM sees several distinct sources."""
    chunks = []
    for i in range(6):  # one page, 6 chunks, all strongly on-topic for "chapter"
        chunks.append({"chunk_id": f"100-{i}", "pageid": 100, "title": "Big Page", "game": "XC1",
                       "heading": f"h{i}", "url": "https://w/Big",
                       "text": f"[XC1] Big Page > chapter {i}: this section covers chapter {i} of the story."})
    for p in range(101, 106):  # five other pages, also about chapters
        chunks.append({"chunk_id": f"{p}-0", "pageid": p, "title": f"Page {p}", "game": "XC1",
                       "heading": "intro", "url": f"https://w/{p}",
                       "text": f"[XC1] Page {p} > intro: another entry discussing a story chapter, number {p}."})
    c = {**cfg, "collection_name": "diversify_test", "max_chunks_per_page": 2}
    build_index(chunks, c, embedder=embedder)
    results = query("chapter", c, k=6, embedder=embedder)
    from collections import Counter
    per_page = Counter(r["pageid"] for r in results)
    assert per_page[100] <= 2, f"page 100 should be capped at 2, got {per_page[100]}"
    assert len(per_page) >= 3, "results should span several distinct pages"


def test_build_index_is_idempotent(cfg, embedder, indexed):
    """Re-running over already-embedded chunks must not duplicate (enables resume)."""
    again = build_index(CHUNKS, cfg, embedder=embedder)
    assert again == len(CHUNKS)


def test_build_index_skips_already_embedded(cfg, embedder, indexed):
    """On resume, chunks already in the collection are not re-encoded."""

    class CountingEmbedder:
        def __init__(self, inner):
            self.inner = inner
            self.encoded = 0

        def encode(self, texts):
            texts = list(texts)
            self.encoded += len(texts)
            return self.inner.encode(texts)

        def embed_query(self, text):
            return self.inner.embed_query(text)

    counting = CountingEmbedder(embedder)
    build_index(CHUNKS, cfg, embedder=counting)
    assert counting.encoded == 0  # everything already present → nothing re-encoded


def test_run_indexes_from_chunks_file(tmp_path, embedder):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("\n".join(json.dumps(c) for c in CHUNKS), encoding="utf-8")
    cfg2 = {
        "embed_model": "Qwen/Qwen3-Embedding-0.6B", "embed_device": "cpu",
        "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "embed_tokenizer_kwargs": {"padding_side": "left"},
        "collection_name": "run_test",
        "paths": {"vectorstore": str(tmp_path / "vs"), "chunks": str(chunks_path)},
    }
    assert run_embed(cfg2, embedder=embedder) == len(CHUNKS)


def test_embedder_reads_generic_instruction_and_threads_load_kwargs(monkeypatch):
    """Qwen config: the Embedder reads ``query_instruction``, threads ``embed_tokenizer_kwargs`` to the
    loader (Qwen needs left padding for its last-token pooling), and embed_query prepends the
    instruction to the query text only. Uses a fake SentenceTransformer so no model is downloaded."""
    import sentence_transformers

    from xeno_rag.embed_index import Embedder

    recorded = {"encoded": []}

    class FakeST:
        def __init__(self, name, device=None, model_kwargs=None, tokenizer_kwargs=None, **kw):
            recorded.update(name=name, device=device, tokenizer_kwargs=tokenizer_kwargs)

        def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True):
            recorded["encoded"].extend(texts)
            return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)

    emb = Embedder({
        "embed_model": "Qwen/Qwen3-Embedding-0.6B",
        "embed_device": "cpu",
        "query_instruction": "Instruct: task\nQuery:",
        "embed_tokenizer_kwargs": {"padding_side": "left"},
    })

    assert recorded["device"] == "cpu"
    assert recorded["tokenizer_kwargs"] == {"padding_side": "left"}   # left padding threaded through
    assert emb.query_instruction == "Instruct: task\nQuery:"          # query_instruction is read

    emb.embed_query("what element is Mythra")
    assert recorded["encoded"] == ["Instruct: task\nQuery:what element is Mythra"]  # query-only prefix


def test_read_path_with_missing_store_raises_setup_error_and_creates_nothing(tmp_path):
    from xeno_rag.embed_index import _collection
    from xeno_rag.errors import SetupError

    store = tmp_path / "no-such-store"
    cfg = {"paths": {"vectorstore": str(store)}, "collection_name": "xeno_wiki"}
    with pytest.raises(SetupError, match="scripts.setup"):
        _collection(cfg, create=False)
    assert not store.exists()


def test_drop_collection_swallows_only_not_found():
    """Dropping a collection that does not exist is fine; any other failure (locked/corrupt store)
    must surface instead of letting a rebuild carry on over a half-dropped store."""
    from chromadb.errors import NotFoundError

    from xeno_rag.embed_index import drop_collection

    class Missing:
        def delete_collection(self, name):
            raise NotFoundError(f"Collection [{name}] does not exist")

    class Broken:
        def delete_collection(self, name):
            raise RuntimeError("database is locked")

    drop_collection({}, client=Missing())
    with pytest.raises(RuntimeError, match="locked"):
        drop_collection({}, client=Broken())
