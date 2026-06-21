"""Tests for embedding + ChromaDB indexing. CPU device, temp store, real bge-base model.

The model downloads once on first run (free, no key); subsequent runs use the HF cache.
"""

import pytest

from xeno_rag.embed_index import Embedder, build_index, query

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
        "embed_model": "BAAI/bge-base-en-v1.5",
        "embed_device": "cpu",
        "bge_query_instruction": "Represent this sentence for searching relevant passages: ",
        "collection_name": "test_xeno",
        "top_k": 3,
        "paths": {"vectorstore": str(d)},
    }


@pytest.fixture(scope="module")
def embedder(cfg):
    return Embedder(cfg)


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
    assert all(r["game"] == "XC2" for r in results)
