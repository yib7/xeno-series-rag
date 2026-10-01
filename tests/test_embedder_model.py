"""Integration test against the real Qwen3-Embedding model (marker ``model``, skipped by default).

Every other retrieval test uses ``tests.fakes.HashingEmbedder`` so the default suite needs no model, no
network, and no ~1.2 GB download. This one proves the production embedder + Chroma + cosine ranking
work together end to end. Run it with ``pytest -m model``; the first run downloads the model from the
Hugging Face hub (free, no key) and later runs use the local cache.
"""

import pytest

from xeno_rag.embed_index import Embedder, build_index, query

pytestmark = pytest.mark.model

CHUNKS = [
    {"chunk_id": "1-0", "pageid": 1, "title": "Infinity Blade (XC3) (Noah)", "game": "XC3",
     "heading": "infobox", "url": "https://w/Infinity_Blade",
     "text": "[XC3] Infinity Blade > infobox: Infobox XC3 art. Power: 250. Type: Talent Art. User: Noah."},
    {"chunk_id": "2-0", "pageid": 2, "title": "Rex (XC2)", "game": "XC2",
     "heading": "Introduction", "url": "https://w/Rex",
     "text": "[XC2] Rex > Introduction: Rex is the salvager protagonist of Xenoblade Chronicles 2."},
    {"chunk_id": "3-0", "pageid": 3, "title": "Fei Fong Wong (XG)", "game": "XG",
     "heading": "Introduction", "url": "https://w/Fei",
     "text": "[XG] Fei Fong Wong > Introduction: Fei pilots the gear Weltall in Xenogears."},
]


def test_real_qwen_embedder_ranks_the_on_topic_chunk_first(tmp_path):
    cfg = {
        "embed_model": "Qwen/Qwen3-Embedding-0.6B",
        "embed_device": "cpu",
        "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "embed_tokenizer_kwargs": {"padding_side": "left"},
        "collection_name": "model_test",
        "top_k": 3,
        "paths": {"vectorstore": str(tmp_path)},
    }
    embedder = Embedder(cfg)
    assert build_index(CHUNKS, cfg, embedder=embedder) == len(CHUNKS)
    results = query("How much power does Infinity Blade have?", cfg, embedder=embedder)
    assert results and results[0]["game"] == "XC3"
    assert "Infinity Blade" in results[0]["text"]
