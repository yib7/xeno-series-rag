"""Tests for retrieval + generation. Mock LLM, no network."""

import pytest

from xeno_rag.embed_index import Embedder, build_index
from xeno_rag.rag import build_prompt, answer, MockLLM, GeminiClient

CHUNKS = [
    {"chunk_id": "1-0", "pageid": 1, "title": "Infinity Blade (XC3) (Noah)", "game": "XC3",
     "heading": "infobox", "url": "https://w/Infinity_Blade",
     "text": "[XC3] Infinity Blade > infobox: Infobox XC3 art. Power: 250. Type: Talent Art."},
    {"chunk_id": "2-0", "pageid": 2, "title": "Rex (XC2)", "game": "XC2",
     "heading": "Introduction", "url": "https://w/Rex",
     "text": "[XC2] Rex > Introduction: Rex is the salvager protagonist of Xenoblade Chronicles 2."},
]


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    d = tmp_path_factory.mktemp("vs_rag")
    return {
        "embed_model": "BAAI/bge-base-en-v1.5", "embed_device": "cpu",
        "bge_query_instruction": "Represent this sentence for searching relevant passages: ",
        "collection_name": "rag_test", "top_k": 3, "gemini_model": "gemini-1.5-flash",
        "paths": {"vectorstore": str(d)},
    }


@pytest.fixture(scope="module")
def embedder(cfg):
    return Embedder(cfg)


@pytest.fixture(scope="module")
def indexed(cfg, embedder):
    build_index(CHUNKS, cfg, embedder=embedder)
    return True


# --- build_prompt ---

def test_build_prompt_has_grounding_rules_and_context():
    system, user = build_prompt("What is the power of Infinity Blade?", CHUNKS)
    s = system.lower()
    assert "only" in s and "context" in s
    assert "infobox" in s
    assert "cite" in s or "source" in s
    assert "Power: 250" in user           # context included
    assert "Source: https://w/Infinity_Blade" in user
    assert "What is the power of Infinity Blade?" in user


# --- answer ---

def test_answer_uses_llm_and_returns_sources(cfg, embedder, indexed):
    llm = MockLLM("Infinity Blade has 250 power.")
    res = answer("How much power does Infinity Blade have?", cfg=cfg, llm=llm, embedder=embedder)
    assert res["answer"] == "Infinity Blade has 250 power."
    assert "https://w/Infinity_Blade" in res["sources"]
    # the mock received grounded context
    assert "Power: 250" in llm.last_prompt


def test_answer_game_filter_restricts_sources(cfg, embedder, indexed):
    llm = MockLLM("ok")
    res = answer("Who is the protagonist?", cfg=cfg, llm=llm, game_filter="XC2", embedder=embedder)
    assert res["sources"] == ["https://w/Rex"]


def test_answer_sources_are_deduped(cfg, embedder, indexed):
    llm = MockLLM("ok")
    res = answer("Infinity Blade", cfg=cfg, llm=llm, embedder=embedder)
    assert len(res["sources"]) == len(set(res["sources"]))


# --- GeminiClient credential gate (no network) ---

def test_gemini_raises_without_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiClient({"gemini_model": "gemini-1.5-flash"})
    with pytest.raises(RuntimeError):
        client.generate("system", "prompt")
