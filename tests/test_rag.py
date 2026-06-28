"""Tests for retrieval + generation. Mock LLM, no network."""

from types import SimpleNamespace

import pytest

from xeno_rag.embed_index import Embedder, build_index
from xeno_rag.rag import (
    build_prompt, answer, answer_stream, MockLLM, GeminiClient, _extract_text, _dedupe_sources,
    _retrieval_query, _apply_answer_style, NO_QUESTION_MESSAGE, SYSTEM_PROMPT,
)

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
        "embed_model": "Qwen/Qwen3-Embedding-0.6B", "embed_device": "cpu",
        "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "embed_tokenizer_kwargs": {"padding_side": "left"},
        "collection_name": "rag_test", "top_k": 3, "gemini_model": "gemini-1.5-flash",
        "max_chunks_per_page": 2, "hybrid_candidates": 10,
        "use_bm25": False, "use_reranker": False,   # answer() tests stay dense-only (no index/model)
        "paths": {"vectorstore": str(d)},
    }


@pytest.fixture(scope="module")
def embedder(cfg):
    return Embedder(cfg)


@pytest.fixture(scope="module")
def indexed(cfg, embedder):
    build_index(CHUNKS, cfg, embedder=embedder)
    return True


# --- answer-style retrieval depth ---

STYLES_CFG = {
    "top_k": 16, "max_chunks_per_page": 4, "hybrid_candidates": 60, "rerank_candidates": 50,
    "answer_styles": {
        "gemini-3.1-flash-lite": {"top_k": 20, "max_chunks_per_page": 5},
        "gemini-3.5-flash": {"top_k": 40, "max_chunks_per_page": 6,
                             "hybrid_candidates": 120, "rerank_candidates": 100},
        "gemini-3.1-pro-preview": {"top_k": 96, "max_chunks_per_page": 10,
                                   "hybrid_candidates": 256, "rerank_candidates": 224},
    },
}


def test_apply_answer_style_thinking_model_goes_deeper():
    cfg = _apply_answer_style({**STYLES_CFG, "gemini_model": "gemini-3.5-flash"})
    assert cfg["top_k"] == 40 and cfg["max_chunks_per_page"] == 6
    assert cfg["hybrid_candidates"] == 120 and cfg["rerank_candidates"] == 100


def test_apply_answer_style_faster_model_stays_lean():
    cfg = _apply_answer_style({**STYLES_CFG, "gemini_model": "gemini-3.1-flash-lite"})
    assert cfg["top_k"] == 20 and cfg["max_chunks_per_page"] == 5
    # not overridden by this style -> base pools retained
    assert cfg["hybrid_candidates"] == 60 and cfg["rerank_candidates"] == 50


def test_apply_answer_style_scholar_model_goes_deepest():
    cfg = _apply_answer_style({**STYLES_CFG, "gemini_model": "gemini-3.1-pro-preview"})
    assert cfg["top_k"] == 96 and cfg["max_chunks_per_page"] == 10
    assert cfg["hybrid_candidates"] == 256 and cfg["rerank_candidates"] == 224


def test_apply_answer_style_unlisted_model_falls_back_to_base():
    cfg = _apply_answer_style({**STYLES_CFG, "gemini_model": "some-other-model"})
    assert cfg["top_k"] == 16 and cfg["max_chunks_per_page"] == 4


def test_apply_answer_style_is_noop_without_map_and_never_mutates():
    src = {"gemini_model": "gemini-3.5-flash", "top_k": 7}  # no answer_styles key
    assert _apply_answer_style(src) is src                  # unchanged identity
    src2 = dict(STYLES_CFG, gemini_model="gemini-3.5-flash")
    _apply_answer_style(src2)
    assert src2["top_k"] == 16                              # original not mutated


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


def test_build_prompt_includes_game_scope_when_filtered():
    _, user = build_prompt("Who is the protagonist?", CHUNKS, game_filter="XC2")
    assert "Xenoblade Chronicles 2" in user or "XC2" in user
    # No filter -> no scope line (asking across the whole series).
    _, plain = build_prompt("Who is the protagonist?", CHUNKS)
    assert "focused on" not in plain.lower()


def test_system_prompt_requests_markdown_tables_and_structure():
    s = SYSTEM_PROMPT.lower()
    assert "table" in s                    # nudge to tabulate multi-stat comparisons
    assert "markdown" in s                  # answer is rendered as markdown
    assert "do not invent" in s or "only" in s   # grounding guard retained


def test_build_prompt_invites_grounded_reasoning():
    # The model must be allowed to reason over the context (count, total, take the max, infer a
    # range) and give a best-effort partial answer — not flatly refuse when no single chunk states
    # the answer verbatim. This is what lets "chapters up to 17" -> "about 17 chapters".
    system, _ = build_prompt("How many chapters are in the game?", CHUNKS)
    s = system.lower()
    assert "reason" in s or "infer" in s or "count" in s
    assert "partial" in s or "best" in s


# --- answer ---

def test_answer_uses_llm_and_returns_sources(cfg, embedder, indexed):
    llm = MockLLM("Infinity Blade has 250 power.")
    res = answer("How much power does Infinity Blade have?", cfg=cfg, llm=llm, embedder=embedder)
    assert res["answer"] == "Infinity Blade has 250 power."
    assert any(s["url"] == "https://w/Infinity_Blade" for s in res["sources"])
    # the mock received grounded context
    assert "Power: 250" in llm.last_prompt


STAT_PAGE_CHUNKS = [
    {"chunk_id": "50-0000", "pageid": 50, "title": "Big Boss", "game": "XC1", "heading": "infobox",
     "url": "https://w/Big", "text": "[XC1] Big Boss > infobox: Species: Gogol. Location: Bionis' Leg. Level 81."},
    {"chunk_id": "50-0001", "pageid": 50, "title": "Big Boss", "game": "XC1", "heading": "Enemy",
     "url": "https://w/Big", "text": "[XC1] Big Boss > Enemy: HP: 416200. STR: 1721."},
    {"chunk_id": "50-0002", "pageid": 50, "title": "Big Boss", "game": "XC1", "heading": "Enemy",
     "url": "https://w/Big", "text": "[XC1] Big Boss > Enemy: Break 100% Res. Topple Vulnerable."},
    {"chunk_id": "50-0003", "pageid": 50, "title": "Big Boss", "game": "XC1", "heading": "Drops",
     "url": "https://w/Big", "text": "[XC1] Big Boss > Drops: Gogol Horn 83 percent."},
]


def test_answer_merges_stat_page_profile_into_prompt(tmp_path):
    """End-to-end: a location question retrieves only the infobox, but auto-merge folds the page's
    full profile into the prompt — so sibling facts the query would never rank first (resistances,
    drops) still reach the model. No re-embed; retrieval is unchanged."""
    c = {"embed_model": "Qwen/Qwen3-Embedding-0.6B", "embed_device": "cpu",
         "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
         "embed_tokenizer_kwargs": {"padding_side": "left"},
         "collection_name": "rag_merge_test", "top_k": 2, "max_chunks_per_page": 2,
         "hybrid_candidates": 10, "use_bm25": False, "use_reranker": False,
         "merge_min_small": 3, "paths": {"vectorstore": str(tmp_path)}}
    emb = Embedder(c)
    build_index(STAT_PAGE_CHUNKS, c, embedder=emb)
    llm = MockLLM("ok")
    answer("Where is Big Boss located?", cfg=c, llm=llm, embedder=emb)
    assert "Location: Bionis' Leg" in llm.last_prompt   # the retrieved infobox
    assert "Break 100% Res" in llm.last_prompt           # sibling resistances pulled in by merge
    assert "Gogol Horn" in llm.last_prompt               # sibling drops pulled in by merge


def test_answer_game_filter_restricts_sources(cfg, embedder, indexed):
    llm = MockLLM("ok")
    res = answer("Who is the protagonist?", cfg=cfg, llm=llm, game_filter="XC2", embedder=embedder)
    assert [s["url"] for s in res["sources"]] == ["https://w/Rex"]


def test_answer_sources_are_deduped(cfg, embedder, indexed):
    llm = MockLLM("ok")
    res = answer("Infinity Blade", cfg=cfg, llm=llm, embedder=embedder)
    urls = [s["url"] for s in res["sources"]]
    assert len(urls) == len(set(urls))


def test_dedupe_sources_returns_rich_deduped_dicts():
    chunks = [
        {"url": "https://w/Rex", "title": "Rex (XC2)", "game": "XC2",
         "text": "[XC2] Rex > Introduction: Rex is the salvager protagonist of Xenoblade 2."},
        {"url": "https://w/Rex", "title": "Rex (XC2)", "game": "XC2", "text": "duplicate page"},
        {"url": "https://w/Pyra", "title": "Pyra (XC2)", "game": "XC2",
         "text": "[XC2] Pyra > infobox: The Aegis. Element: Fire."},
    ]
    src = _dedupe_sources(chunks)
    assert [s["url"] for s in src] == ["https://w/Rex", "https://w/Pyra"]   # deduped, order kept
    assert src[0]["title"] == "Rex (XC2)" and src[0]["game"] == "XC2"
    assert "salvager protagonist" in src[0]["snippet"]
    assert "[XC2]" not in src[0]["snippet"]                                  # breadcrumb stripped


def test_dedupe_sources_tiers_by_rank_position():
    # sources arrive best-first; bubble size tracks rank position (top -> 1.0, bottom -> 0.0)
    chunks = [
        {"url": "https://w/A", "title": "A", "game": "XC2", "text": "best", "_score": 10.0},
        {"url": "https://w/B", "title": "B", "game": "XC2", "text": "mid", "_score": 4.0},
        {"url": "https://w/C", "title": "C", "game": "XC2", "text": "low", "_score": 0.0},
    ]
    src = _dedupe_sources(chunks)
    assert [s["relevance"] for s in src] == [1.0, 0.5, 0.0]
    assert [s["tier"] for s in src] == ["high", "med", "low"]
    assert "_score" not in src[0]                                    # internal score dropped from payload


def test_dedupe_sources_tier_gradient_survives_score_clustering():
    # a dozen near-equal cross-encoder scores must NOT collapse into one size (the "all same" bug):
    # rank-based sizing still spreads them across high/med/low
    chunks = [{"url": f"https://w/{i}", "title": str(i), "game": "XC1", "text": "t",
               "_score": 9.0 - i * 0.01} for i in range(12)]
    src = _dedupe_sources(chunks)
    assert {s["tier"] for s in src} == {"high", "med", "low"}        # all three sizes present
    assert src[0]["tier"] == "high" and src[-1]["tier"] == "low"


def test_dedupe_sources_rank_fallback_without_scores():
    # reranker off / no _score -> relevance derived from rank position, still tiered + ordered
    chunks = [
        {"url": "https://w/A", "title": "A", "game": "", "text": "a"},
        {"url": "https://w/B", "title": "B", "game": "", "text": "b"},
        {"url": "https://w/C", "title": "C", "game": "", "text": "c"},
    ]
    src = _dedupe_sources(chunks)
    assert src[0]["tier"] == "high" and src[-1]["tier"] == "low"
    assert src[0]["relevance"] >= src[1]["relevance"] >= src[2]["relevance"]


def test_dedupe_sources_single_source_is_high():
    src = _dedupe_sources([{"url": "https://w/A", "title": "A", "game": "", "text": "a", "_score": 3.0}])
    assert src[0]["relevance"] == 1.0 and src[0]["tier"] == "high"


# --- GeminiClient credential gate (no network) ---

def test_gemini_raises_without_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiClient({"gemini_model": "gemini-1.5-flash"})
    with pytest.raises(RuntimeError):
        client.generate("system", "prompt")


def test_gemini_default_model_is_not_retired():
    # The hardcoded fallback model must not be a retired id (1.5-flash is gone) — a config without
    # gemini_model should still get a usable current default.
    client = GeminiClient({})
    assert client.model != "gemini-1.5-flash"
    assert "gemini" in client.model


# --- bug #3: empty question must short-circuit (no retrieval, no LLM call) ---

def test_empty_question_short_circuits_without_llm():
    llm = MockLLM("should not be used")
    res = answer("   ", cfg=None, llm=llm)
    assert llm.last_prompt is None            # LLM never invoked
    assert res["sources"] == []
    assert res["answer"].strip()              # a friendly, non-empty message
    assert "question" in res["answer"].lower()


# --- bug #2: a blocked / empty LLM response must degrade gracefully, not crash or go blank ---

def test_llm_empty_response_falls_back_gracefully(cfg, embedder, indexed):
    llm = MockLLM("")                          # simulate a safety-blocked / empty Gemini response
    res = answer("How much power does Infinity Blade have?", cfg=cfg, llm=llm, embedder=embedder)
    assert res["answer"].strip()               # not blank — a helpful fallback
    assert "could not" in res["answer"].lower() or "couldn't" in res["answer"].lower()


def test_extract_text_returns_plain_text_when_present():
    assert _extract_text(SimpleNamespace(text="hello", candidates=[])) == "hello"


def test_extract_text_from_candidate_parts_when_text_missing():
    resp = SimpleNamespace(
        text=None,
        candidates=[SimpleNamespace(
            content=SimpleNamespace(parts=[SimpleNamespace(text="hi "), SimpleNamespace(text="there")])
        )],
    )
    assert _extract_text(resp) == "hi there"


def test_extract_text_handles_raising_text_property():
    class Blocked:
        @property
        def text(self):
            raise ValueError("response was blocked; no Part")
    assert _extract_text(Blocked()) == ""      # never raises, returns empty


def test_extract_text_handles_none_everywhere():
    assert _extract_text(SimpleNamespace(text=None, candidates=None)) == ""


# --- SP3: real token streaming ---

def test_mock_generate_stream_yields_multiple_pieces():
    llm = MockLLM("Rex is the salvager protagonist of Xenoblade Chronicles 2.")
    pieces = list(llm.generate_stream("sys", "prompt"))
    assert len(pieces) > 1                                  # genuinely streamed, not one shot
    assert "".join(pieces) == "Rex is the salvager protagonist of Xenoblade Chronicles 2."
    assert llm.last_prompt == "prompt"                      # records the call like generate()


def test_answer_stream_yields_text_then_sources(cfg, embedder, indexed):
    llm = MockLLM("Infinity Blade has 250 power.")
    events = list(answer_stream("How much power does Infinity Blade have?",
                                cfg=cfg, llm=llm, embedder=embedder))
    kinds = [k for k, _ in events]
    assert "text" in kinds
    assert kinds[-1] == "sources"                           # sources come last
    text = "".join(p for k, p in events if k == "text")
    assert text == "Infinity Blade has 250 power."
    sources = next(p for k, p in events if k == "sources")
    assert any(s["url"] == "https://w/Infinity_Blade" for s in sources)


def test_answer_stream_empty_question_short_circuits():
    events = list(answer_stream("   ", cfg=None))
    assert ("text", NO_QUESTION_MESSAGE) in events
    assert events[-1] == ("sources", [])


def test_answer_stream_surfaces_errors_as_an_event(cfg, embedder, indexed):
    class Boom:
        def generate_stream(self, system, prompt):
            raise RuntimeError("kaboom")
            yield  # make it a generator; never reached
    events = list(answer_stream("Infinity Blade power?", cfg=cfg, llm=Boom(), embedder=embedder))
    assert any(k == "error" for k, _ in events)             # error surfaced, not raised


def test_answer_stream_empty_model_response_falls_back(cfg, embedder, indexed):
    llm = MockLLM("")                                       # blocked/empty stream
    events = list(answer_stream("Infinity Blade power?", cfg=cfg, llm=llm, embedder=embedder))
    text = "".join(p for k, p in events if k == "text")
    assert text.strip()                                     # a fallback message, not blank


# --- SP7: multi-turn follow-up context ---

def test_build_prompt_includes_conversation_history():
    history = [{"question": "Who is Pyra?", "answer": "Pyra is the Aegis Blade in XC2."}]
    _, user = build_prompt("What is her element?", CHUNKS, history=history)
    assert "Who is Pyra?" in user
    assert "Pyra is the Aegis Blade in XC2." in user
    assert "What is her element?" in user
    # no history -> no conversation preamble
    _, plain = build_prompt("What is her element?", CHUNKS)
    assert "earlier in this conversation" not in plain.lower()


def test_build_prompt_threads_full_answer_not_truncated():
    """History carries the FULL prior answer (just Q/A text, no chunk data — cheap), so a follow-up
    can see details that used to fall past the old 500-char clip."""
    long_answer = "HEAD_MARK " + ("filler " * 120) + "TAIL_MARK"   # ~870 chars, > old 500 cap
    history = [{"question": "list everything", "answer": long_answer}]
    _, user = build_prompt("follow up", CHUNKS, history=history)
    assert "HEAD_MARK" in user and "TAIL_MARK" in user             # whole answer survives, untruncated


def test_build_prompt_windows_history_to_last_6_turns():
    """Only the most recent turns go into the prompt (sliding window) so a long session can't rot the
    context with stale, off-topic turns. The oldest turns beyond the window are dropped."""
    history = [{"question": f"Q{i}_MARK", "answer": f"A{i}_MARK"} for i in range(1, 9)]  # 8 turns
    _, user = build_prompt("current question", CHUNKS, history=history)
    for old in ("Q1_MARK", "A1_MARK", "Q2_MARK", "A2_MARK"):
        assert old not in user, f"{old} should have fallen out of the window"
    for kept in ("Q3_MARK", "A3_MARK", "Q8_MARK", "A8_MARK"):
        assert kept in user, f"{kept} should be within the last-6 window"


def test_retrieval_query_expands_with_previous_question():
    history = [{"question": "Who is Pyra?", "answer": "The Aegis."}]
    q = _retrieval_query("What is her element?", history)
    assert "Pyra" in q and "her element" in q               # prior question helps resolve "her"
    assert _retrieval_query("standalone question", None) == "standalone question"


def test_answer_stream_threads_history_into_prompt(cfg, embedder, indexed):
    llm = MockLLM("ok")
    history = [{"question": "Who is Rex?", "answer": "Rex is the salvager protagonist."}]
    list(answer_stream("What about his weapon?", cfg=cfg, llm=llm, embedder=embedder, history=history))
    assert "Rex is the salvager protagonist." in llm.last_prompt
