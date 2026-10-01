"""Tests for retrieval + generation. Mock LLM, no network."""

from types import SimpleNamespace

import pytest

from xeno_rag import rag as rag_mod
from xeno_rag.answerability import Verdict
from xeno_rag.embed_index import build_index
from xeno_rag.rag import (
    FORMAT_LINES,
    NO_QUESTION_MESSAGE,
    NOT_COVERED_MESSAGE,
    OFF_TOPIC_MESSAGE,
    SYSTEM_PROMPT,
    GeminiClient,
    MockLLM,
    _dedupe_sources,
    _extract_text,
    _retrieval_query,
    answer,
    answer_stream,
    build_prompt,
)
from xeno_rag.router import Route

from .fakes import HashingEmbedder

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
    return HashingEmbedder(cfg)


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


def test_build_prompt_includes_game_scope_when_filtered():
    _, user = build_prompt("Who is the protagonist?", CHUNKS, game_filter="XC2")
    assert "Xenoblade Chronicles 2" in user or "XC2" in user
    # No filter -> no scope line (asking across the whole series).
    _, plain = build_prompt("Who is the protagonist?", CHUNKS)
    assert "focused on" not in plain.lower()


def test_build_prompt_numbers_sources_for_inline_citations():
    """Each context block is prefixed with the bracketed number of its source page ([1]..[n],
    first-seen url order, the exact ordering _dedupe_sources gives the SSE sources payload), and
    the prompt instructs the model to cite claims with those markers."""
    _, user = build_prompt("q", CHUNKS)
    assert "[1] [Infinity Blade (XC3) (Noah) (XC3)]" in user
    assert "[2] [Rex (XC2) (XC2)]" in user
    # the numbering matches the sources payload order 1:1
    srcs = _dedupe_sources(list(CHUNKS))
    assert [s["url"] for s in srcs] == ["https://w/Infinity_Blade", "https://w/Rex"]
    # the citation instruction names the marker style and the valid range
    assert "Cite inline" in user
    assert "[1]" in user and "[2]" in user
    assert "Never invent a number" in user


def test_build_prompt_repeats_number_for_same_page_chunks():
    """Two chunks of the SAME page share one citation number (sources are deduped per page), and a
    later distinct page continues the sequence."""
    chunks = [CHUNKS[0],
              {**CHUNKS[0], "chunk_id": "1-1",
               "text": "[XC3] Infinity Blade > acquisition: Noah's Talent Art."},
              CHUNKS[1]]
    _, user = build_prompt("q", chunks)
    assert user.count("[1] [Infinity Blade") == 2       # both sibling chunks carry [1]
    assert "[2] [Rex (XC2) (XC2)]" in user
    assert "[1]–[2]" in user                             # marker range covers 2 distinct sources


def test_build_prompt_no_citation_instruction_without_sources():
    """No retrieved context -> no numbered blocks and no dangling citation instruction."""
    _, user = build_prompt("q", [])
    assert "(no context retrieved)" in user
    assert "Cite inline" not in user


# --- SP2: format hint (Jev's routed table/list/prose choice) ---

def test_build_prompt_format_line_present_and_placed_before_question():
    for fmt in ("table", "list", "prose"):
        _, user = build_prompt("q", CHUNKS, answer_format=fmt)
        assert FORMAT_LINES[fmt] in user
        # verbatim line + "\n\n" inserted immediately before "Question:"
        assert user.rstrip("\n").endswith(f"{FORMAT_LINES[fmt]}\n\nQuestion: q")


def test_build_prompt_format_line_absent_for_none_or_unknown():
    _, plain = build_prompt("q", CHUNKS)
    _, unknown = build_prompt("q", CHUNKS, answer_format="paragraph")
    for user in (plain, unknown):
        for line in FORMAT_LINES.values():
            assert line not in user


def test_build_prompt_no_format_arg_is_byte_identical_to_none():
    _, a = build_prompt("q", CHUNKS)
    _, b = build_prompt("q", CHUNKS, answer_format=None)
    assert a == b


def test_system_prompt_instructs_bracketed_citations():
    s = SYSTEM_PROMPT.lower()
    assert "[1]" in SYSTEM_PROMPT and "[2]" in SYSTEM_PROMPT
    assert "cite" in s
    # the no-URLs rule is retained (the UI renders the linked source cards itself)
    assert "url" in s


def test_system_prompt_requests_markdown_tables_and_structure():
    s = SYSTEM_PROMPT.lower()
    assert "table" in s                    # nudge to tabulate multi-stat comparisons
    assert "markdown" in s                  # answer is rendered as markdown
    assert "do not invent" in s or "only" in s   # grounding guard retained


def test_build_prompt_invites_grounded_reasoning():
    # The model must be allowed to reason over the context (count, total, take the max, infer a
    # range) and give a best-effort partial answer, not flatly refuse when no single chunk states
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
    full profile into the prompt, so sibling facts the query would never rank first (resistances,
    drops) still reach the model. No re-embed; retrieval is unchanged."""
    c = {"embed_model": "Qwen/Qwen3-Embedding-0.6B", "embed_device": "cpu",
         "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
         "embed_tokenizer_kwargs": {"padding_side": "left"},
         "collection_name": "rag_merge_test", "top_k": 2, "max_chunks_per_page": 2,
         "hybrid_candidates": 10, "use_bm25": False, "use_reranker": False,
         "merge_min_small": 3, "paths": {"vectorstore": str(tmp_path)}}
    emb = HashingEmbedder(c)
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
    # The hardcoded fallback model must not be a retired id (1.5-flash is gone): a config without
    # gemini_model should still get a usable current default.
    client = GeminiClient({})
    assert client.model != "gemini-1.5-flash"
    assert "gemini" in client.model
    assert client.model == "gemini-3.8-flash"


# --- answer-tier routing ---

TIER_CFG = {
    "top_k": 3, "gemini_model": "gemini-3.8-flash",
    "answer_tiers": {"fast": {"model": "gemini-3.5-flash-lite", "top_k": 20},
                     "thinking": {"model": "gemini-3.8-flash", "top_k": 40},
                     "scholar": {"model": "gemini-3.8-flash", "thinking_level": "high", "top_k": 96}},
}


@pytest.fixture
def spy_retrieval(monkeypatch):
    """Capture the cfg (and query_embedding) retrieval runs with; skip the real index AND the real
    Qwen embedder entirely (``_embed_query`` is what ``answer``/``answer_stream`` submit to the
    concurrent executor, so patching it here is what keeps these tests from loading the real model)."""
    seen = {}

    def fake_retrieve(text, cfg, **kw):
        seen["cfg"] = cfg
        seen["query_embedding"] = kw.get("query_embedding")
        return [dict(CHUNKS[0], _score=1.0)]

    def fake_embed_query(text, cfg, embedder=None):
        return ["fake-vector-for", text]

    monkeypatch.setattr(rag_mod, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag_mod, "merge_fragmented_pages", lambda chunks, cfg: chunks)
    monkeypatch.setattr(rag_mod, "_embed_query", fake_embed_query)
    return seen


def test_answer_tier_override_skips_router(monkeypatch, spy_retrieval):
    """A forced tier still calls route() (now that route() itself owns the forced-tier short-circuit,
    per SP1): route() is passed forced_tier and returns an override without a network call, but the
    caller must not skip calling it, or a key'd deployment would never read topic/format for a forced
    tier."""
    seen = {}

    def fake_route(question, cfg, history=None, game=None, http_post=None, forced_tier=None):
        seen["forced_tier"] = forced_tier
        return Route(forced_tier, "override")
    monkeypatch.setattr(rag_mod, "route", fake_route)
    res = answer("q", cfg=TIER_CFG, llm=MockLLM("ok"), tier="fast")
    assert res["tier"] == "fast"
    assert seen["forced_tier"] == "fast"
    assert spy_retrieval["cfg"]["top_k"] == 20
    assert spy_retrieval["cfg"]["gemini_model"] == "gemini-3.5-flash-lite"


def test_answer_auto_routes_and_applies_tier(monkeypatch, spy_retrieval):
    seen = {}

    def fake_route(question, cfg, history=None, game=None, http_post=None, forced_tier=None):
        seen.update(question=question, game=game)
        return Route("scholar", "jev", 0.9)
    monkeypatch.setattr(rag_mod, "route", fake_route)
    res = answer("q", cfg=TIER_CFG, llm=MockLLM("ok"), game_filter="XC3")
    assert res["tier"] == "scholar"
    assert spy_retrieval["cfg"]["top_k"] == 96 and spy_retrieval["cfg"]["thinking_level"] == "high"
    assert seen["game"] == "Xenoblade Chronicles 3"       # display name, not the code


def test_answer_unknown_tier_override_is_auto_routed(monkeypatch, spy_retrieval):
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("thinking", "fallback"))
    res = answer("q", cfg=TIER_CFG, llm=MockLLM("ok"), tier="bogus")
    assert res["tier"] == "thinking"


def test_answer_stream_emits_tier_event_first(monkeypatch, spy_retrieval):
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.8))
    events = list(answer_stream("q", cfg=TIER_CFG, llm=MockLLM("hello world")))
    assert events[0] == ("tier", {"tier": "fast", "source": "jev"})
    kinds = [k for k, _ in events]
    assert "text" in kinds and kinds[-1] == "sources"


def test_answer_stream_override_tier_event(monkeypatch, spy_retrieval):
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("scholar", "override"))
    events = list(answer_stream("q", cfg=TIER_CFG, llm=MockLLM("x"), tier="scholar"))
    assert events[0] == ("tier", {"tier": "scholar", "source": "override"})


# --- SP2: off-topic short-circuit, concurrent embed, format hint passthrough ---

OFF_TOPIC_CFG = {**TIER_CFG, "router": {"off_topic_gate": True}}


def _boom_retrieve(*a, **kw):
    raise AssertionError("retrieve must not run when the off-topic gate fires")


def test_answer_off_topic_short_circuits(monkeypatch, spy_retrieval):
    monkeypatch.setattr(rag_mod, "route",
                        lambda *a, **kw: Route("fast", "jev", 0.9, topic="off_topic"))
    monkeypatch.setattr(rag_mod, "retrieve", _boom_retrieve)
    llm = MockLLM("should not be used")
    res = answer("play chess with me", cfg=OFF_TOPIC_CFG, llm=llm)
    assert res == {"answer": OFF_TOPIC_MESSAGE, "sources": [], "tier": None}
    assert llm.last_prompt is None                 # LLM never invoked


def test_answer_off_topic_gate_disabled_answers_normally(monkeypatch, spy_retrieval):
    # TIER_CFG carries no router.off_topic_gate -> defaults off, so an off_topic routing decision is
    # ignored and the question is answered as usual.
    monkeypatch.setattr(rag_mod, "route",
                        lambda *a, **kw: Route("fast", "jev", 0.9, topic="off_topic"))
    res = answer("play chess with me", cfg=TIER_CFG, llm=MockLLM("ok"))
    assert res["tier"] == "fast" and res["answer"] == "ok"


def test_answer_topic_none_answers_normally_even_with_gate_on(monkeypatch, spy_retrieval):
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9, topic=None))
    res = answer("q", cfg=OFF_TOPIC_CFG, llm=MockLLM("ok"))
    assert res["tier"] == "fast" and res["answer"] == "ok"


def test_answer_stream_off_topic_emits_no_tier_event(monkeypatch, spy_retrieval):
    monkeypatch.setattr(rag_mod, "route",
                        lambda *a, **kw: Route("fast", "jev", 0.9, topic="off_topic"))
    monkeypatch.setattr(rag_mod, "retrieve", _boom_retrieve)
    events = list(answer_stream("play chess with me", cfg=OFF_TOPIC_CFG, llm=MockLLM("x")))
    assert events == [("text", OFF_TOPIC_MESSAGE), ("sources", [])]
    assert not any(k == "tier" for k, _ in events)


def test_answer_embeds_concurrently_with_routing(monkeypatch, spy_retrieval):
    """The query embedding is submitted to _EXECUTOR before route() runs, so Jev's HTTP round-trip
    and the embed overlap. Proven by a fake route() that blocks (bounded, 2s) on an Event the fake
    embed sets: route() only observes it set if the embed had already started in the background."""
    import threading
    started = threading.Event()

    def fake_embed_query(text, cfg, embedder=None):
        started.set()
        return ["vec"]
    monkeypatch.setattr(rag_mod, "_embed_query", fake_embed_query)

    seen = {}

    def fake_route(question, cfg, history=None, game=None, http_post=None, forced_tier=None):
        seen["embed_started"] = started.wait(timeout=2)
        return Route("fast", "jev", 0.9)
    monkeypatch.setattr(rag_mod, "route", fake_route)

    answer("q", cfg=TIER_CFG, llm=MockLLM("ok"))
    assert seen["embed_started"] is True


def test_answer_retrieve_receives_the_embedded_vector(monkeypatch, spy_retrieval):
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    answer("q", cfg=TIER_CFG, llm=MockLLM("ok"))
    assert spy_retrieval["query_embedding"] == ["fake-vector-for", "q"]


def test_answer_passes_routed_format_to_prompt(monkeypatch, spy_retrieval):
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9, format="table"))
    llm = MockLLM("ok")
    answer("q", cfg=TIER_CFG, llm=llm)
    assert FORMAT_LINES["table"] in llm.last_prompt


def test_answer_no_format_leaves_prompt_without_format_line(monkeypatch, spy_retrieval):
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9, format=None))
    llm = MockLLM("ok")
    answer("q", cfg=TIER_CFG, llm=llm)
    for line in FORMAT_LINES.values():
        assert line not in llm.last_prompt


def test_answer_embedding_future_failure_propagates(monkeypatch, spy_retrieval):
    def boom_embed(text, cfg, embedder=None):
        raise RuntimeError("embed exploded")
    monkeypatch.setattr(rag_mod, "_embed_query", boom_embed)
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    with pytest.raises(RuntimeError):
        answer("q", cfg=TIER_CFG, llm=MockLLM("ok"))


def test_answer_stream_embedding_future_failure_is_an_error_event(monkeypatch, spy_retrieval):
    def boom_embed(text, cfg, embedder=None):
        raise RuntimeError("embed exploded")
    monkeypatch.setattr(rag_mod, "_embed_query", boom_embed)
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    events = list(answer_stream("q", cfg=TIER_CFG, llm=MockLLM("ok")))
    assert events[-1][0] == "error"


# --- Minor 3: reported tier must match the tier apply_tier actually applied ---

def test_answer_reports_the_applied_tier_not_the_routed_one(monkeypatch, spy_retrieval):
    # answer_tiers has no "scholar" entry; apply_tier falls back to "thinking" internally, so the
    # reported tier (and the cfg retrieval ran with) must say "thinking", not the routed "scholar".
    cfg = {k: v for k, v in TIER_CFG.items() if k != "answer_tiers"}
    cfg["answer_tiers"] = {k: v for k, v in TIER_CFG["answer_tiers"].items() if k != "scholar"}
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("scholar", "jev", 0.9))
    res = answer("q", cfg=cfg, llm=MockLLM("ok"))
    assert res["tier"] == "thinking"
    assert spy_retrieval["cfg"]["top_k"] == 40                # the "thinking" entry's depth


def test_answer_stream_reports_the_applied_tier_not_the_routed_one(monkeypatch, spy_retrieval):
    cfg = {k: v for k, v in TIER_CFG.items() if k != "answer_tiers"}
    cfg["answer_tiers"] = {k: v for k, v in TIER_CFG["answer_tiers"].items() if k != "scholar"}
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("scholar", "jev", 0.9))
    events = list(answer_stream("q", cfg=cfg, llm=MockLLM("x")))
    assert events[0] == ("tier", {"tier": "thinking", "source": "jev"})


def test_gemini_thinking_level_reaches_generation_config():
    from google.genai import types
    cfg = GeminiClient({"gemini_model": "gemini-3.8-flash", "thinking_level": "high"})._gen_config(types, "sys")
    assert cfg.system_instruction == "sys"
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.HIGH


def test_gemini_no_thinking_level_leaves_config_default():
    from google.genai import types
    cfg = GeminiClient({"gemini_model": "gemini-3.5-flash-lite"})._gen_config(types, "sys")
    assert cfg.thinking_config is None


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
    assert res["answer"].strip()               # not blank: a helpful fallback
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
    """History carries the FULL prior answer (just Q/A text, no chunk data, cheap), so a follow-up
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


# --- SP3: answerability check, escalation, decline ---

ANSWERABILITY_TIER_CFG = {
    "top_k": 3, "gemini_model": "gemini-3.8-flash",
    "answer_tiers": {
        "fast": {"model": "gemini-3.5-flash-lite", "top_k": 20, "rerank_candidates": 50},
        "thinking": {"model": "gemini-3.8-flash", "top_k": 40, "rerank_candidates": 100},
        "scholar": {"model": "gemini-3.8-flash", "thinking_level": "high", "top_k": 96,
                   "rerank_candidates": 224},
    },
    "router": {"provider": "jev", "answerability_check": True, "decline_confidence": 0.8,
              "answerability_passages": 8},
}

# Fix round 1, item 4: no "scholar" entry at all -> apply_tier(base_cfg, "scholar") can only fall back
# to another tier's entry, so escalating would retrieve no deeper than the current tier already did.
NO_SCHOLAR_TIER_CFG = {
    **ANSWERABILITY_TIER_CFG,
    "answer_tiers": {k: v for k, v in ANSWERABILITY_TIER_CFG["answer_tiers"].items() if k != "scholar"},
}

JEV_KEY = "test-key-not-real"


@pytest.fixture
def with_jev_key(monkeypatch):
    """The answerability check now also requires a Jev key to be set (fix round 1, item 1: spec §4
    says "a key is set"), so any test that wants ``_ground`` to actually invoke
    ``answerability.check`` needs this alongside ``router.provider: "jev"`` in its cfg."""
    monkeypatch.setenv("TYPESAFE_API_KEY", JEV_KEY)


def _check_sequence(*verdicts):
    """A stand-in for ``answerability.check`` that returns each ``verdict`` in order, one per call --
    lets a test script "first check says not_covered, second (post-escalation) check says answered"
    without a real Jev call."""
    it = iter(verdicts)

    def fake_check(question, chunks, cfg, http_post=None, history=None):
        return next(it)
    return fake_check


def _counting_check(verdict):
    """A stand-in for ``answerability.check`` that always returns the same ``verdict`` but records how
    many times (and with which cfg) it was called -- for the two-checks-two-retrieves production
    decline path, where both the initial and the escalated check return ``not_covered``."""
    calls = []

    def fake_check(question, chunks, cfg, http_post=None, history=None):
        calls.append(cfg)
        return verdict
    fake_check.calls = calls
    return fake_check


def _recording_gemini_client(calls):
    """A ``GeminiClient`` stand-in that records the cfg it's constructed with (so a test can assert
    which tier's model/thinking_level actually reached the LLM) without importing google-genai, and
    without recording anything at all when the LLM is never constructed (the decline path)."""
    class _Recorder:
        def __init__(self, cfg):
            calls.append(cfg)
            self.cfg = cfg

        def generate(self, system, prompt):
            return "recorded answer"

        def generate_stream(self, system, prompt):
            yield "recorded answer"
    return _Recorder


@pytest.fixture
def track_retrieves(monkeypatch, spy_retrieval):
    """Like ``spy_retrieval`` but records every retrieve() call (not just the last), so escalation's
    second retrieve can be inspected -- and returns _something_ retrievable so ``_ground`` always has
    chunks to check/decline over."""
    calls = []

    def fake_retrieve(text, cfg, **kw):
        calls.append({"cfg": cfg, "query_embedding": kw.get("query_embedding")})
        return [dict(CHUNKS[0], _score=1.0)]
    monkeypatch.setattr(rag_mod, "retrieve", fake_retrieve)
    return calls


@pytest.fixture
def track_empty_retrieves(monkeypatch, spy_retrieval):
    """Like ``track_retrieves``, but simulates a genuinely empty retrieval (no chunks hit at all) --
    the scenario ``answerability.check()`` short-circuits to ``Verdict("not_covered", 1.0)`` for, with
    NO Jev call, regardless of whether a key is even set."""
    calls = []

    def fake_retrieve(text, cfg, **kw):
        calls.append({"cfg": cfg, "query_embedding": kw.get("query_embedding")})
        return []
    monkeypatch.setattr(rag_mod, "retrieve", fake_retrieve)
    return calls


def test_ground_answered_verdict_no_escalation(monkeypatch, with_jev_key, track_retrieves):
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    monkeypatch.setattr(rag_mod.answerability, "check", _check_sequence(Verdict("answered", 0.9)))
    res = answer("q", cfg=ANSWERABILITY_TIER_CFG, llm=MockLLM("ok"))
    assert res["answer"] == "ok"
    assert res["tier"] == "fast"
    assert len(track_retrieves) == 1


def test_ground_not_covered_at_fast_escalates_and_builds_llm_with_scholar_model(
        monkeypatch, with_jev_key, track_retrieves):
    """Covers fix round 1, item 3 (plan: "LLM built with scholar model"): passing llm=None so
    answer() constructs GeminiClient itself, with a recorder standing in for it, proves the ESCALATED
    tier's model/thinking_level -- not the original fast tier's -- is what actually reaches the LLM."""
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    monkeypatch.setattr(rag_mod.answerability, "check",
                        _check_sequence(Verdict("not_covered", 0.9), Verdict("answered", 0.9)))
    llm_calls = []
    monkeypatch.setattr(rag_mod, "GeminiClient", _recording_gemini_client(llm_calls))
    res = answer("q", cfg=ANSWERABILITY_TIER_CFG, llm=None)
    assert res["tier"] == "scholar"
    assert res["answer"] == "recorded answer"
    assert len(track_retrieves) == 2                                        # initial + escalated
    assert track_retrieves[1]["cfg"]["rerank_candidates"] == 224            # scholar's depth, not fast's
    # the escalation re-retrieves with the SAME already-embedded query vector, not a fresh embed
    assert track_retrieves[0]["query_embedding"] == track_retrieves[1]["query_embedding"]
    assert len(llm_calls) == 1
    assert llm_calls[0]["gemini_model"] == "gemini-3.8-flash"
    assert llm_calls[0]["thinking_level"] == "high"


def test_answer_stream_not_covered_at_fast_emits_escalated_tier_event(monkeypatch, with_jev_key, track_retrieves):
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    monkeypatch.setattr(rag_mod.answerability, "check",
                        _check_sequence(Verdict("not_covered", 0.9), Verdict("answered", 0.9)))
    events = list(answer_stream("q", cfg=ANSWERABILITY_TIER_CFG, llm=MockLLM("hi")))
    kinds = [k for k, _ in events]
    # Fix round 1, item 6: precise event order, not the old "text in events or any text at all"
    # near-tautology -- first the initial routing tier, then the escalated tier, then text deltas,
    # then sources last.
    assert events[0] == ("tier", {"tier": "fast", "source": "jev"})
    assert events[1] == ("tier", {"tier": "scholar", "source": "escalated"})
    assert kinds[2:-1] and all(k == "text" for k in kinds[2:-1])
    assert kinds[-1] == "sources"


def test_ground_not_covered_at_scholar_declines_with_no_llm_call(monkeypatch, with_jev_key, track_retrieves):
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("scholar", "jev", 0.9))
    monkeypatch.setattr(rag_mod.answerability, "check", _check_sequence(Verdict("not_covered", 0.9)))
    llm_calls = []
    monkeypatch.setattr(rag_mod, "GeminiClient", _recording_gemini_client(llm_calls))
    res = answer("q", cfg=ANSWERABILITY_TIER_CFG, llm=None)
    assert res["answer"] == NOT_COVERED_MESSAGE
    assert res["tier"] == "scholar"
    assert res["sources"] and res["sources"][0]["url"] == CHUNKS[0]["url"]   # closest matches shown
    assert llm_calls == []                                                  # GeminiClient never built
    assert len(track_retrieves) == 1                                        # already scholar: one retrieve


def test_answer_stream_not_covered_at_scholar_declines_with_no_llm_call(monkeypatch, with_jev_key, track_retrieves):
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("scholar", "jev", 0.9))
    monkeypatch.setattr(rag_mod.answerability, "check", _check_sequence(Verdict("not_covered", 0.9)))
    llm_calls = []
    monkeypatch.setattr(rag_mod, "GeminiClient", _recording_gemini_client(llm_calls))
    events = list(answer_stream("q", cfg=ANSWERABILITY_TIER_CFG, llm=None))
    assert events[0] == ("tier", {"tier": "scholar", "source": "jev"})
    assert sum(1 for k, _ in events if k == "tier") == 1                    # no "escalated" second event
    assert ("text", NOT_COVERED_MESSAGE) in events
    assert events[-1][0] == "sources" and events[-1][1]
    assert llm_calls == []                                                  # GeminiClient never built


def test_answer_full_decline_path_two_checks_two_retrieves_no_llm(monkeypatch, with_jev_key, track_retrieves):
    """Fix round 1, item 5: the main production decline path -- fast routes, the first check says
    not_covered, escalates to scholar, the SECOND check also says not_covered, so it declines. Two
    checks, two retrieves, no LLM call."""
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    fake_check = _counting_check(Verdict("not_covered", 0.9))
    monkeypatch.setattr(rag_mod.answerability, "check", fake_check)
    llm_calls = []
    monkeypatch.setattr(rag_mod, "GeminiClient", _recording_gemini_client(llm_calls))
    res = answer("q", cfg=ANSWERABILITY_TIER_CFG, llm=None)
    assert res["answer"] == NOT_COVERED_MESSAGE
    assert res["tier"] == "scholar"
    assert len(fake_check.calls) == 2                    # initial fast check + escalated scholar check
    assert len(track_retrieves) == 2                     # initial + escalated retrieve
    assert llm_calls == []                                # no LLM call


def test_answer_stream_full_decline_path_two_checks_two_retrieves_no_llm(monkeypatch, with_jev_key, track_retrieves):
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    fake_check = _counting_check(Verdict("not_covered", 0.9))
    monkeypatch.setattr(rag_mod.answerability, "check", fake_check)
    llm_calls = []
    monkeypatch.setattr(rag_mod, "GeminiClient", _recording_gemini_client(llm_calls))
    events = list(answer_stream("q", cfg=ANSWERABILITY_TIER_CFG, llm=None))
    assert len(fake_check.calls) == 2
    assert len(track_retrieves) == 2
    assert llm_calls == []
    kinds = [k for k, _ in events]
    assert events[0] == ("tier", {"tier": "fast", "source": "jev"})
    assert events[1] == ("tier", {"tier": "scholar", "source": "escalated"})
    assert events[2] == ("text", NOT_COVERED_MESSAGE)
    assert kinds.count("tier") == 2
    assert kinds[-1] == "sources"


def test_ground_skips_escalation_when_no_scholar_tier_is_configured(monkeypatch, with_jev_key, track_retrieves):
    """Fix round 1, item 4: apply_tier(base_cfg, "scholar") silently falls back to another tier's
    entry when answer_tiers has no "scholar" key at all, so escalating would just re-retrieve at the
    SAME depth. _ground must notice (apply_tier(...).get("answer_tier") != "scholar") and skip
    escalating, declining off the original (unescalated) verdict instead."""
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    monkeypatch.setattr(rag_mod.answerability, "check", _check_sequence(Verdict("not_covered", 0.9)))
    res = answer("q", cfg=NO_SCHOLAR_TIER_CFG, llm=MockLLM("should not be used"))
    assert res["answer"] == NOT_COVERED_MESSAGE
    assert res["tier"] == "fast"                        # never touched: nowhere deeper to escalate to
    assert len(track_retrieves) == 1                    # no second/escalated retrieve


def test_ground_forced_tier_skips_the_check_entirely(monkeypatch, with_jev_key, track_retrieves):
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "override"))

    def boom_check(*a, **kw):
        raise AssertionError("the answerability check must not run for a forced (override) tier")
    monkeypatch.setattr(rag_mod.answerability, "check", boom_check)
    res = answer("q", cfg=ANSWERABILITY_TIER_CFG, llm=MockLLM("ok"), tier="fast")
    assert res["answer"] == "ok"
    assert len(track_retrieves) == 1


def test_ground_check_disabled_by_config_skips_the_check(monkeypatch, with_jev_key, track_retrieves):
    """Fix round 2, item 3: isolate the ``answerability_on()`` flag itself, not merely the absence of
    a key or provider. ANSWERABILITY_TIER_CFG (what every other ``_ground`` test uses) with
    ``answerability_check`` explicitly turned off, plus a real Jev key, proves the flag alone -- not
    a missing key/provider -- is what's gating the check here."""
    def boom_check(*a, **kw):
        raise AssertionError("the answerability check must not run when router.answerability_check is off")
    monkeypatch.setattr(rag_mod.answerability, "check", boom_check)
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    cfg = {**ANSWERABILITY_TIER_CFG,
          "router": {**ANSWERABILITY_TIER_CFG["router"], "answerability_check": False}}
    res = answer("q", cfg=cfg, llm=MockLLM("ok"))
    assert res["answer"] == "ok"


def test_ground_skips_check_without_a_key_even_on_empty_retrieval(monkeypatch, track_empty_retrieves):
    """Fix round 1, item 1 (regression): answerability.check() trivially returns
    Verdict("not_covered", 1.0) for an empty retrieval with NO Jev call at all. Without gating
    _ground's check on jev_available(base_cfg), a deployment with the check enabled but no
    TYPESAFE_API_KEY would escalate and decline on every empty-retrieval question even though Jev was
    never actually reachable -- spec §4 requires "a key is set" for the check to run at all."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))

    def boom_check(*a, **kw):
        raise AssertionError("the answerability check must not run without a Jev key")
    monkeypatch.setattr(rag_mod.answerability, "check", boom_check)
    res = answer("q", cfg=ANSWERABILITY_TIER_CFG, llm=MockLLM("ok"))
    assert res["answer"] == "ok"                        # not declined
    assert len(track_empty_retrieves) == 1               # no escalation retrieve either


def test_ground_skips_check_when_routing_itself_failed(monkeypatch, with_jev_key, track_retrieves):
    """Fix round 1, item 7: route() falling all the way back with NO confidence at all
    (source="fallback", confidence=None) means the Jev call itself returned no answers whatsoever --
    Jev is unreachable right now, so paying the answerability check's own timeout against an
    already-down provider would waste the request budget for nothing. A "fallback" with a REAL
    confidence (Jev answered, just with a low-confidence tier) is a different case and still checks --
    see ``test_ground_checks_when_routing_fell_back_with_a_real_confidence`` below, which is the case
    that actually exercises source="fallback" with a non-None confidence (the tests above this one all
    use source="jev")."""
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("thinking", "fallback", None))

    def boom_check(*a, **kw):
        raise AssertionError("the answerability check must not run when routing itself already failed")
    monkeypatch.setattr(rag_mod.answerability, "check", boom_check)
    res = answer("q", cfg=ANSWERABILITY_TIER_CFG, llm=MockLLM("ok"))
    assert res["answer"] == "ok"
    assert len(track_retrieves) == 1


def test_ground_verdict_none_proceeds_normally(monkeypatch, with_jev_key, track_retrieves):
    """A check that couldn't run (malformed reply, etc.) returns Verdict(None); _ground must treat
    that as "proceed", not as covered/not-covered, and never escalate or decline on it."""
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    monkeypatch.setattr(rag_mod.answerability, "check", _check_sequence(Verdict(None)))
    res = answer("q", cfg=ANSWERABILITY_TIER_CFG, llm=MockLLM("ok"))
    assert res["answer"] == "ok"
    assert res["tier"] == "fast"
    assert len(track_retrieves) == 1


def test_ground_checks_when_routing_fell_back_with_a_real_confidence(
        monkeypatch, with_jev_key, track_retrieves):
    """Fix round 2, item 3: a 'fallback' Route with a REAL (non-None) confidence means Jev answered
    successfully, just with a low-confidence tier choice -- unlike a routing_failed fallback (source
    'fallback', confidence None), this is NOT 'Jev is unreachable', so the check still runs. Proven by
    the first (not_covered) verdict actually triggering an escalation retrieve -- observable only if
    the check genuinely executed."""
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("thinking", "fallback", 0.3))
    monkeypatch.setattr(rag_mod.answerability, "check",
                        _check_sequence(Verdict("not_covered", 0.9), Verdict("answered", 0.9)))
    res = answer("q", cfg=ANSWERABILITY_TIER_CFG, llm=MockLLM("ok"))
    assert res["tier"] == "scholar"
    assert len(track_retrieves) == 2                    # initial + escalated: the check ran twice


def test_ground_escalated_check_returning_verdict_none_proceeds_without_declining(
        monkeypatch, with_jev_key, track_retrieves):
    """The SECOND (post-escalation) check can itself fail (malformed reply, timeout) and come back
    Verdict(None); that must be treated as 'proceed', not as covered/not-covered -- generation runs
    at the escalated scholar tier/model, with no decline."""
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    monkeypatch.setattr(rag_mod.answerability, "check",
                        _check_sequence(Verdict("not_covered", 0.9), Verdict(None)))
    res = answer("q", cfg=ANSWERABILITY_TIER_CFG, llm=MockLLM("ok"))
    assert res["answer"] == "ok"
    assert res["tier"] == "scholar"
    assert len(track_retrieves) == 2


# --- Final-review fix 1: the answerability check must judge the SAME merged text Gemini sees ---

def test_ground_checks_the_merged_prompt_chunks_not_raw_chunks(monkeypatch, with_jev_key):
    """merge_fragmented_pages must run BEFORE the answerability check (not just before generation, as
    it did before this fix): checking one-line fragment scraps under-represents a merged stat page's
    real coverage and produces false not_covered declines. Sources still come from the raw chunks."""
    raw = [dict(CHUNKS[1], _score=1.0)]
    merged = [{"title": "Rex (XC2)", "game": "XC2", "url": raw[0]["url"],
              "text": "[XC2] Rex > combined: MERGED PROFILE BLOCK"}]
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    monkeypatch.setattr(rag_mod, "retrieve", lambda *a, **kw: raw)
    monkeypatch.setattr(rag_mod, "merge_fragmented_pages", lambda chunks, cfg: merged)
    monkeypatch.setattr(rag_mod, "_embed_query", lambda text, cfg, embedder=None: ["vec"])
    seen = {}

    def fake_check(question, chunks, cfg, http_post=None, history=None):
        seen["chunks"] = chunks
        return Verdict("answered", 0.9)
    monkeypatch.setattr(rag_mod.answerability, "check", fake_check)
    llm = MockLLM("ok")
    res = answer("q", cfg=ANSWERABILITY_TIER_CFG, llm=llm)
    assert seen["chunks"] == merged                          # the check saw the merged block
    assert "MERGED PROFILE BLOCK" in llm.last_prompt          # generation used the same merged text
    assert res["sources"][0]["url"] == raw[0]["url"]          # sources still built from raw chunks


def test_ground_reuses_merged_chunks_across_escalation(monkeypatch, with_jev_key, track_retrieves):
    """The escalated re-retrieve must be re-merged too, and both checks (initial + escalated) must
    see their own retrieval's merged chunks, not a stale first-pass merge."""
    merge_calls = []

    def fake_merge(chunks, cfg):
        merge_calls.append(cfg.get("answer_tier"))
        return [{**chunks[0], "text": f"merged-for-{cfg.get('answer_tier')}"}] if chunks else chunks
    monkeypatch.setattr(rag_mod, "merge_fragmented_pages", fake_merge)
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    seen = []

    def fake_check(question, chunks, cfg, http_post=None, history=None):
        seen.append(chunks[0]["text"] if chunks else None)
        return Verdict("not_covered", 0.9) if len(seen) == 1 else Verdict("answered", 0.9)
    monkeypatch.setattr(rag_mod.answerability, "check", fake_check)
    res = answer("q", cfg=ANSWERABILITY_TIER_CFG, llm=MockLLM("ok"))
    assert res["tier"] == "scholar"
    assert seen == ["merged-for-fast", "merged-for-scholar"]  # each check saw its own retrieval's merge


# --- Final-review fix 2: follow-up context (previous_question) reaches the answerability check ---

def test_ground_threads_history_into_the_answerability_check(monkeypatch, with_jev_key, track_retrieves):
    """A terse follow-up ("what is her element?") has no antecedent of its own; the check needs the
    previous question exactly as routing does, or it judges coverage blind."""
    monkeypatch.setattr(rag_mod, "route", lambda *a, **kw: Route("fast", "jev", 0.9))
    seen = {}

    def fake_check(question, chunks, cfg, http_post=None, history=None):
        seen["history"] = history
        return Verdict("answered", 0.9)
    monkeypatch.setattr(rag_mod.answerability, "check", fake_check)
    history = [{"question": "Who is Nia in Xenoblade Chronicles 2?", "answer": "She is a Gormotti."}]
    res = answer("what is her element?", cfg=ANSWERABILITY_TIER_CFG, llm=MockLLM("ok"), history=history)
    assert seen["history"] == history
    assert res["answer"] == "ok"
