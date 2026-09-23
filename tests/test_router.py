"""Tests for Jev answer-tier routing. No network: the HTTP call is injected."""

import logging

import pytest

from xeno_rag.router import TIERS, Route, apply_tier, build_request, fallback_tier, route

KEY = "test-key-not-real"

TIER_CFG = {
    "top_k": 16, "max_chunks_per_page": 4, "hybrid_candidates": 60, "rerank_candidates": 50,
    "gemini_model": "gemini-3.8-flash",
    "answer_tiers": {
        "fast": {"model": "gemini-3.5-flash-lite", "top_k": 20, "max_chunks_per_page": 5},
        "thinking": {"model": "gemini-3.8-flash", "top_k": 40, "max_chunks_per_page": 6,
                     "hybrid_candidates": 120, "rerank_candidates": 100},
        "scholar": {"model": "gemini-3.8-flash", "thinking_level": "high", "top_k": 96,
                    "max_chunks_per_page": 10, "hybrid_candidates": 256, "rerank_candidates": 224},
    },
    "router": {"provider": "jev", "model": "jev-latest", "url": "https://example.invalid/v1/systemone",
               "fallback_tier": "thinking", "min_confidence": 0.5, "timeout_seconds": 2},
}


def _post_returning(choice, confidence, calls=None):
    def post(url, *, json, headers, timeout):
        if calls is not None:
            calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return {"model": "jev-latest",
                "answers": {"tier": {"type": "choice", "choice": choice, "confidence": confidence,
                                     "probabilities": {choice: confidence}}}}
    return post


def _post_raising(exc):
    def post(url, *, json, headers, timeout):
        raise exc
    return post


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)


@pytest.mark.parametrize("tier", TIERS)
def test_route_returns_jev_choice(with_key, tier):
    r = route("q", TIER_CFG, http_post=_post_returning(tier, 0.9))
    assert r == Route(tier, "jev", 0.9)


def test_route_low_confidence_falls_back(with_key):
    r = route("q", TIER_CFG, http_post=_post_returning("fast", 0.3))
    assert r.tier == "thinking" and r.source == "fallback" and r.confidence == 0.3


def test_route_unknown_choice_falls_back(with_key):
    r = route("q", TIER_CFG, http_post=_post_returning("ultra", 0.99))
    assert r.tier == "thinking" and r.source == "fallback"


class _StatusError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.response = type("R", (), {"status_code": code})()


@pytest.mark.parametrize("exc", [TimeoutError("slow"), ConnectionError("down"), ValueError("bad json"),
                                 _StatusError(401), _StatusError(422), _StatusError(429), _StatusError(529)])
def test_route_errors_fall_back_without_raising(with_key, exc):
    r = route("q", TIER_CFG, http_post=_post_raising(exc))
    assert r == Route("thinking", "fallback")


def test_route_malformed_response_falls_back(with_key):
    r = route("q", TIER_CFG, http_post=lambda url, **kw: {"answers": {}})
    assert r == Route("thinking", "fallback")


def test_route_without_key_makes_no_call(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    calls = []
    r = route("q", TIER_CFG, http_post=_post_returning("fast", 0.9, calls))
    assert r == Route("thinking", "fallback") and calls == []


def test_route_fixed_provider_makes_no_call(with_key):
    calls = []
    cfg = {**TIER_CFG, "router": {**TIER_CFG["router"], "provider": "fixed"}}
    r = route("q", cfg, http_post=_post_returning("fast", 0.9, calls))
    assert r == Route("thinking", "fallback") and calls == []


def test_route_without_router_block_is_offline(with_key):
    calls = []
    cfg = {k: v for k, v in TIER_CFG.items() if k != "router"}
    r = route("q", cfg, http_post=_post_returning("fast", 0.9, calls))
    assert r.source == "fallback" and calls == []


def test_route_sends_bearer_key_url_and_timeout(with_key):
    calls = []
    route("q", TIER_CFG, http_post=_post_returning("fast", 0.9, calls))
    c = calls[0]
    assert c["url"] == "https://example.invalid/v1/systemone"
    assert c["headers"]["Authorization"] == f"Bearer {KEY}"
    assert c["timeout"] == 2.0


def test_route_never_logs_the_key(with_key, caplog):
    caplog.set_level(logging.DEBUG, logger="xeno_rag.router")
    route("q", TIER_CFG, http_post=_post_raising(_StatusError(401)))
    assert KEY not in caplog.text
    assert "fallback" in caplog.text.lower()


def test_build_request_shape():
    body = build_request("Who is Rex?", TIER_CFG,
                         history=[{"question": "Who is Pyra?", "answer": "A Blade."}],
                         game="Xenoblade Chronicles 2")
    assert body["model"] == "jev-latest"
    assert body["state"] == {"question": "Who is Rex?", "previous_question": "Who is Pyra?",
                             "game": "Xenoblade Chronicles 2"}
    q = body["questions"]["tier"]
    assert q["type"] == "choice" and q["instructions"]
    assert set(q["criteria"]) == set(TIERS)


def test_build_request_minimal_state_and_truncation():
    body = build_request("x" * 10_000, TIER_CFG)
    assert set(body["state"]) == {"question"}
    assert len(body["state"]["question"]) <= 2000


def test_fallback_tier_defaults_and_validates():
    assert fallback_tier({}) == "thinking"
    assert fallback_tier({"router": {"fallback_tier": "fast"}}) == "fast"
    assert fallback_tier({"router": {"fallback_tier": "bogus"}}) == "thinking"


def test_apply_tier_fast_sets_model_and_depth():
    cfg = apply_tier(TIER_CFG, "fast")
    assert cfg["gemini_model"] == "gemini-3.5-flash-lite"
    assert cfg["top_k"] == 20 and cfg["max_chunks_per_page"] == 5
    assert cfg["hybrid_candidates"] == 60          # untouched base value
    assert "thinking_level" not in cfg
    assert cfg["answer_tier"] == "fast"
    assert "model" not in cfg                       # tier's `model` key maps to gemini_model only


def test_apply_tier_scholar_sets_thinking_level():
    cfg = apply_tier(TIER_CFG, "scholar")
    assert cfg["gemini_model"] == "gemini-3.8-flash" and cfg["thinking_level"] == "high"
    assert cfg["top_k"] == 96 and cfg["rerank_candidates"] == 224


def test_apply_tier_unknown_uses_fallback_entry():
    cfg = apply_tier(TIER_CFG, "nope")
    assert cfg["top_k"] == 40 and cfg["answer_tier"] == "thinking"


def test_apply_tier_without_map_is_identity_and_never_mutates():
    src = {"gemini_model": "gemini-3.8-flash", "top_k": 7}
    assert apply_tier(src, "fast") is src
    before = {**TIER_CFG}
    apply_tier(TIER_CFG, "scholar")
    assert TIER_CFG == before
