"""Tests for Jev answer-tier routing. No network: the HTTP call is injected."""

import logging
import math

import httpx
import pytest

from xeno_rag import router as router_mod
from xeno_rag.router import (
    FORMATS,
    TIERS,
    TOPICS,
    Route,
    answerability_on,
    apply_tier,
    build_request,
    fallback_tier,
    jev_available,
    off_topic_gate_on,
    route,
)
from xeno_rag.router import _jev_call as jev_call

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


def _post_multi(calls=None, **wanted):
    """Build a stub Jev response with any subset of the three answers. Each keyword is
    ``qid=(choice, confidence)``; a ``qid`` left out of ``wanted`` is simply absent from
    ``answers``, the same shape a real Jev reply has when a question was skipped or dropped."""
    def post(url, *, json, headers, timeout):
        if calls is not None:
            calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        answers = {}
        for qid, val in wanted.items():
            if val is None:
                continue
            choice, confidence = val
            answers[qid] = {"type": "choice", "choice": choice, "confidence": confidence,
                            "probabilities": {choice: confidence}}
        return {"model": "jev-latest", "answers": answers}
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


# --- Minor 1: bad router config must never raise ---

def test_route_non_dict_router_block_is_offline(with_key):
    calls = []
    cfg = {**TIER_CFG, "router": "bogus"}
    r = route("q", cfg, http_post=_post_returning("fast", 0.9, calls))
    assert r.source == "fallback" and calls == []


def test_route_bad_min_confidence_falls_back_to_default(with_key):
    cfg = {**TIER_CFG, "router": {**TIER_CFG["router"], "min_confidence": "high"}}
    # default min_confidence (0.5) applies: 0.9 clears it, so the jev choice is still usable.
    r = route("q", cfg, http_post=_post_returning("fast", 0.9))
    assert r == Route("fast", "jev", 0.9)


def test_route_bad_timeout_seconds_falls_back_to_default(with_key):
    calls = []
    cfg = {**TIER_CFG, "router": {**TIER_CFG["router"], "timeout_seconds": "slow"}}
    route("q", cfg, http_post=_post_returning("fast", 0.9, calls))
    assert calls[0]["timeout"] == 2.0


# --- Minor 2: NaN confidence must not pass the threshold ---

def test_route_nan_confidence_falls_back(with_key):
    r = route("q", TIER_CFG, http_post=_post_returning("fast", math.nan))
    assert r.tier == "thinking" and r.source == "fallback"
    assert math.isnan(r.confidence)


# --- Minor 4: missing key logs once per process ---

def test_route_missing_key_logs_once(monkeypatch, caplog):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(router_mod, "_missing_key_warned", False)
    caplog.set_level(logging.INFO, logger="xeno_rag.router")
    cfg = {**TIER_CFG, "router": {**TIER_CFG["router"]}}
    route("q", cfg)
    route("q", cfg)
    msgs = [r.message for r in caplog.records if "TYPESAFE_API_KEY not set" in r.message]
    assert len(msgs) == 1


# --- Important 2: shared httpx.Client, real path via httpx.MockTransport (no network) ---

@pytest.fixture
def mock_transport_client():
    """Reset the module-level shared client before and after each test so a mock transport injected
    here never leaks into other tests (and no other test's real client leaks in here)."""
    router_mod._reset_client_for_tests()
    yield
    router_mod._reset_client_for_tests()


def test_default_post_200_returns_jev_route(with_key, mock_transport_client):
    def handler(request):
        assert request.headers["authorization"] == f"Bearer {KEY}"
        return httpx.Response(200, json={"answers": {"tier": {"choice": "scholar", "confidence": 0.9}}})
    router_mod._get_client(transport=httpx.MockTransport(handler))
    r = route("q", TIER_CFG)
    assert r == Route("scholar", "jev", 0.9)


def test_default_post_401_falls_back_and_logs_status_not_key(with_key, mock_transport_client, caplog):
    def handler(request):
        return httpx.Response(401, json={"error": "unauthorized"})
    router_mod._get_client(transport=httpx.MockTransport(handler))
    caplog.set_level(logging.WARNING, logger="xeno_rag.router")
    r = route("q", TIER_CFG)
    assert r.tier == "thinking" and r.source == "fallback"
    assert "401" in caplog.text
    assert KEY not in caplog.text


def test_default_post_reuses_shared_client_across_calls(with_key, mock_transport_client):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"answers": {"tier": {"choice": "fast", "confidence": 0.9}}})
    router_mod._get_client(transport=httpx.MockTransport(handler))
    client_before = router_mod._get_client()
    route("q", TIER_CFG)
    route("q", TIER_CFG)
    client_after = router_mod._get_client()
    assert len(calls) == 2
    assert client_before is client_after       # same client instance, not rebuilt per call


# --- SP1: three-question request + shared _jev_call ---------------------------------------------

def test_build_request_has_three_questions_with_verbatim_criteria():
    body = build_request("Who is Rex?", TIER_CFG)
    q = body["questions"]
    assert set(q) == {"tier", "topic", "format"}
    assert set(q["tier"]["criteria"]) == set(TIERS)
    assert set(q["topic"]["criteria"]) == set(TOPICS)
    assert set(q["format"]["criteria"]) == set(FORMATS)
    for qid in ("tier", "topic", "format"):
        assert q[qid]["type"] == "choice" and q[qid]["instructions"]
    assert q["topic"]["criteria"]["xeno"] == router_mod.TOPIC_CRITERIA["xeno"]
    assert q["topic"]["criteria"]["off_topic"] == router_mod.TOPIC_CRITERIA["off_topic"]
    assert q["format"]["criteria"]["table"] == router_mod.FORMAT_CRITERIA["table"]
    assert q["format"]["criteria"]["list"] == router_mod.FORMAT_CRITERIA["list"]
    assert q["format"]["criteria"]["prose"] == router_mod.FORMAT_CRITERIA["prose"]
    assert q["topic"]["instructions"] == router_mod.TOPIC_INSTRUCTIONS
    assert q["format"]["instructions"] == router_mod.FORMAT_INSTRUCTIONS


def test_post_returning_single_tier_answer_leaves_topic_and_format_none(with_key):
    """The pre-SP1 stub, which only ever returned a `tier` answer, must keep working unmodified."""
    r = route("q", TIER_CFG, http_post=_post_returning("fast", 0.9))
    assert r.tier == "fast" and r.topic is None and r.format is None


def test_route_tier_ok_topic_malformed_keeps_tier_drops_topic(with_key):
    post = _post_multi(tier=("fast", 0.9), topic=("nonsense", 0.9))
    r = route("q", TIER_CFG, http_post=post)
    assert r.tier == "fast" and r.source == "jev"
    assert r.topic is None


def test_route_topic_off_topic_above_threshold(with_key):
    post = _post_multi(tier=("fast", 0.9), topic=("off_topic", 0.9))
    r = route("q", TIER_CFG, http_post=post)
    assert r.topic == "off_topic"


def test_route_topic_off_topic_below_threshold_is_none(with_key):
    post = _post_multi(tier=("fast", 0.9), topic=("off_topic", 0.06))
    r = route("q", TIER_CFG, http_post=post)
    assert r.topic is None


def test_route_topic_xeno_is_recorded_regardless_of_confidence(with_key):
    post = _post_multi(tier=("fast", 0.9), topic=("xeno", 0.55))
    r = route("q", TIER_CFG, http_post=post)
    assert r.topic == "xeno"


def test_route_topic_missing_answer_is_none(with_key):
    post = _post_multi(tier=("fast", 0.9))
    r = route("q", TIER_CFG, http_post=post)
    assert r.topic is None


def test_route_format_low_confidence_is_none(with_key):
    post = _post_multi(tier=("fast", 0.9), format=("table", 0.2))
    r = route("q", TIER_CFG, http_post=post)
    assert r.format is None


def test_route_format_unknown_choice_is_none(with_key):
    post = _post_multi(tier=("fast", 0.9), format=("paragraph", 0.9))
    r = route("q", TIER_CFG, http_post=post)
    assert r.format is None


def test_route_format_valid_and_confident_is_recorded(with_key):
    post = _post_multi(tier=("fast", 0.9), format=("list", 0.8))
    r = route("q", TIER_CFG, http_post=post)
    assert r.format == "list"


def test_route_forced_tier_with_key_calls_jev_for_topic_and_format(with_key):
    calls = []
    post = _post_multi(calls, tier=("scholar", 0.99), topic=("xeno", 0.9), format=("table", 0.9))
    r = route("q", TIER_CFG, http_post=post, forced_tier="fast")
    assert len(calls) == 1                     # the call was made...
    assert r.tier == "fast" and r.source == "override" and r.confidence is None  # ...but tier ignored
    assert r.topic == "xeno" and r.format == "table"


def test_route_forced_tier_without_key_makes_no_call(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    calls = []
    r = route("q", TIER_CFG, http_post=_post_returning("fast", 0.9, calls), forced_tier="thinking")
    assert r == Route("thinking", "override") and calls == []


def test_route_forced_tier_fixed_provider_makes_no_call(with_key):
    calls = []
    cfg = {**TIER_CFG, "router": {**TIER_CFG["router"], "provider": "fixed"}}
    r = route("q", cfg, http_post=_post_returning("fast", 0.9, calls), forced_tier="scholar")
    assert r == Route("scholar", "override") and calls == []


def test_route_forced_tier_call_fails_still_overrides(with_key):
    r = route("q", TIER_CFG, http_post=_post_raising(_StatusError(500)), forced_tier="fast")
    assert r == Route("fast", "override")


def test_route_forced_tier_invalid_is_ignored_and_auto_routes(with_key):
    post = _post_multi(tier=("scholar", 0.9))
    r = route("q", TIER_CFG, http_post=post, forced_tier="bogus")
    assert r.tier == "scholar" and r.source == "jev"


# --- _jev_call: the shared helper reused by the answerability check (spec section 4) -------------

def test_jev_call_returns_answers_dict(with_key):
    post = _post_multi(tier=("fast", 0.9))
    answers = jev_call({"question": "q"}, router_mod._questions(), TIER_CFG, http_post=post)
    assert answers == {"tier": {"type": "choice", "choice": "fast", "confidence": 0.9,
                                "probabilities": {"fast": 0.9}}}


def test_jev_call_no_provider_returns_none_without_calling(with_key):
    calls = []
    cfg = {**TIER_CFG, "router": {**TIER_CFG["router"], "provider": "fixed"}}
    assert jev_call({}, {}, cfg, http_post=_post_returning("fast", 0.9, calls)) is None
    assert calls == []


def test_jev_call_no_key_returns_none_without_calling(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    calls = []
    assert jev_call({}, {}, TIER_CFG, http_post=_post_returning("fast", 0.9, calls)) is None
    assert calls == []


def test_jev_call_exception_returns_none_and_logs_no_key(with_key, caplog):
    caplog.set_level(logging.DEBUG, logger="xeno_rag.router")
    result = jev_call({}, {}, TIER_CFG, http_post=_post_raising(_StatusError(500)))
    assert result is None
    assert KEY not in caplog.text
    assert "500" in caplog.text


def test_jev_call_non_dict_answers_is_treated_as_failure(with_key, caplog):
    caplog.set_level(logging.DEBUG, logger="xeno_rag.router")

    def bad_post(url, **kw):
        return {"answers": "not-a-dict"}
    assert jev_call({}, {}, TIER_CFG, http_post=bad_post) is None


def test_jev_call_never_raises_on_broken_post(with_key):
    def broken(url, **kw):
        raise RuntimeError("network is down")
    assert jev_call({}, {}, TIER_CFG, http_post=broken) is None


# --- Config getters --------------------------------------------------------------------------

def test_jev_available_requires_provider_and_key(with_key):
    assert jev_available(TIER_CFG) is True
    cfg = {**TIER_CFG, "router": {**TIER_CFG["router"], "provider": "fixed"}}
    assert jev_available(cfg) is False


def test_jev_available_false_without_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert jev_available(TIER_CFG) is False


def test_off_topic_gate_on_default_false_and_reads_cfg():
    assert off_topic_gate_on(TIER_CFG) is False
    cfg = {**TIER_CFG, "router": {**TIER_CFG["router"], "off_topic_gate": True}}
    assert off_topic_gate_on(cfg) is True


def test_answerability_on_default_false_and_reads_cfg():
    assert answerability_on(TIER_CFG) is False
    cfg = {**TIER_CFG, "router": {**TIER_CFG["router"], "answerability_check": True}}
    assert answerability_on(cfg) is True
