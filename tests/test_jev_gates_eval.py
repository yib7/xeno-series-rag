"""Tests for eval/run_jev_gates_eval.py: pure threshold-sweep math, the call budget guard, the
cases file schema, and one end-to-end proof of the JSONL record shape. No network: every test
supplies its own stubbed ``http_post`` (a fake real transport wrapped exactly the way the script
wraps it), a fake ``retrieve`` seam, and a fake embedder -- this script is never run live here.

`eval/` is not an importable package (see tests/test_run_gold_eval.py's docstring for why), so the
module is loaded by file path via `importlib.util.spec_from_file_location`.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from xeno_rag import rag as rag_mod

RUN_JEV_GATES_EVAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "eval", "run_jev_gates_eval.py")
CASES_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "jev_gates_cases.json")


def _load_module():
    spec = importlib.util.spec_from_file_location("run_jev_gates_eval", RUN_JEV_GATES_EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m():
    return _load_module()


# --- rate() ---

def test_rate_empty_list_is_zero(m):
    assert m.rate([], lambda r: True) == 0.0


def test_rate_counts_matching_fraction(m):
    records = [{"x": 1}, {"x": 2}, {"x": 1}, {"x": 1}]
    assert m.rate(records, lambda r: r["x"] == 1) == 0.75


# --- sweep(): topic ---

TOPIC_RECORDS = [
    {"topic": {"choice": "off_topic", "confidence": 0.75}},
    {"topic": {"choice": "off_topic", "confidence": 0.65}},
    {"topic": {"choice": "xeno", "confidence": 0.99}},
    {"topic": {"choice": None, "confidence": None}},
]


def test_sweep_topic_blocks_at_or_above_threshold_only(m):
    swept = m.sweep(TOPIC_RECORDS, "topic", thresholds=(0.7, 0.8))
    assert swept[0.7] == [True, False, False, False]
    assert swept[0.8] == [False, False, False, False]


def test_topic_blocked_ignores_xeno_regardless_of_confidence(m):
    assert m._topic_blocked_at({"topic": {"choice": "xeno", "confidence": 1.0}}, 0.1) is False


def test_topic_blocked_handles_missing_topic_key(m):
    assert m._topic_blocked_at({}, 0.7) is False


# --- sweep(): answerability (escalate/decline replay) ---

DECLINE_AT_SCHOLAR = {"tier": "scholar",
                      "check1": {"verdict": "not_covered", "confidence": 0.85}, "check2": None}
ESCALATE_AND_DECLINE = {"tier": "fast",
                        "check1": {"verdict": "not_covered", "confidence": 0.85},
                        "check2": {"verdict": "not_covered", "confidence": 0.82}}
ESCALATE_NOT_DECLINED = {"tier": "fast",
                         "check1": {"verdict": "not_covered", "confidence": 0.85},
                         "check2": {"verdict": "answered", "confidence": 0.9}}
ANSWERED_NO_ESCALATION = {"tier": "fast",
                          "check1": {"verdict": "answered", "confidence": 0.95}, "check2": None}
CHECK_NEVER_ATTEMPTED = {"tier": "fast", "check1": None, "check2": None}


@pytest.mark.parametrize("record,threshold,expected", [
    (DECLINE_AT_SCHOLAR, 0.7, (False, True)),        # already scholar: declined, no further escalation
    (DECLINE_AT_SCHOLAR, 0.9, (False, False)),        # 0.85 < 0.9: doesn't clear the threshold at all
    (ESCALATE_AND_DECLINE, 0.7, (True, True)),
    (ESCALATE_AND_DECLINE, 0.8, (True, True)),        # both checks (.85, .82) still clear 0.8
    (ESCALATE_AND_DECLINE, 0.9, (False, False)),      # check1 .85 < 0.9
    (ESCALATE_NOT_DECLINED, 0.7, (True, False)),      # escalated, but check2 says answered
    (ANSWERED_NO_ESCALATION, 0.7, (False, False)),
    (CHECK_NEVER_ATTEMPTED, 0.7, (False, False)),
])
def test_answerability_at_threshold(m, record, threshold, expected):
    assert m._answerability_at(record, threshold) == expected


def test_sweep_answerability_declined_column(m):
    records = [ESCALATE_AND_DECLINE, ESCALATE_NOT_DECLINED, ANSWERED_NO_ESCALATION]
    swept = m.sweep(records, "answerability", thresholds=(0.7, 0.9))
    assert swept[0.7] == [True, False, False]
    assert swept[0.9] == [False, False, False]


def test_sweep_rejects_unknown_conf_field(m):
    with pytest.raises(ValueError):
        m.sweep([], "bogus")


# --- ship_threshold() ---

def test_ship_threshold_picks_lowest_passing_threshold(m):
    # 5/200 = 2.5% false-block at 0.7 (fails the <=1% rule); those same 5 have confidence 0.75, so
    # at 0.8 none of them clear the threshold -> 0% false-block, which passes.
    records = []
    for i in range(200):
        blocked, conf = (True, 0.75) if i < 5 else (False, 0.0)
        records.append({"topic": {"choice": "off_topic" if blocked else "xeno", "confidence": conf}})
    assert m.ship_threshold(records, "topic") == 0.8


def test_ship_threshold_disables_when_no_threshold_passes(m):
    records = [{"topic": {"choice": "off_topic", "confidence": 0.99}} for _ in range(200)]
    assert m.ship_threshold(records, "topic") == "disable"


def test_ship_threshold_handles_empty_gold(m):
    assert m.ship_threshold([], "topic") == 0.7   # 0/0 -> 0.0 false rate, trivially passes


# --- CallBudget ---

def test_call_budget_allows_up_to_the_cap(m):
    budget = m.CallBudget(2)
    wrapped = budget.wrap(lambda url, *, json, headers, timeout: {"answers": {}})
    wrapped("u", json={}, headers={}, timeout=1)
    wrapped("u", json={}, headers={}, timeout=1)
    assert budget.count == 2


def test_call_budget_raises_past_the_cap(m):
    budget = m.CallBudget(2)
    calls = []

    def fake_post(url, *, json, headers, timeout):
        calls.append(url)
        return {"answers": {}}
    wrapped = budget.wrap(fake_post)
    wrapped("u", json={}, headers={}, timeout=1)
    wrapped("u", json={}, headers={}, timeout=1)
    with pytest.raises(m.BudgetExceeded):
        wrapped("u", json={}, headers={}, timeout=1)
    assert budget.count == 3
    assert len(calls) == 2      # the call that would exceed the budget never reaches the real post


# --- cases file schema ---

def test_cases_file_has_30_off_topic_20_not_covered_and_10_follow_up():
    cases = json.loads(Path(CASES_PATH).read_text(encoding="utf-8"))
    assert len(cases["off_topic"]) == 30
    assert len(cases["not_covered"]) == 20
    assert len(cases["follow_up"]) == 10


def test_cases_file_off_topic_entries_are_nonempty_strings():
    cases = json.loads(Path(CASES_PATH).read_text(encoding="utf-8"))
    for q in cases["off_topic"]:
        assert isinstance(q, str) and q.strip()


def test_cases_file_not_covered_entries_have_question_and_game():
    cases = json.loads(Path(CASES_PATH).read_text(encoding="utf-8"))
    for item in cases["not_covered"]:
        assert set(item.keys()) == {"question", "game"}
        assert isinstance(item["question"], str) and item["question"].strip()
        assert item["game"] is None or isinstance(item["game"], str)


def test_cases_file_follow_up_entries_have_previous_question_question_and_game():
    cases = json.loads(Path(CASES_PATH).read_text(encoding="utf-8"))
    for item in cases["follow_up"]:
        assert set(item.keys()) == {"previous_question", "question", "game"}
        assert isinstance(item["previous_question"], str) and item["previous_question"].strip()
        assert isinstance(item["question"], str) and item["question"].strip()
        assert item["game"] is None or isinstance(item["game"], str)


# --- end-to-end: run_case() proves the record shape via production route()/_ground() ---

BASE_CFG = {
    "top_k": 3, "gemini_model": "gemini-3.8-flash",
    "answer_tiers": {
        "fast": {"model": "gemini-3.5-flash-lite", "top_k": 20, "rerank_candidates": 50},
        "thinking": {"model": "gemini-3.8-flash", "top_k": 40, "rerank_candidates": 100},
        "scholar": {"model": "gemini-3.8-flash", "thinking_level": "high", "top_k": 96,
                   "rerank_candidates": 224},
    },
    "router": {"provider": "jev", "model": "jev-latest",
              "url": "https://example.invalid/v1/systemone", "timeout_seconds": 2},
}

CHUNK = {"title": "Rex", "game": "XC2", "url": "https://w/Rex",
        "text": "[XC2] Rex > Introduction: Rex is the salvager protagonist."}


class _FakeEmbedder:
    def embed_query(self, text):
        return ["fake-vector-for", text]


def _fake_real_post(url, *, json, headers, timeout):
    """Stands in for the real HTTP transport: routes on request shape, exactly like the live
    endpoint would -- a 3-question body for route(), a 1-question ``coverage`` body for check()."""
    questions = json.get("questions") or {}
    if "coverage" in questions:
        verdict, confidence = _fake_real_post.coverage_answers.pop(0)
        return {"model": "jev-latest",
               "answers": {"coverage": {"type": "choice", "choice": verdict, "confidence": confidence}}}
    return {"model": "jev-latest", "answers": {
        "tier": {"type": "choice", "choice": "fast", "confidence": 0.9},
        "topic": {"type": "choice", "choice": "xeno", "confidence": 0.95},
        "format": {"type": "choice", "choice": "list", "confidence": 0.9},
    }}


def test_run_case_end_to_end_record_shape(monkeypatch, m):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")
    monkeypatch.setattr(rag_mod, "retrieve", lambda *a, **kw: [dict(CHUNK, _score=1.0)])
    # First check says not_covered (>= the run's forced 0.7 decline_confidence) -> escalates once;
    # the second (post-escalation) check says answered -> not declined.
    _fake_real_post.coverage_answers = [("not_covered", 0.85), ("answered", 0.9)]

    cfg = m._eval_cfg(BASE_CFG)
    budget = m.CallBudget(1000)
    state = m._RecorderState()
    http_post = budget.wrap(m._make_recorder(_fake_real_post, state))

    rec = m.run_case("gold", "Who is Rex?", "XC2", cfg, _FakeEmbedder(), http_post, state)

    assert rec["kind"] == "gold"
    assert rec["question"] == "Who is Rex?"
    assert rec["game"] == "XC2"
    assert rec["previous_question"] is None
    assert rec["tier"] == "fast"
    assert rec["topic"] == {"choice": "xeno", "confidence": 0.95}
    assert rec["format"] == {"choice": "list", "confidence": 0.9}
    assert rec["check1"] == {"verdict": "not_covered", "confidence": 0.85}
    assert rec["check2"] == {"verdict": "answered", "confidence": 0.9}
    assert rec["escalated"] is True
    assert rec["declined_0_7"] is False
    assert isinstance(rec["ms"], float)
    assert budget.count == 3            # 1 routing call + 2 coverage calls (initial + escalated)


def test_run_case_declines_when_both_checks_say_not_covered(monkeypatch, m):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")
    monkeypatch.setattr(rag_mod, "retrieve", lambda *a, **kw: [dict(CHUNK, _score=1.0)])
    _fake_real_post.coverage_answers = [("not_covered", 0.85), ("not_covered", 0.82)]

    cfg = m._eval_cfg(BASE_CFG)
    budget = m.CallBudget(1000)
    state = m._RecorderState()
    http_post = budget.wrap(m._make_recorder(_fake_real_post, state))

    rec = m.run_case("not_covered", "Who is Captain Varnoth?", "XC2", cfg, _FakeEmbedder(),
                     http_post, state)
    assert rec["escalated"] is True
    assert rec["declined_0_7"] is True
    assert rec["check1"]["verdict"] == "not_covered"
    assert rec["check2"]["verdict"] == "not_covered"


def test_run_case_check_not_attempted_when_no_key(monkeypatch, m):
    """Without TYPESAFE_API_KEY, router._jev_call returns None before ever calling http_post (see
    router.py), so route() falls all the way back and _answerability_would_run is False: the record
    must show check1/check2 as None (never attempted), not a fabricated not_covered verdict."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(rag_mod, "retrieve", lambda *a, **kw: [dict(CHUNK, _score=1.0)])

    def _unreachable_post(url, *, json, headers, timeout):
        raise AssertionError("no HTTP call should be made without an API key")

    cfg = m._eval_cfg(BASE_CFG)
    budget = m.CallBudget(1000)
    state = m._RecorderState()
    http_post = budget.wrap(m._make_recorder(_unreachable_post, state))

    rec = m.run_case("gold", "Who is Rex?", "XC2", cfg, _FakeEmbedder(), http_post, state)

    assert rec["check1"] is None
    assert rec["check2"] is None
    assert rec["escalated"] is False
    assert rec["declined_0_7"] is False
    assert budget.count == 0


# --- print_summary(): follow_up rates are reported next to gold's (offline, no network) ---

def test_print_summary_reports_follow_up_rates(m, capsys):
    records = [
        {"kind": "gold", "topic": {"choice": "xeno", "confidence": 0.9}, "tier": "fast",
         "check1": {"verdict": "answered", "confidence": 0.9}, "check2": None,
         "format": {"choice": "list", "confidence": 0.9}},
        {"kind": "follow_up", "topic": {"choice": "off_topic", "confidence": 0.9}, "tier": "fast",
         "check1": {"verdict": "not_covered", "confidence": 0.9}, "check2": None,
         "format": {"choice": None, "confidence": None}},
    ]
    m.print_summary(records)
    out = capsys.readouterr().out
    assert "follow_up 1" in out
    assert "follow_up false-block" in out
    assert "follow_up false-decline" in out


# --- run_cases(): abort on budget overrun via the real route()/_ground() path ---

def test_run_case_threads_previous_question_into_routing_and_grounding(monkeypatch, m):
    """Final-review fix 2: a 'follow_up' case's ``previous_question`` must reach BOTH the routing
    call and the answerability check as ``state["previous_question"]`` -- production's ``history``
    shape, threaded through exactly as a real follow-up turn would be."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")
    monkeypatch.setattr(rag_mod, "retrieve", lambda *a, **kw: [dict(CHUNK, _score=1.0)])
    _fake_real_post.coverage_answers = [("answered", 0.9)]
    bodies = []

    def capturing_post(url, *, json, headers, timeout):
        bodies.append(json)
        return _fake_real_post(url, json=json, headers=headers, timeout=timeout)

    cfg = m._eval_cfg(BASE_CFG)
    budget = m.CallBudget(1000)
    state = m._RecorderState()
    http_post = budget.wrap(m._make_recorder(capturing_post, state))

    rec = m.run_case("follow_up", "What species is she?", "XC2", cfg, _FakeEmbedder(), http_post,
                     state, previous_question="Who is Nia in Xenoblade Chronicles 2?")

    assert rec["kind"] == "follow_up"
    assert rec["previous_question"] == "Who is Nia in Xenoblade Chronicles 2?"
    routing_body, coverage_body = bodies[0], bodies[1]
    assert routing_body["state"]["previous_question"] == "Who is Nia in Xenoblade Chronicles 2?"
    assert coverage_body["state"]["previous_question"] == "Who is Nia in Xenoblade Chronicles 2?"


def test_run_case_without_previous_question_omits_it(monkeypatch, m):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")
    monkeypatch.setattr(rag_mod, "retrieve", lambda *a, **kw: [dict(CHUNK, _score=1.0)])
    _fake_real_post.coverage_answers = [("answered", 0.9)]
    bodies = []

    def capturing_post(url, *, json, headers, timeout):
        bodies.append(json)
        return _fake_real_post(url, json=json, headers=headers, timeout=timeout)

    cfg = m._eval_cfg(BASE_CFG)
    budget = m.CallBudget(1000)
    state = m._RecorderState()
    http_post = budget.wrap(m._make_recorder(capturing_post, state))

    m.run_case("gold", "Who is Rex?", "XC2", cfg, _FakeEmbedder(), http_post, state)

    assert "previous_question" not in bodies[0]["state"]


def test_run_cases_accepts_4_tuples_for_follow_up_cases(monkeypatch, m):
    """run_cases() must accept both plain 3-tuples (gold/off_topic/not_covered) and 4-tuples carrying
    a previous_question (follow_up) in the same cases list."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")
    monkeypatch.setattr(rag_mod, "retrieve", lambda *a, **kw: [dict(CHUNK, _score=1.0)])
    _fake_real_post.coverage_answers = [("answered", 0.9), ("answered", 0.9)]

    cfg = m._eval_cfg(BASE_CFG)
    budget = m.CallBudget(1000)
    state = m._RecorderState()
    http_post = budget.wrap(m._make_recorder(_fake_real_post, state))
    cases = [("gold", "Who is Rex?", "XC2"),
            ("follow_up", "What species is she?", "XC2", "Who is Nia in Xenoblade Chronicles 2?")]

    records, aborted = m.run_cases(cases, cfg, embedder=_FakeEmbedder(), http_post=http_post,
                                   budget=budget, state=state)

    assert aborted is False
    assert [r["kind"] for r in records] == ["gold", "follow_up"]


def test_run_cases_aborts_and_discards_the_over_budget_record(monkeypatch, m):
    """BudgetExceeded raised inside http_post never propagates out of route()/_ground() -- router.
    _jev_call swallows it via its broad ``except Exception`` and returns None, so a caught-exception
    abort (the original, buggy design) would never fire. run_cases() must instead poll budget.count
    after each case and stop -- proven here by driving it through the real route()/_ground() pipeline
    (only the transport is stubbed), never through main() (which would reload the real .env key)."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")
    monkeypatch.setattr(rag_mod, "retrieve", lambda *a, **kw: [dict(CHUNK, _score=1.0)])
    # No escalation needed: 1 routing call + 1 coverage call per case.
    _fake_real_post.coverage_answers = [("answered", 0.9), ("answered", 0.9), ("answered", 0.9)]

    cfg = m._eval_cfg(BASE_CFG)
    budget = m.CallBudget(2)      # exceeded on the 2nd case's routing call (the 3rd call overall)
    state = m._RecorderState()
    http_post = budget.wrap(m._make_recorder(_fake_real_post, state))
    cases = [("gold", "Who is Rex?", "XC2"), ("gold", "Who is Nia?", "XC2"),
            ("gold", "Who is Pyra?", "XC2")]

    records, aborted = m.run_cases(cases, cfg, embedder=_FakeEmbedder(), http_post=http_post,
                                   budget=budget, state=state)

    assert aborted is True
    assert len(records) == 1                 # only the 1st case's record kept; the over-budget one dropped
    assert records[0]["question"] == "Who is Rex?"
    assert budget.count > budget.max_calls    # confirms the overrun genuinely happened, not a fluke pass


def test_run_cases_hands_each_kept_record_to_on_record_as_it_is_made(monkeypatch, m):
    """A caller that persists via ``on_record`` keeps everything already paid for even if a later case
    blows up: the callback sees record 1 before case 2 runs, and the over-budget record is never
    handed over."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")
    monkeypatch.setattr(rag_mod, "retrieve", lambda *a, **kw: [dict(CHUNK, _score=1.0)])
    _fake_real_post.coverage_answers = [("answered", 0.9), ("answered", 0.9), ("answered", 0.9)]

    cfg = m._eval_cfg(BASE_CFG)
    budget = m.CallBudget(2)
    state = m._RecorderState()
    http_post = budget.wrap(m._make_recorder(_fake_real_post, state))
    seen = []
    cases = [("gold", "Who is Rex?", "XC2"), ("gold", "Who is Nia?", "XC2")]

    records, aborted = m.run_cases(cases, cfg, embedder=_FakeEmbedder(), http_post=http_post,
                                   budget=budget, state=state, on_record=seen.append)

    assert aborted is True
    assert [r["question"] for r in seen] == ["Who is Rex?"] == [r["question"] for r in records]
