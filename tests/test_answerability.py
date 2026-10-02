"""Tests for the post-rerank answerability check. No network: the HTTP call is injected."""

import pytest

from xeno_rag.answerability import (
    COVERAGE,
    COVERAGE_CRITERIA,
    Verdict,
    check,
    is_not_covered,
    passages,
)

KEY = "test-key-not-real"

CFG = {
    "router": {"provider": "jev", "model": "jev-latest", "url": "https://example.invalid/v1/systemone",
               "timeout_seconds": 2, "decline_confidence": 0.8, "answerability_passages": 8},
}

CHUNKS = [
    {"title": "Rex", "game": "XC2",
     "text": "[XC2] Rex > Introduction: Rex is the salvager protagonist of Xenoblade Chronicles 2."},
    {"title": "Pyra", "game": "XC2",
     "text": "[XC2] Pyra > infobox: The Aegis. Element: Fire."},
]


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)


def _post_returning(choice, confidence, calls=None):
    def post(url, *, json, headers, timeout):
        if calls is not None:
            calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return {"model": "jev-latest",
                "answers": {"coverage": {"type": "choice", "choice": choice, "confidence": confidence}}}
    return post


# --- passages ---

def test_passages_strips_breadcrumb_and_formats():
    out = passages(CHUNKS, 8)
    assert out[0] == "Rex (XC2): Introduction: Rex is the salvager protagonist of Xenoblade Chronicles 2."
    assert out[1] == "Pyra (XC2): infobox: The Aegis. Element: Fire."
    assert "[XC2]" not in out[0]                   # breadcrumb game tag dropped, not double-counted


def test_passages_limits_to_n():
    assert len(passages(CHUNKS, 1)) == 1


def test_passages_collapses_whitespace():
    chunks = [{"title": "T", "game": "G", "text": "line one\n\n  line   two"}]
    assert passages(chunks, 8) == ["T (G): line one line two"]


def test_passages_trims_long_text_on_word_boundary_with_ellipsis():
    long_text = "lorem " * 400  # 2400 chars, well over the 1500-char cut
    chunks = [{"title": "T", "game": "G", "text": long_text}]
    out = passages(chunks, 8)
    body = out[0].split(": ", 1)[1]
    assert body.endswith("…")
    assert len(body) <= 1501                      # 1500 chars + the ellipsis
    assert not body[:-1].endswith(" ")             # cut cleanly on a word boundary, no dangling space


def test_passages_short_text_is_untouched():
    chunks = [{"title": "T", "game": "G", "text": "short"}]
    assert passages(chunks, 8) == ["T (G): short"]


# --- check ---

@pytest.mark.parametrize("choice", COVERAGE)
def test_check_parses_each_coverage_choice(with_key, choice):
    v = check("q", CHUNKS, CFG, http_post=_post_returning(choice, 0.85))
    assert v == Verdict(choice, 0.85)


def test_check_malformed_choice_is_verdict_none(with_key):
    v = check("q", CHUNKS, CFG, http_post=_post_returning("bogus_choice", 0.9))
    assert v == Verdict(None)


def test_check_empty_chunks_is_not_covered_without_a_call(with_key):
    calls = []
    v = check("q", [], CFG, http_post=_post_returning("answered", 0.9, calls))
    assert v == Verdict("not_covered", 1.0)
    assert calls == []


def test_check_call_failure_is_verdict_none(with_key):
    def boom(url, *, json, headers, timeout):
        raise RuntimeError("boom")
    assert check("q", CHUNKS, CFG, http_post=boom) == Verdict(None)


def test_check_no_key_is_verdict_none_and_makes_no_call(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    calls = []
    v = check("q", CHUNKS, CFG, http_post=_post_returning("answered", 0.9, calls))
    assert v == Verdict(None)
    assert calls == []


def test_check_never_raises(with_key):
    """check() must swallow anything, including a completely broken http_post (e.g. one that returns
    a non-JSON-shaped object), the same guarantee _jev_call makes."""
    def broken(url, *, json, headers, timeout):
        return "not a dict"
    assert check("q", CHUNKS, CFG, http_post=broken) == Verdict(None)


def test_check_sends_question_and_top_n_passages(with_key):
    calls = []
    check("What is Rex's weapon?", CHUNKS, CFG, http_post=_post_returning("answered", 0.9, calls))
    body = calls[0]["json"]
    assert body["state"]["question"] == "What is Rex's weapon?"
    assert body["state"]["passages"] == passages(CHUNKS, 8)
    assert body["questions"]["coverage"]["criteria"] == COVERAGE_CRITERIA
    assert body["questions"]["coverage"]["instructions"]


def test_check_respects_configured_passage_count(with_key):
    calls = []
    cfg = {"router": {**CFG["router"], "answerability_passages": 1}}
    check("q", CHUNKS, cfg, http_post=_post_returning("answered", 0.9, calls))
    assert calls[0]["json"]["state"]["passages"] == passages(CHUNKS, 1)


# --- answerability_passages can arrive as a float, string, or garbage config value (YAML and JSON
# don't force ints). It must be coerced before it reaches a list slice, which would raise TypeError. ---

@pytest.mark.parametrize("raw,expected_n", [
    (8.0, 8),               # float -> int
    ("8", 8),                # numeric string -> int
    ("not a number", 8),     # garbage -> falls back to the 8 default, never raises
    (None, 8),                # explicit None -> falls back to the default
    (0, 1),                   # clamped up to at least 1 (never "judge zero passages")
    (-3, 1),                  # negative also clamps to 1
    (2, 2),                   # a plain valid int passes through unchanged
])
def test_check_coerces_and_clamps_passage_count(with_key, raw, expected_n):
    cfg = {"router": {"provider": "jev", "answerability_passages": raw}}
    calls = []
    check("q", CHUNKS, cfg, http_post=_post_returning("answered", 0.9, calls))
    assert calls[0]["json"]["state"]["passages"] == passages(CHUNKS, expected_n)


def test_check_never_logs_the_key(with_key, caplog):
    caplog.set_level("DEBUG")

    def boom(url, *, json, headers, timeout):
        raise RuntimeError("boom")
    check("q", CHUNKS, CFG, http_post=boom)
    assert KEY not in caplog.text


# --- is_not_covered ---

def test_is_not_covered_true_at_or_above_threshold():
    assert is_not_covered(Verdict("not_covered", 0.8), CFG) is True
    assert is_not_covered(Verdict("not_covered", 0.95), CFG) is True


def test_is_not_covered_false_below_threshold():
    assert is_not_covered(Verdict("not_covered", 0.5), CFG) is False


def test_is_not_covered_false_for_answered_or_partial():
    assert is_not_covered(Verdict("answered", 0.99), CFG) is False
    assert is_not_covered(Verdict("partial", 0.99), CFG) is False


def test_is_not_covered_false_for_verdict_none():
    assert is_not_covered(Verdict(None), CFG) is False


def test_is_not_covered_false_without_confidence():
    assert is_not_covered(Verdict("not_covered", None), CFG) is False


def test_is_not_covered_uses_default_threshold_when_unset():
    cfg = {"router": {"provider": "jev"}}
    assert is_not_covered(Verdict("not_covered", 0.8), cfg) is True
    assert is_not_covered(Verdict("not_covered", 0.79), cfg) is False


# --- Final-review fix 2: follow-up context (history -> state["previous_question"]) ---

def test_check_adds_previous_question_from_history(with_key):
    calls = []
    history = [{"question": "Who is Nia in Xenoblade Chronicles 2?", "answer": "She is a Gormotti."}]
    check("what is her element?", CHUNKS, CFG, http_post=_post_returning("answered", 0.9, calls),
         history=history)
    assert calls[0]["json"]["state"]["previous_question"] == "Who is Nia in Xenoblade Chronicles 2?"


def test_check_no_history_omits_previous_question(with_key):
    calls = []
    check("q", CHUNKS, CFG, http_post=_post_returning("answered", 0.9, calls), history=None)
    assert "previous_question" not in calls[0]["json"]["state"]


def test_check_history_without_a_question_omits_previous_question(with_key):
    calls = []
    check("q", CHUNKS, CFG, http_post=_post_returning("answered", 0.9, calls), history=[{"question": ""}])
    assert "previous_question" not in calls[0]["json"]["state"]


def test_check_previous_question_trimmed_to_max_state_chars(with_key):
    from xeno_rag.router import MAX_STATE_CHARS
    calls = []
    long_prev = "x" * (MAX_STATE_CHARS + 500)
    check("q", CHUNKS, CFG, http_post=_post_returning("answered", 0.9, calls),
         history=[{"question": long_prev}])
    assert calls[0]["json"]["state"]["previous_question"] == long_prev[:MAX_STATE_CHARS]
