"""Live-eval the Jev off-topic and answerability gates against the gold set and hand-written negative
sets plus a follow-up set (eval/jev_gates_cases.json), and sweep the ship-rule threshold from one
pass.

Makes live Jev (TypeSafe) calls, never Gemini. This is the ONLY script in the repo meant to spend
real money; everything else in the test suite stubs ``http_post``. ``CallBudget`` guards the spend
(default 1,000 calls, ~$0.20 upper bound at <=$0.0002/call): past the cap the eval aborts, and still
writes whatever it collected, rather than run away.

Design:

- Uses the PRODUCTION code paths for routing and grounding, ``router.route()`` and ``rag._ground()``
  (which calls ``answerability.check()``), instead of reimplementing the off-topic, escalate and
  decline decisions here. The only local logic recomputes those decisions at swept thresholds from
  RAW recorded confidences (no extra calls). ``Route.topic`` and ``Route.format`` already hide a
  choice below their configured threshold, and the escalate/decline decision has to be replayed at
  0.7/0.8/0.9 without running the pipeline three times.
- The run forces ``router.decline_confidence`` to 0.7 (the lowest sweep threshold),
  ``answerability_check: True`` and ``off_topic_gate: False`` in a COPY of the loaded config. Every
  case then runs the answerability check, so check1/check2 raw verdicts get recorded even for
  off_topic/not_covered cases. Escalation, which re-retrieves and re-checks, also happens at the
  lowest threshold, so a higher threshold's replay always has check2 whenever it needs it (see
  ``_answerability_at``).
- A single ``http_post`` callable (``CallBudget.wrap`` around a small recorder around the real
  transport) is passed into BOTH ``route()`` and ``answerability.check()`` (via ``rag._ground``'s
  ``http_post=`` seam). One counter covers the whole run, and the raw Jev ``answers`` for every
  call, routing and coverage alike, are captured before any threshold discards them.

Usage:
  python -m eval.run_jev_gates_eval                       # full run: 200 gold + 30 off-topic + 20 not-covered + 10 follow-ups
  python -m eval.run_jev_gates_eval --limit 20             # smoke run: first 20 gold questions only
  python -m eval.run_jev_gates_eval --skip-gold            # off-topic, not-covered and follow-up cases only
  python -m eval.run_jev_gates_eval --skip-negatives       # gold only (also skips follow-ups)
  python -m eval.run_jev_gates_eval --max-calls 50         # tighter budget guard for a smoke run
"""

import argparse
import json
import math
import time
from pathlib import Path

from xeno_rag import router
from xeno_rag.answerability import COVERAGE
from xeno_rag.config import load_config
from xeno_rag.embed_index import Embedder
from xeno_rag.rag import GAME_NAMES, _answerability_would_run, _embed_query, _ground, _retrieval_query
from xeno_rag.router import FORMATS, TOPICS, apply_tier, jev_available, route

GOLD_PATH = Path("eval") / "gold_questions.json"
CASES_PATH = Path("eval") / "jev_gates_cases.json"
OUT_PATH = Path("eval") / "jev_gates_results.jsonl"

# The lowest sweep threshold. The run itself declines and escalates at this value (see the module
# docstring), so every higher threshold's replay finds what it needs already recorded.
RUN_DECLINE_CONFIDENCE = 0.7
THRESHOLDS = (0.7, 0.8, 0.9)
# Ship rule (spec §6): a gate ships enabled only if its gold false rate is <= 1% (2 of 200 questions).
SHIP_RULE_MAX_RATE = 0.01


# --------------------------------------------------------------------------------------------
# Budget guard
# --------------------------------------------------------------------------------------------

class BudgetExceeded(RuntimeError):
    """Raised by a ``CallBudget``-wrapped ``http_post`` once ``max_calls`` would be exceeded."""


class CallBudget:
    """Counts every call through a wrapped ``http_post`` and raises ``BudgetExceeded`` past the cap.

    Construct with just the cap (``CallBudget(1000)``), then ``.wrap(http_post)`` the real transport
    (or another wrapper) to get a callable with the same ``post(url, *, json, headers, timeout)``
    signature ``router._jev_call`` expects. ``.count`` is the number of calls attempted so far
    (including the one that raised, if any), so a live run can report exactly how much it spent."""

    def __init__(self, max_calls: int = 1000):
        self.max_calls = max_calls
        self.count = 0

    def wrap(self, http_post):
        def wrapped(url, *, json, headers, timeout):
            self.count += 1
            if self.count > self.max_calls:
                raise BudgetExceeded(
                    f"jev gates eval: exceeded its call budget ({self.max_calls} calls)")
            return http_post(url, json=json, headers=headers, timeout=timeout)
        return wrapped


# --------------------------------------------------------------------------------------------
# Recording http_post: keeps the RAW Jev answers dict from every call, tagged by request shape
# --------------------------------------------------------------------------------------------

class _RecorderState:
    """Per-case scratch space the recording ``http_post`` writes into. Call ``reset()`` before each
    case so its record only reflects calls made while processing that case."""

    def __init__(self):
        self.routing = None     # raw `answers` dict from the one tier/topic/format call, or None
        self.coverage = []      # raw `answers` dicts from each coverage call, in call order (0-2)

    def reset(self):
        self.routing = None
        self.coverage = []


def _make_recorder(real_post, state: _RecorderState):
    """Wrap ``real_post`` (the actual HTTP transport) so every call's raw ``answers`` dict is kept on
    ``state``, tagged as the routing call or a coverage call by the request's ``questions`` key, NOT
    re-derived from the threshold-filtered return values of route() and check(). That keeps the raw
    topic/format confidence available for the sweep even when it is below
    ``off_topic_confidence``/``min_confidence`` (``Route`` would report ``None``)."""

    def post(url, *, json, headers, timeout):
        data = real_post(url, json=json, headers=headers, timeout=timeout)
        answers = data.get("answers") if isinstance(data, dict) else None
        answers = answers if isinstance(answers, dict) else {}
        if "coverage" in (json.get("questions") or {}):
            state.coverage.append(answers)
        else:
            state.routing = answers
        return data
    return post


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------

def _eval_cfg(cfg: dict) -> dict:
    """A copy of ``cfg`` with the router gates forced for this eval run: ``answerability_check: True``
    and ``off_topic_gate: False`` (every case runs grounding the same way regardless of topic; the
    off-topic short-circuit is never exercised here, only its raw topic confidence is recorded) and
    ``decline_confidence`` pinned to the lowest sweep threshold (see the module docstring). Never
    mutates the input."""
    router_cfg = dict(cfg.get("router") or {})
    router_cfg["answerability_check"] = True
    router_cfg["off_topic_gate"] = False
    router_cfg["decline_confidence"] = RUN_DECLINE_CONFIDENCE
    return {**cfg, "router": router_cfg}


# --------------------------------------------------------------------------------------------
# One case: route() + production grounding, recording raw topic/format/coverage
# --------------------------------------------------------------------------------------------

def _pop_coverage(queue: list) -> dict:
    """Pop the next recorded coverage call. When the queue is empty because
    ``answerability.check()`` short-circuited on empty retrieval (``Verdict("not_covered", 1.0)``
    with no call made, per its docstring), return that same constant. Callers invoke this only when
    a check was actually attempted (``run_case`` guards that), so an empty queue can only mean the
    empty-chunks short-circuit, never "never checked"."""
    if queue:
        answers = queue.pop(0)
        choice, confidence = router._choice(answers, "coverage", COVERAGE)
        return {"verdict": choice, "confidence": confidence}
    return {"verdict": "not_covered", "confidence": 1.0}


def run_case(kind: str, question: str, game: str | None, cfg: dict, embedder, http_post,
            state: _RecorderState, previous_question: str | None = None) -> dict:
    """Route and ground one case through production code paths, returning its JSONL record. ``cfg``
    is already the forced eval cfg (see ``_eval_cfg``).

    ``previous_question`` (set only for ``"follow_up"`` cases, see ``eval/jev_gates_cases.json``)
    is wrapped into the ``history`` shape ``rag.py`` uses everywhere: a list of one
    ``{"question", "answer"}`` turn with an empty ``answer``, since neither ``route()`` nor
    ``_ground()``/``answerability.check()`` reads the answer text, only the previous question. It is
    passed to BOTH the routing call and grounding, as a live follow-up turn would be, so the recorded
    topic/format/coverage answers reflect production's follow-up handling instead of a
    context-free reading of a pronoun with no antecedent."""
    state.reset()
    t0 = time.time()
    game_name = GAME_NAMES.get(game, game) if game else None
    history = [{"question": previous_question, "answer": ""}] if previous_question else None
    picked = route(question, cfg, history=history, game=game_name, http_post=http_post)
    tiered_cfg = apply_tier(cfg, picked.tier)
    applied_tier = tiered_cfg.get("answer_tier", picked.tier)
    qvec = _embed_query(_retrieval_query(question, history), cfg, embedder)
    g = _ground(question, cfg, tiered_cfg, applied_tier, picked, qvec, game, None, embedder, history,
               http_post=http_post)
    ms = round((time.time() - t0) * 1000, 1)

    topic_choice, topic_conf = router._choice(state.routing or {}, "topic", TOPICS)
    format_choice, format_conf = router._choice(state.routing or {}, "format", FORMATS)

    # Same guard _ground() uses to decide whether to call answerability.check() at all. It tells us
    # whether a check was attempted without re-running _ground's escalation logic, which is how
    # "never checked" is told apart from "checked, but the queue is empty because retrieval came back
    # with zero chunks" when check1/check2 are rebuilt below.
    check_attempted = _answerability_would_run(cfg, picked)

    check1 = _pop_coverage(state.coverage) if check_attempted else None
    check2 = _pop_coverage(state.coverage) if (check_attempted and g.escalated) else None

    return {
        "kind": kind,
        "question": question,
        "previous_question": previous_question,
        "game": game,
        "tier": applied_tier,
        "topic": {"choice": topic_choice, "confidence": topic_conf},
        "format": {"choice": format_choice, "confidence": format_conf},
        "check1": check1,
        "check2": check2,
        "escalated": g.escalated,
        "declined_0_7": g.declined,
        "ms": ms,
    }


# --------------------------------------------------------------------------------------------
# Pure analysis helpers (offline-testable: no network, operate on already-recorded records)
# --------------------------------------------------------------------------------------------

def rate(records: list, pred) -> float:
    """Fraction of ``records`` for which ``pred(record)`` is true. Returns 0.0 for an empty list
    (never divides by zero, never raises), so a threshold table prints even when one case kind was
    skipped (``--skip-gold`` / ``--skip-negatives``)."""
    if not records:
        return 0.0
    return sum(1 for r in records if pred(r)) / len(records)


def _finite_ge(confidence, threshold: float) -> bool:
    return confidence is not None and math.isfinite(confidence) and confidence >= threshold


def _topic_blocked_at(record: dict, threshold: float) -> bool:
    """Whether the off-topic gate would block ``record`` at ``threshold``: its raw topic choice is
    ``off_topic`` at or above that confidence (``router._topic``'s rule, replayed from the raw
    value)."""
    topic = record.get("topic") or {}
    return topic.get("choice") == "off_topic" and _finite_ge(topic.get("confidence"), threshold)


def _answerability_at(record: dict, threshold: float) -> tuple[bool, bool]:
    """Replay the escalate/decline decision for ``record`` at ``threshold``, returning
    ``(escalated, declined)``.

    Rule: if check1 is ``not_covered`` at or above ``threshold``, a record already at scholar depth
    is declined outright. Any other record is escalated, and declined only if check2 is ALSO
    ``not_covered`` at or above ``threshold``. If check1 does not clear the threshold, neither
    happens.

    This holds for every threshold in ``THRESHOLDS`` because the live run pins
    ``decline_confidence`` to 0.7, the lowest of them. Whenever check1 clears a threshold t >= 0.7 it
    also cleared 0.7, so escalation happened during the run and check2 was recorded;
    ``_pop_coverage`` never has to fabricate it on this branch. A ``record`` whose check1 is ``None``
    (no answerability check was attempted, see ``run_case``) always returns ``(False, False)``.

    Assumes all three answer tiers (fast/thinking/scholar) are configured. ``_ground`` declines on
    check1 alone, with no escalation, when there is no scholar tier to escalate to; this replay does
    not special-case that because the shipped config always configures all three."""
    check1 = record.get("check1")
    if not check1 or check1.get("verdict") != "not_covered" or not _finite_ge(check1.get("confidence"), threshold):
        return False, False
    if record.get("tier") == "scholar":
        return False, True
    check2 = record.get("check2") or {}
    declined = check2.get("verdict") == "not_covered" and _finite_ge(check2.get("confidence"), threshold)
    return True, declined


def sweep(records: list, conf_field: str, thresholds=THRESHOLDS) -> dict:
    """For each threshold, the per-record boolean decision for ``conf_field``: ``"topic"`` is
    off-topic-gate-blocked, ``"answerability"`` is answerability-gate-declined. Returns
    ``{threshold: [bool, ...]}`` aligned to ``records`` order. It is pure and offline (it replays
    already-recorded raw confidences, no extra calls), so a caller can ``rate()`` any slice of it."""
    if conf_field not in ("topic", "answerability"):
        raise ValueError(f"sweep: unknown conf_field {conf_field!r}")
    out = {}
    for t in thresholds:
        if conf_field == "topic":
            out[t] = [_topic_blocked_at(r, t) for r in records]
        else:
            out[t] = [_answerability_at(r, t)[1] for r in records]
    return out


def escalation_rate_at(records: list, threshold: float) -> float:
    return rate(records, lambda r: _answerability_at(r, threshold)[0])


def ship_threshold(gold_records: list, conf_field: str, thresholds=THRESHOLDS):
    """The lowest threshold whose GOLD false rate is <= 1% (2 of 200), else ``"disable"``. This is the
    ship rule documented in docs/ARCHITECTURE.md."""
    swept = sweep(gold_records, conf_field, thresholds)
    for t in thresholds:
        false_rate = sum(swept[t]) / len(gold_records) if gold_records else 0.0
        if false_rate <= SHIP_RULE_MAX_RATE:
            return t
    return "disable"


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------

def _format_distribution(records: list) -> dict:
    dist = {"table": 0, "list": 0, "prose": 0, "none": 0}
    for r in records:
        choice = (r.get("format") or {}).get("choice")
        dist[choice if choice in FORMATS else "none"] += 1
    return dist


def print_summary(records: list):
    gold = [r for r in records if r["kind"] == "gold"]
    off_topic = [r for r in records if r["kind"] == "off_topic"]
    not_covered = [r for r in records if r["kind"] == "not_covered"]
    # follow_up cases are answerable Xeno follow-ups, like gold, so a block or decline on one is a FALSE
    # positive in the same sense the gold columns report. They print next to gold using the same
    # _topic_blocked_at / _answerability_at replay, but stay OUT of ship_threshold below, which is
    # gold-based per spec (follow_up is a smaller, hand-written set).
    follow_up = [r for r in records if r["kind"] == "follow_up"]

    print(f"\n{'='*98}\nJEV GATES EVAL: {len(records)} cases "
          f"(gold {len(gold)}, off_topic {len(off_topic)}, not_covered {len(not_covered)}, "
          f"follow_up {len(follow_up)})\n{'='*98}")
    header = (f"{'thr':>5} {'gold false-block':>17} {'gold false-decline':>19} "
              f"{'follow_up false-block':>22} {'follow_up false-decline':>24} "
              f"{'off_topic catch':>16} {'not_covered decline':>20} {'escalation':>11}")
    print(header)
    for t in THRESHOLDS:
        gold_block = rate(gold, lambda r, t=t: _topic_blocked_at(r, t))
        gold_decline = rate(gold, lambda r, t=t: _answerability_at(r, t)[1])
        fu_block = rate(follow_up, lambda r, t=t: _topic_blocked_at(r, t))
        fu_decline = rate(follow_up, lambda r, t=t: _answerability_at(r, t)[1])
        catch = rate(off_topic, lambda r, t=t: _topic_blocked_at(r, t))
        nc_decline = rate(not_covered, lambda r, t=t: _answerability_at(r, t)[1])
        esc = escalation_rate_at(records, t)
        print(f"{t:>5.1f} {gold_block:>16.1%} {gold_decline:>19.1%} "
              f"{fu_block:>21.1%} {fu_decline:>23.1%} "
              f"{catch:>16.1%} {nc_decline:>20.1%} {esc:>11.1%}")

    print(f"\nformat distribution (raw, all cases): {_format_distribution(records)}")

    topic_ship = ship_threshold(gold, "topic")
    answerability_ship = ship_threshold(gold, "answerability")
    print(f"\nship rule (gold false rate <= {SHIP_RULE_MAX_RATE:.0%}):")
    print(f"  off_topic_gate       -> {'disable' if topic_ship == 'disable' else f'enable at {topic_ship}'}")
    print(f"  answerability_check  -> "
          f"{'disable' if answerability_ship == 'disable' else f'enable at {answerability_ship}'}")


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------

def _load_gold(limit: int | None) -> list:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))["questions"]
    if limit:
        gold = gold[:limit]
    return gold


def _load_cases() -> dict:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def run_cases(cases: list, cfg: dict, embedder, http_post, budget: CallBudget,
             state: _RecorderState, on_record=None) -> tuple[list, bool]:
    """Run every case in ``cases`` through ``run_case()``, printing a one-line progress log, and stop
    early if ``budget`` was exceeded mid-case. Returns ``(records, aborted)``.

    Each case is a ``(kind, question, game)`` triple, or a ``(kind, question, game,
    previous_question)`` 4-tuple for a ``"follow_up"`` case. Accepting both means the gold,
    off_topic and not_covered call sites do not need a ``previous_question`` slot they never use.

    This does NOT rely on ``BudgetExceeded`` propagating out of ``run_case()``, because it does not
    propagate: ``router._jev_call`` wraps every ``http_post`` call (including ours) in a broad
    ``except Exception`` and returns ``None`` on any failure, swallowing the budget's exception and
    letting routing and grounding continue on fallback data. Instead, ``budget.count`` is polled
    after each case. Once it has crossed ``max_calls``, that case's record (built from calls that ran
    over budget) is discarded and the loop stops.

    ``on_record(rec)``, when given, is called for each kept record as soon as it exists, so a caller
    can persist it immediately. A Ctrl-C or an unexpected error mid-run then loses nothing already
    paid for."""
    records = []
    aborted = False
    for i, case in enumerate(cases, 1):
        kind, question, game = case[0], case[1], case[2]
        previous_question = case[3] if len(case) > 3 else None
        rec = run_case(kind, question, game, cfg, embedder, http_post, state, previous_question)
        if budget.count > budget.max_calls:
            print(f"\n[ABORT] jev gates eval: exceeded its call budget ({budget.max_calls} calls). "
                 f"Discarding this case's record and writing {len(records)} collected records.")
            aborted = True
            break
        records.append(rec)
        if on_record is not None:
            on_record(rec)
        flag = "declined" if rec["declined_0_7"] else ("esc" if rec["escalated"] else "ok")
        print(f"[{i:3d}/{len(cases)}] [{kind:11s}] {flag:8s} {rec['ms']:6.0f}ms  {question[:60]}")
    return records, aborted


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N gold questions (negatives are unaffected)")
    ap.add_argument("--skip-gold", action="store_true", help="skip the 200-question gold set")
    ap.add_argument("--skip-negatives", action="store_true",
                    help="skip eval/jev_gates_cases.json (off_topic/not_covered)")
    ap.add_argument("--max-calls", type=int, default=1000, help="CallBudget cap (default 1000)")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    cfg = _eval_cfg(load_config())     # load_config() loads .env via the normal path
    if not jev_available(cfg):
        raise SystemExit("jev gates eval: router.provider must be 'jev' and TYPESAFE_API_KEY must be "
                         "set. This script makes live paid calls and refuses to run without a key.")

    embedder = Embedder(cfg)
    budget = CallBudget(args.max_calls)
    state = _RecorderState()
    http_post = budget.wrap(_make_recorder(router._default_post, state))

    cases = []
    if not args.skip_gold:
        cases += [("gold", q["question"], q["game"]) for q in _load_gold(args.limit)]
    if not args.skip_negatives:
        negatives = _load_cases()
        cases += [("off_topic", q, None) for q in negatives["off_topic"]]
        cases += [("not_covered", n["question"], n.get("game")) for n in negatives["not_covered"]]
        cases += [("follow_up", n["question"], n.get("game"), n.get("previous_question"))
                 for n in negatives["follow_up"]]

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        def _persist(rec):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
        records, aborted = run_cases(cases, cfg, embedder, http_post, budget, state, on_record=_persist)
    print(f"\nWrote {len(records)} records to {out_path} (Jev calls made: {budget.count})")

    if records:
        print_summary(records)
    if aborted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
