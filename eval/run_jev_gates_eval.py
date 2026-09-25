"""Live-eval the Jev off-topic and answerability gates against the gold set and hand-written
negative sets (eval/jev_gates_cases.json), and sweep the ship-rule threshold from one pass.

Makes live Jev (TypeSafe) calls -- never Gemini. This is the ONLY script in the repo that is meant
to spend real money; everything else in the test suite stubs ``http_post``. Budget-guarded via
``CallBudget`` (default 1,000 calls, ~$0.20 upper bound at <=$0.0002/call): the eval aborts (and
still writes whatever it collected) rather than run away.

Design (see docs/superpowers/specs/2026-09-25-jev-gates-design.md §6 and .autopilot/PLAN.md SP4):

- Uses PRODUCTION code paths for routing and grounding -- ``router.route()`` and ``rag._ground()``
  (which calls ``answerability.check()``) -- rather than reimplementing the off-topic/escalate/
  decline decisions here. The only local logic is recomputing those decisions at swept thresholds
  from RAW recorded confidences (no extra calls), because ``Route.topic``/``Route.format`` already
  hide a choice below their configured threshold, and because the escalate/decline decision itself
  needs to be replayed at 0.7/0.8/0.9 without re-running the pipeline three times.
- The run itself forces ``router.decline_confidence`` to 0.7 (the lowest sweep threshold) and
  ``answerability_check: True`` / ``off_topic_gate: False`` in a COPY of the loaded config, so (a)
  every case actually runs the answerability check (needed to record check1/check2 raw verdicts even
  for off_topic/not_covered cases) and (b) escalation itself -- which genuinely re-retrieves and
  re-checks -- happens at the lowest threshold, so a higher sweep threshold's escalate/decline replay
  always has check2 available whenever it needs it (see ``_answerability_at``'s docstring).
- A single ``http_post`` callable (``CallBudget.wrap`` around a small recorder around the real
  transport) is passed into BOTH ``route()`` and (via ``rag._ground``'s ``http_post=`` seam)
  ``answerability.check()``, so one counter covers the whole run and the raw Jev ``answers`` for
  every call -- routing and coverage alike -- are captured before any threshold discards them.

Usage:
  python -m eval.run_jev_gates_eval                       # full run: 200 gold + 30 + 20 negatives
  python -m eval.run_jev_gates_eval --limit 20             # smoke run: first 20 gold questions only
  python -m eval.run_jev_gates_eval --skip-gold            # negatives only
  python -m eval.run_jev_gates_eval --skip-negatives       # gold only
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

# The lowest of the sweep thresholds: the run itself declines/escalates at this threshold (see
# module docstring), so every higher threshold's replay has whatever it needs already recorded.
RUN_DECLINE_CONFIDENCE = 0.7
THRESHOLDS = (0.7, 0.8, 0.9)
# Ship rule (spec §6): a gate ships enabled only if its gold false rate is <= 1% (<= 2/200 questions).
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
    (including the one that raised, if any) -- a live run can report exactly how much it spent."""

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
# Recording http_post: captures the RAW Jev answers dict from every call, tagged by request shape
# --------------------------------------------------------------------------------------------

class _RecorderState:
    """Per-case scratch space the recording ``http_post`` writes into. ``reset()`` before each case
    so a case's record only reflects calls made while processing that case."""

    def __init__(self):
        self.routing = None     # raw `answers` dict from the one tier/topic/format call, or None
        self.coverage = []      # raw `answers` dicts from each coverage call, in call order (0-2)

    def reset(self):
        self.routing = None
        self.coverage = []


def _make_recorder(real_post, state: _RecorderState):
    """Wrap ``real_post`` (the actual HTTP transport) so every call's raw ``answers`` dict is kept on
    ``state``, tagged as the routing call or a coverage call by the request's ``questions`` key --
    NOT by re-deriving it from route()/check()'s own (threshold-filtered) return values. This is what
    makes the raw topic/format confidence available for the sweep even when it's below
    ``off_topic_confidence``/``min_confidence`` (``Route`` would otherwise report ``None``)."""

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
    and ``off_topic_gate: False`` (so every case runs grounding uniformly, regardless of topic --
    the off-topic short-circuit itself is never exercised here, only its raw topic confidence is
    recorded) and ``decline_confidence`` pinned to the lowest sweep threshold (see module docstring).
    Never mutates the input."""
    router_cfg = dict(cfg.get("router") or {})
    router_cfg["answerability_check"] = True
    router_cfg["off_topic_gate"] = False
    router_cfg["decline_confidence"] = RUN_DECLINE_CONFIDENCE
    return {**cfg, "router": router_cfg}


# --------------------------------------------------------------------------------------------
# One case: route() + production grounding, recording raw topic/format/coverage
# --------------------------------------------------------------------------------------------

def _pop_coverage(queue: list) -> dict:
    """Pop the next recorded coverage call, or -- when the queue is exhausted because
    ``answerability.check()`` short-circuited on empty retrieval (see its docstring: ``Verdict(
    "not_covered", 1.0)`` with no call made) -- return that same documented constant. Called only
    when a check is known to have actually been attempted (``_run_case`` guards that separately), so
    an empty queue here can only mean the empty-chunks short-circuit, never "never checked"."""
    if queue:
        answers = queue.pop(0)
        choice, confidence = router._choice(answers, "coverage", COVERAGE)
        return {"verdict": choice, "confidence": confidence}
    return {"verdict": "not_covered", "confidence": 1.0}


def run_case(kind: str, question: str, game: str | None, cfg: dict, embedder, http_post,
            state: _RecorderState) -> dict:
    """Route + ground one case through production code paths, returning its JSONL record. ``cfg`` is
    already the forced eval cfg (see ``_eval_cfg``)."""
    state.reset()
    t0 = time.time()
    game_name = GAME_NAMES.get(game, game) if game else None
    picked = route(question, cfg, history=None, game=game_name, http_post=http_post)
    tiered_cfg = apply_tier(cfg, picked.tier)
    applied_tier = tiered_cfg.get("answer_tier", picked.tier)
    qvec = _embed_query(_retrieval_query(question), cfg, embedder)
    g = _ground(question, cfg, tiered_cfg, applied_tier, picked, qvec, game, None, embedder, None,
               http_post=http_post)
    ms = round((time.time() - t0) * 1000, 1)

    topic_choice, topic_conf = router._choice(state.routing or {}, "topic", TOPICS)
    format_choice, format_conf = router._choice(state.routing or {}, "format", FORMATS)

    # Same guard _ground() itself uses to decide whether to call answerability.check() at all, so this
    # knows, without re-running any of _ground's escalation logic, whether a check was even attempted
    # this case -- needed to tell "never checked" apart from "checked, but the queue is empty because
    # retrieval came back with zero chunks" when reconstructing check1/check2 below.
    check_attempted = _answerability_would_run(cfg, picked)

    check1 = _pop_coverage(state.coverage) if check_attempted else None
    check2 = _pop_coverage(state.coverage) if (check_attempted and g.escalated) else None

    return {
        "kind": kind,
        "question": question,
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
    """Fraction of ``records`` for which ``pred(record)`` is true. 0.0 for an empty list (never
    divides by zero, never raises) so a threshold table can always be printed even when one case
    kind was skipped (``--skip-gold`` / ``--skip-negatives``)."""
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

    Rule (spec's SP4 design requirements): if check1 is ``not_covered`` at or above ``threshold``:
    already at scholar depth -> declined outright; otherwise -> escalated, and declined only if
    check2 is ALSO ``not_covered`` at or above ``threshold``. Otherwise (check1 doesn't clear the
    threshold) -> neither.

    This is sound for every threshold in ``THRESHOLDS`` because the live run pins
    ``decline_confidence`` to 0.7 (the lowest of them): whenever check1 clears a threshold t >= 0.7,
    it also cleared 0.7, so escalation genuinely happened during the run and check2 was actually
    recorded -- ``_pop_coverage`` never had to fabricate it for this branch. A ``record`` whose check1
    is ``None`` (the answerability check was never attempted for this case -- see ``run_case``) always
    returns ``(False, False)``.

    Assumes all three answer tiers (fast/thinking/scholar) are configured: ``_ground`` itself declines
    on check1 alone, with no escalation, whenever there's no scholar tier to escalate to -- this replay
    doesn't special-case that, since the shipped config always configures all three."""
    check1 = record.get("check1")
    if not check1 or check1.get("verdict") != "not_covered" or not _finite_ge(check1.get("confidence"), threshold):
        return False, False
    if record.get("tier") == "scholar":
        return False, True
    check2 = record.get("check2") or {}
    declined = check2.get("verdict") == "not_covered" and _finite_ge(check2.get("confidence"), threshold)
    return True, declined


def sweep(records: list, conf_field: str, thresholds=THRESHOLDS) -> dict:
    """For each threshold, the per-record boolean decision for ``conf_field``:
    ``"topic"`` -> off-topic-gate-blocked; ``"answerability"`` -> answerability-gate-declined.
    Returns ``{threshold: [bool, ...]}`` aligned to ``records`` order -- pure and offline (replays
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
    """The lowest threshold whose GOLD false rate is <= 1% (<= 2/200), else ``"disable"`` -- the ship
    rule from spec §6 / PLAN.md SP4."""
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

    print(f"\n{'='*78}\nJEV GATES EVAL -- {len(records)} cases "
          f"(gold {len(gold)}, off_topic {len(off_topic)}, not_covered {len(not_covered)})\n{'='*78}")
    header = (f"{'thr':>5} {'gold false-block':>17} {'gold false-decline':>19} "
              f"{'off_topic catch':>16} {'not_covered decline':>20} {'escalation':>11}")
    print(header)
    for t in THRESHOLDS:
        gold_block = rate(gold, lambda r, t=t: _topic_blocked_at(r, t))
        gold_decline = rate(gold, lambda r, t=t: _answerability_at(r, t)[1])
        catch = rate(off_topic, lambda r, t=t: _topic_blocked_at(r, t))
        nc_decline = rate(not_covered, lambda r, t=t: _answerability_at(r, t)[1])
        esc = escalation_rate_at(records, t)
        print(f"{t:>5.1f} {gold_block:>16.1%} {gold_decline:>19.1%} "
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
             state: _RecorderState) -> tuple[list, bool]:
    """Run every ``(kind, question, game)`` in ``cases`` through ``run_case()``, printing a one-line
    progress log, and stop early if ``budget`` was exceeded mid-case. Returns ``(records, aborted)``.

    Does NOT rely on ``BudgetExceeded`` propagating out of ``run_case()`` -- it doesn't:
    ``router._jev_call`` wraps every ``http_post`` call (including ours) in a broad ``except
    Exception`` and returns ``None`` on any failure, silently swallowing the budget's exception and
    letting routing/grounding continue on corrupted fallback data. Instead, ``budget.count`` is polled
    directly after each case: once it has crossed ``max_calls``, that case's record (built from calls
    that ran over budget) is discarded and the loop stops."""
    records = []
    aborted = False
    for i, (kind, question, game) in enumerate(cases, 1):
        rec = run_case(kind, question, game, cfg, embedder, http_post, state)
        if budget.count > budget.max_calls:
            print(f"\n[ABORT] jev gates eval: exceeded its call budget ({budget.max_calls} calls) "
                 f"-- discarding this case's record and writing {len(records)} collected records.")
            aborted = True
            break
        records.append(rec)
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
                         "set -- this script makes live paid calls and refuses to run without a key.")

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

    out_path = Path(args.out)
    records, aborted = run_cases(cases, cfg, embedder, http_post, budget, state)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(records)} records to {out_path} (Jev calls made: {budget.count})")

    if records:
        print_summary(records)
    if aborted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
