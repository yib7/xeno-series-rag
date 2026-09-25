"""Per-question routing via Jev, TypeSafe AI's decision model.

Each question is judged against a single ``systemone`` request carrying three independent
questions: ``tier`` (how much of the wiki to read and how hard to reason), ``topic`` (whether the
question is about the Xeno series at all), and ``format`` (which answer shape suits it). Jev returns
typed choices plus confidences instead of generated text, so routing costs a fraction of a cent and
adds one short HTTP round-trip.

Routing must never break answering: no key, a timeout, any HTTP error, a malformed reply, or a
low-confidence choice all fall back to today's behaviour. Each answer is parsed independently, so one
bad answer (e.g. a malformed ``topic``) never discards the others. There are no retries, because the
router has to stay cheaper than the retrieval work it controls.
"""

import logging
import math
import os
import threading
from dataclasses import dataclass

log = logging.getLogger(__name__)

TIERS = ("fast", "thinking", "scholar")
TOPICS = ("xeno", "off_topic")
FORMATS = ("table", "list", "prose")
DEFAULT_TIER = "thinking"
DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
MAX_STATE_CHARS = 2000   # a routing decision never needs more than the start of a question

INSTRUCTIONS = (
    "Decide how much of the Xeno Series Wiki an assistant must read, and how hard it must reason, "
    "to answer this question about the Xeno video game series well."
)
CRITERIA = {
    "fast": "A single factual lookup about one entity: a stat, level, location, drop, affinity, or "
            "who or what something is.",
    "thinking": "An explanation or comparison that combines a few topics, or follows one game's "
                "story arc or mechanic in depth.",
    "scholar": "Broad synthesis across many wiki pages or several games: series-wide themes, "
               "timelines, or connections between games.",
}

TOPIC_INSTRUCTIONS = "Is this message about the Xeno video game series?"
TOPIC_CRITERIA = {
    "xeno": "About the Xeno video game series (Xenogears, Xenosaga, Xenoblade Chronicles): its "
            "characters, places, story, lore, mechanics, items, enemies, music, or development -- "
            "including a short follow-up to the previous question.",
    "off_topic": "Not about the Xeno series: greetings, thanks, small talk, other games or media, "
                 "general knowledge, coding, or requests to write unrelated content.",
}

FORMAT_INSTRUCTIONS = "Which answer shape suits this question best?"
FORMAT_CRITERIA = {
    "table": "Compares numbers or attributes across several items, or asks for numeric data (stats, "
             "levels, drops, locations) for multiple entries.",
    "list": "Asks for a set of several items, members, steps, or requirements.",
    "prose": "Asks for an explanation, story, lore, motivation, or a single fact.",
}


@dataclass(frozen=True)
class Route:
    tier: str
    source: str                      # "jev" | "fallback" | "override"
    confidence: float | None = None
    topic: str | None = None         # "xeno" | "off_topic" | None (unusable/missing answer)
    format: str | None = None        # "table" | "list" | "prose" | None (unusable/missing answer)


def _router_cfg(cfg: dict) -> dict:
    rc = cfg.get("router")
    return rc if isinstance(rc, dict) else {}


def _safe_float(value, default: float) -> float:
    """Parse ``value`` as a float, falling back to ``default`` on a bad type or value (including
    non-finite results) so a malformed config value never raises out of route()."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def fallback_tier(cfg: dict) -> str:
    """The configured fallback tier, or ``thinking`` when unset or not a known tier."""
    tier = _router_cfg(cfg).get("fallback_tier", DEFAULT_TIER)
    return tier if tier in TIERS else DEFAULT_TIER


def jev_available(cfg: dict) -> bool:
    """Whether a Jev call would actually be made: provider is ``jev`` and a key is set. Forced-tier
    callers (CLI --tier, evals) use this to decide whether the override should still call Jev for
    ``topic``/``format``, or skip the call entirely."""
    rc = _router_cfg(cfg)
    return rc.get("provider", "fixed") == "jev" and bool(os.environ.get("TYPESAFE_API_KEY"))


def off_topic_gate_on(cfg: dict) -> bool:
    """Whether the off-topic short-circuit is enabled (default off, so an old config keeps today's
    behaviour of always retrieving and answering)."""
    return bool(_router_cfg(cfg).get("off_topic_gate", False))


def answerability_on(cfg: dict) -> bool:
    """Whether the post-rerank answerability check is enabled (default off)."""
    return bool(_router_cfg(cfg).get("answerability_check", False))


def _build_state(question: str, history=None, game: str | None = None) -> dict:
    state = {"question": (question or "")[:MAX_STATE_CHARS]}
    if history:
        prev = (history[-1] or {}).get("question")
        if prev:
            state["previous_question"] = str(prev)[:MAX_STATE_CHARS]
    if game:
        state["game"] = game
    return state


def _questions() -> dict:
    """A fresh copy of the three-question map every request sends, so no caller can mutate the
    module-level criteria dicts by editing a returned request body."""
    return {
        "tier": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict(CRITERIA)},
        "topic": {"type": "choice", "instructions": TOPIC_INSTRUCTIONS, "criteria": dict(TOPIC_CRITERIA)},
        "format": {"type": "choice", "instructions": FORMAT_INSTRUCTIONS, "criteria": dict(FORMAT_CRITERIA)},
    }


def build_request(question: str, cfg: dict, history=None, game: str | None = None) -> dict:
    """The Jev ``systemone`` request body: three independent ``choice`` questions (tier, topic,
    format) over one shared state. The state carries the previous question (so a terse follow-up
    routes like its topic) and the game scope."""
    return {
        "model": _router_cfg(cfg).get("model", "jev-latest"),
        "state": _build_state(question, history=history, game=game),
        "questions": _questions(),
    }


_client_lock = threading.Lock()
_client = None


def _get_client(transport=None):
    """Return the shared ``httpx.Client``, building it lazily on first use so routing never pays a
    new-connection TLS handshake per question. The lock makes creation safe under concurrent
    requests: two threads racing here only ever build one client. ``transport`` is test-only (it
    lets a test inject ``httpx.MockTransport`` instead of opening real sockets); production call
    sites never pass it, so the client is built once with the real transport and reused forever."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                import httpx

                _client = httpx.Client(transport=transport)
    return _client


def _reset_client_for_tests():
    """Test-only: drop the cached shared client so the next call rebuilds it, e.g. with a fresh mock
    transport."""
    global _client
    _client = None


def _default_post(url, *, json, headers, timeout):
    import httpx

    client = _get_client()
    resp = client.post(url, json=json, headers=headers, timeout=httpx.Timeout(timeout))
    resp.raise_for_status()
    return resp.json()


_missing_key_warned = False


def _jev_call(state: dict, questions: dict, cfg: dict, http_post=None) -> dict | None:
    """Shared HTTP/key/error handling for any Jev ``systemone`` request. Used by ``route()`` and by
    the answerability check (``xeno_rag/answerability.py``), so it takes any ``state``/``questions``
    pair rather than anything routing-specific.

    Returns ``data["answers"]`` (asserted to be a dict) or ``None`` when ``router.provider`` isn't
    ``jev``, no ``TYPESAFE_API_KEY`` is set (logged once per process, INFO), or the call raises for
    any reason -- bad status, timeout, connection error, or a malformed response (logged as exception
    type + HTTP status only, WARNING). Never logs the key, headers, or request body. Never raises.
    """
    global _missing_key_warned
    rc = _router_cfg(cfg)
    if rc.get("provider", "fixed") != "jev":
        return None
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        if not _missing_key_warned:
            _missing_key_warned = True
            log.info("router: TYPESAFE_API_KEY not set; every question uses the fallback tier")
        return None
    timeout = _safe_float(rc.get("timeout_seconds", 2), 2.0)
    post = http_post or _default_post
    try:
        data = post(rc.get("url", DEFAULT_URL),
                    json={"model": rc.get("model", "jev-latest"), "state": state, "questions": questions},
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=timeout)
        answers = data["answers"]
        if not isinstance(answers, dict):
            raise TypeError("Jev response 'answers' was not a dict")
        return answers
    except Exception as exc:  # noqa: BLE001 - a Jev call must never break the caller
        # Log the error type and HTTP status only: never the key, headers, or request body.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        log.warning("router: Jev call failed (%s%s); caller uses its fallback",
                    type(exc).__name__, f" {status}" if status else "")
        return None


def _choice(answers, qid: str, allowed) -> tuple[str | None, float | None]:
    """Parse one answer out of a Jev ``answers`` dict independently, never raising: a non-dict
    ``answers``, a missing or non-dict entry, or a choice outside ``allowed`` yields ``(None,
    None)``. A present, allowed choice keeps its confidence -- parsed leniently via ``_safe_float``,
    so a missing, non-numeric, or NaN confidence comes back as NaN rather than raising -- leaving the
    finite/threshold check to the caller (tier keeps a NaN confidence to explain the fallback; topic
    and format just treat a non-finite confidence as not meeting their threshold)."""
    if not isinstance(answers, dict):
        return None, None
    answer = answers.get(qid)
    if not isinstance(answer, dict):
        return None, None
    choice = answer.get("choice")
    if choice not in allowed:
        return None, None
    return choice, _safe_float(answer.get("confidence", 0.0), math.nan)


def _topic(answers: dict, off_topic_confidence: float) -> str | None:
    """``"off_topic"`` only when confidently so; a valid ``"xeno"`` answer always counts; anything
    else (missing, malformed, unknown, or off_topic below threshold) is treated as on-topic (None)."""
    choice, confidence = _choice(answers, "topic", TOPICS)
    if choice == "off_topic" and confidence >= off_topic_confidence:
        return "off_topic"
    if choice == "xeno":
        return "xeno"
    return None


def _format(answers: dict, min_confidence: float) -> str | None:
    """A valid choice at or above ``min_confidence``; otherwise no hint (None)."""
    choice, confidence = _choice(answers, "format", FORMATS)
    if choice is not None and confidence >= min_confidence:
        return choice
    return None


def route(question: str, cfg: dict, history=None, game: str | None = None, http_post=None,
          forced_tier: str | None = None) -> Route:
    """Pick the answer tier for ``question`` (unless ``forced_tier`` overrides it) and, when a Jev
    call is made, read its ``topic``/``format`` answers too. Offline (fallback, no HTTP call) unless
    ``router.provider`` is ``jev`` AND ``TYPESAFE_API_KEY`` is set -- except a valid ``forced_tier``
    without a key, which also skips the call entirely since nothing would use its answers. Never
    raises.
    """
    rc = _router_cfg(cfg)
    fallback = Route(fallback_tier(cfg), "fallback")
    forced = forced_tier if forced_tier in TIERS else None
    if forced and not jev_available(cfg):
        return Route(forced, "override")

    min_confidence = _safe_float(rc.get("min_confidence", 0.5), 0.5)
    off_topic_confidence = _safe_float(rc.get("off_topic_confidence", 0.8), 0.8)
    state = _build_state(question, history=history, game=game)
    answers = _jev_call(state, _questions(), cfg, http_post=http_post)
    if answers is None:
        return Route(forced, "override") if forced else fallback

    topic = _topic(answers, off_topic_confidence)
    fmt = _format(answers, min_confidence)
    if forced:
        return Route(forced, "override", topic=topic, format=fmt)

    tier_choice, tier_confidence = _choice(answers, "tier", TIERS)
    if (tier_choice is None or tier_confidence is None or not math.isfinite(tier_confidence)
            or tier_confidence < min_confidence):
        log.info("router: Jev tier choice %r at confidence %s not usable; using fallback tier %r",
                 tier_choice, tier_confidence, fallback.tier)
        return Route(fallback.tier, "fallback", tier_confidence, topic=topic, format=fmt)
    return Route(tier_choice, "jev", tier_confidence, topic=topic, format=fmt)


def apply_tier(cfg: dict, tier: str) -> dict:
    """Return a new cfg with ``tier``'s settings merged over the base: retrieval depth keys as-is,
    ``model`` mapped to ``gemini_model``, and ``answer_tier`` recording the tier used. An unknown
    tier uses the fallback tier's entry. Without an ``answer_tiers`` map the cfg is returned
    unchanged (same object). Never mutates the input."""
    tiers = cfg.get("answer_tiers") or {}
    if not tiers:
        return cfg
    name = tier if tier in tiers else fallback_tier(cfg)
    entry = dict(tiers.get(name) or {})
    model = entry.pop("model", None)
    merged = {**cfg, **entry, "answer_tier": name}
    if model:
        merged["gemini_model"] = model
    return merged
