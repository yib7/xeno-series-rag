"""Per-question answer-tier routing via Jev, TypeSafe AI's decision model.

Each question is sorted into one of three tiers before retrieval. A tier fixes the Gemini model, its
thinking level, and how much of the wiki is read (``answer_tiers`` in config.yaml). Jev returns a
typed choice plus a confidence instead of generated text, so routing costs a fraction of a cent and
adds one short HTTP round-trip.

Routing must never break answering: no key, a timeout, any HTTP error, a malformed reply, or a
low-confidence choice all fall back to ``router.fallback_tier`` (default ``thinking``). There are no
retries, because the router has to stay cheaper than the retrieval work it controls.
"""

import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

TIERS = ("fast", "thinking", "scholar")
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


@dataclass(frozen=True)
class Route:
    tier: str
    source: str                      # "jev" | "fallback" | "override"
    confidence: float | None = None


def _router_cfg(cfg: dict) -> dict:
    return cfg.get("router") or {}


def fallback_tier(cfg: dict) -> str:
    """The configured fallback tier, or ``thinking`` when unset or not a known tier."""
    tier = _router_cfg(cfg).get("fallback_tier", DEFAULT_TIER)
    return tier if tier in TIERS else DEFAULT_TIER


def build_request(question: str, cfg: dict, history=None, game: str | None = None) -> dict:
    """The Jev ``systemone`` request body: one ``choice`` question over the three tiers. The state
    carries the previous question (so a terse follow-up routes like its topic) and the game scope."""
    state = {"question": (question or "")[:MAX_STATE_CHARS]}
    if history:
        prev = (history[-1] or {}).get("question")
        if prev:
            state["previous_question"] = str(prev)[:MAX_STATE_CHARS]
    if game:
        state["game"] = game
    return {
        "model": _router_cfg(cfg).get("model", "jev-latest"),
        "state": state,
        "questions": {"tier": {"type": "choice", "instructions": INSTRUCTIONS,
                               "criteria": dict(CRITERIA)}},
    }


def _default_post(url, *, json, headers, timeout):
    import httpx

    resp = httpx.post(url, json=json, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def route(question: str, cfg: dict, history=None, game: str | None = None, http_post=None) -> Route:
    """Pick the answer tier for ``question``. Offline (fallback, no HTTP call) unless
    ``router.provider`` is ``jev`` AND ``TYPESAFE_API_KEY`` is set. Never raises."""
    rc = _router_cfg(cfg)
    fallback = Route(fallback_tier(cfg), "fallback")
    if rc.get("provider", "fixed") != "jev":
        return fallback
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        return fallback
    post = http_post or _default_post
    try:
        data = post(rc.get("url", DEFAULT_URL),
                    json=build_request(question, cfg, history=history, game=game),
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=float(rc.get("timeout_seconds", 2)))
        answer = data["answers"]["tier"]
        choice = answer.get("choice")
        confidence = float(answer.get("confidence", 0.0))
    except Exception as exc:  # noqa: BLE001 - routing must never break answering
        # Log the error type and HTTP status only: never the key, headers, or request body.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        log.warning("router: Jev call failed (%s%s); using fallback tier %r",
                    type(exc).__name__, f" {status}" if status else "", fallback.tier)
        return fallback
    if choice not in TIERS or confidence < float(rc.get("min_confidence", 0.5)):
        log.info("router: Jev choice %r at confidence %.2f not usable; using fallback tier %r",
                 choice, confidence, fallback.tier)
        return Route(fallback.tier, "fallback", confidence)
    return Route(choice, "jev", confidence)


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
