"""Post-rerank answerability check: after retrieval, ask Jev whether the top chunks actually contain
what's needed to answer the question. Feeds ``rag.py``'s escalate-once-then-decline pipeline (see
``rag._ground``): a ``not_covered`` verdict at fast/thinking depth triggers one re-retrieve at scholar
depth, and a ``not_covered`` verdict that survives (or starts) at scholar depth means the wiki
genuinely doesn't seem to cover the question, so ``rag.py`` declines instead of asking Gemini to
generate over thin context.

Like ``router.route()``, ``check()`` never raises: any failure (no key, provider not ``jev``, a
timeout, a malformed reply) returns ``Verdict(None)`` and the caller proceeds as if the check had
never run.
"""

from dataclasses import dataclass

from . import retrieve, router

COVERAGE = ("answered", "partial", "not_covered")

COVERAGE_INSTRUCTIONS = "Do the retrieved wiki passages contain what is needed to answer the question?"
COVERAGE_CRITERIA = {
    "answered": "The passages state the facts needed to answer the question.",
    "partial": "The passages cover the topic and answer part of the question, or support a "
               "reasonable inference, but miss some detail.",
    "not_covered": "The passages are about other topics and do not contain the answer.",
}

# A passage line is cut here (word boundary + an ellipsis) so the routing request stays small and
# cheap even when a retrieved chunk runs long. 1500 (not a smaller, cheaper cut): the check must
# judge the SAME merged-page blocks Gemini ends up seeing (rag._ground merges fragmented stat-page
# chunks into one block, up to merge_max_chars=4000, before checking), not the raw one-line
# "Introduction: X is an enemy..." scraps retrieval returns -- those under-represent a merged block's
# actual coverage and produced false not_covered declines (e.g. "stats of the enemy P.S.S. - P").
_PASSAGE_CHARS = 1500


@dataclass(frozen=True)
class Verdict:
    """``verdict`` is one of ``COVERAGE`` or ``None`` when the check is unavailable (no key, a
    malformed reply, or any other failure) -- callers must treat ``None`` as "proceed normally, as if
    the check were never run", never as "not covered"."""
    verdict: str | None
    confidence: float | None = None


def passages(chunks, n: int) -> list[str]:
    """The top ``n`` chunks as ``"<title> (<game>): <text>"`` lines for the Jev request: the
    "[GAME] Title > " breadcrumb is stripped (``retrieve._strip_breadcrumb``, the same helper
    ``merge_fragmented_pages`` uses), whitespace is collapsed, and the text is cut to
    ``_PASSAGE_CHARS`` on a word boundary with a trailing "…" so one long chunk can't blow up the
    request."""
    out = []
    for c in chunks[:n]:
        text = " ".join(retrieve._strip_breadcrumb(c.get("text", "")).split())
        if len(text) > _PASSAGE_CHARS:
            cut = text[:_PASSAGE_CHARS].rsplit(" ", 1)[0].rstrip()
            text = (cut or text[:_PASSAGE_CHARS]) + "…"
        out.append(f"{c.get('title', '')} ({c.get('game', '')}): {text}")
    return out


def check(question: str, chunks: list[dict], cfg: dict, http_post=None, history=None) -> Verdict:
    """Ask Jev whether ``chunks`` (already reranked, best-first) cover ``question``. Empty chunks are
    trivially not covered -- retrieval found nothing at all -- so this short-circuits with
    ``Verdict("not_covered", 1.0)`` and makes no call. Any other failure (no key, provider not
    ``jev``, an HTTP error, a malformed reply) comes back as ``Verdict(None)`` via
    ``router._jev_call``, which never raises.

    ``history`` (same shape ``rag.py`` threads everywhere: a list of ``{"question", "answer"}``
    turns) supplies the antecedent for a follow-up question ("and what is her element?" has no
    referent on its own): when present, the previous turn's question is added as
    ``state["previous_question"]`` via ``router._previous_question`` -- the exact same trim/shape
    ``router._build_state`` uses for routing, so a follow-up is judged with the same context in both
    the tier/topic/format call and the coverage check."""
    if not chunks:
        return Verdict("not_covered", 1.0)
    rc = router._router_cfg(cfg)
    # Coerce and clamp: a config value can arrive as a float (8.0) or a string ("8") from YAML/JSON,
    # and `passages` slices a list with it (`chunks[:n]`), which raises on anything but an int.
    # `_safe_float` also absorbs outright garbage (falls back to the default 8); clamping to >= 1
    # guards a 0 or negative config value from asking Jev to judge zero passages.
    n = max(1, int(router._safe_float(rc.get("answerability_passages", 8), 8)))
    state = {"question": (question or "")[:router.MAX_STATE_CHARS], "passages": passages(chunks, n)}
    prev = router._previous_question(history)
    if prev:
        state["previous_question"] = prev
    questions = {
        "coverage": {"type": "choice", "instructions": COVERAGE_INSTRUCTIONS,
                    "criteria": dict(COVERAGE_CRITERIA)},
    }
    answers = router._jev_call(state, questions, cfg, http_post=http_post)
    choice, confidence = router._choice(answers, "coverage", COVERAGE)
    return Verdict(choice, confidence)


def is_not_covered(v: Verdict, cfg: dict) -> bool:
    """Whether ``v`` should trigger escalation/decline: a ``not_covered`` verdict at or above
    ``router.decline_confidence`` (default 0.8). ``Verdict(None)`` (check unavailable/failed) and a
    non-finite confidence both fail this -- a check that didn't run, or returned a confidence Jev
    left out, must never block or decline an answer."""
    if v.verdict != "not_covered" or v.confidence is None:
        return False
    threshold = router._safe_float(router._router_cfg(cfg).get("decline_confidence", 0.8), 0.8)
    return v.confidence >= threshold
