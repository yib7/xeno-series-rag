"""Retrieval-augmented generation: retrieve grounded context, prompt an LLM, cite sources.

The LLM is pluggable. `GeminiClient` is the default (reads credentials from the environment at call
time and never embeds a key); `MockLLM` is used in tests. Generation is grounded: the system prompt
forbids inventing facts and requires citing sources.
"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import answerability, embed_index
from .config import load_config
from .errors import SetupError
from .retrieve import merge_fragmented_pages, retrieve
from .router import TIERS, Route, answerability_on, apply_tier, jev_available, off_topic_gate_on, route

log = logging.getLogger(__name__)

# Current default Gemini model when a config omits `gemini_model` (never a retired id like 1.5-flash).
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
# Shown when the model returns nothing usable (e.g. a safety block) so the UI never goes blank.
EMPTY_ANSWER_FALLBACK = (
    "I could not generate an answer for that. Try rephrasing the question or narrowing it to a "
    "specific game."
)
NO_QUESTION_MESSAGE = "Please enter a question to ask about the Xeno series."
# Shown when the off-topic gate fires (router.off_topic_gate). The gate skips the embedding wait,
# retrieval, rerank, and the Gemini call for a question that isn't about the Xeno series.
OFF_TOPIC_MESSAGE = (
    "I can only help with the Xeno series (Xenogears, Xenosaga, and Xenoblade Chronicles). Ask me "
    "about a character, place, story event, or game mechanic."
)
# Shown when the post-rerank answerability check (router.answerability_check) finds that the
# retrieved chunks still don't cover the question, after one escalation to the scholar tier if that
# was possible. No Gemini call is made, but the closest sources are still returned so the user can
# rephrase or narrow the game.
NOT_COVERED_MESSAGE = (
    "The wiki pages I found don't seem to cover that. The closest matches are listed below. Try "
    "rephrasing, or pick a specific game."
)

# One line steering the answer shape, added to the prompt when Jev's `format` answer is usable
# (build_prompt inserts FORMAT_LINES[answer_format] immediately before "Question:"). Keyed by the
# router.FORMATS choices; an unknown or None format adds no line.
FORMAT_LINES = {
    "table": "Format: answer with a compact Markdown table, plus at most one short sentence.",
    "list": "Format: answer with a short Markdown bullet list.",
    "prose": "Format: answer in short prose paragraphs; no table.",
}

# Runs the query embedding concurrently with the Jev routing HTTP call (both start before either
# finishes). Each call embeds and routes one question, so 2 workers is enough, and a small fixed pool
# avoids spawning a thread per request. The thread name makes these easy to find in thread dumps.
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="xeno-embed")


def _preflight(cfg: dict) -> None:
    """The cheap checks a live (non-injected-LLM) answer needs: a Gemini key and a vector store.
    Run before the query embed is submitted or Jev is called, so a first-run user gets one clear
    message instead of a paid routing call (or a model load) followed by the same failure."""
    require_gemini_key()
    embed_index.require_store(cfg)


def _embed_query(text, cfg, embedder=None):
    """Embed ``text`` with ``embedder`` if given, else the cached singleton. A thin, monkeypatchable
    seam so tests can stub out the real Qwen model when they submit this to ``_EXECUTOR``."""
    if embedder is None:
        embed_index.require_store(cfg)     # no store: say so now, not after the ~1.2 GB model loads
        embedder = embed_index._get_embedder(cfg)
    return embedder.embed_query(text)


SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the Xeno video game series "
    "(Xenogears, Xenosaga, Xenoblade Chronicles). "
    "Base your answer only on the provided context: do not invent facts, mechanics, or numbers "
    "the context does not support. But DO reason over the context to work out the answer: count or "
    "total items, take the maximum (e.g. if the highest chapter shown is 17, there are at least 17 "
    "chapters), compare, and combine facts across the retrieved sources. "
    "If the context only partially answers the question, give the best-supported answer you can and "
    "briefly note what is missing or uncertain, instead of just saying you do not know. Make clear "
    "what the context states versus what you reasonably infer. "
    "Prefer the structured infobox entries for stats and numeric questions. "
    "Cite as you go: the context blocks are numbered like [1], [2]. After each claim or stat, "
    "add the bracketed number(s) of the block(s) supporting it, e.g. 'It deals 250 damage [2].' "
    "Use only numbers that appear in the context. "
    "Do not list, cite, or restate the source URLs anywhere in your answer. The interface "
    "displays the sources separately and links your bracketed markers to them, so just write the "
    "answer prose with the markers. "
    "Format the answer as clean, concise Markdown: lead with the answer (no preamble), use short "
    "paragraphs or bullet lists, and use a Markdown table when comparing several numeric stats "
    "across multiple items (e.g. arts, characters, or enemies)."
)

# Display names for the game-scope note added to a filtered prompt.
GAME_NAMES = {
    "XG": "Xenogears",
    "XS1": "Xenosaga Episode I", "XS2": "Xenosaga Episode II", "XS3": "Xenosaga Episode III",
    "XC1": "Xenoblade Chronicles", "XC2": "Xenoblade Chronicles 2", "XC3": "Xenoblade Chronicles 3",
    "XCX": "Xenoblade Chronicles X",
}


class MockLLM:
    """Deterministic stand-in for tests. Records the last call."""

    def __init__(self, canned: str = "MOCK ANSWER"):
        self.canned = canned
        self.last_system = None
        self.last_prompt = None

    def generate(self, system: str, prompt: str) -> str:
        self.last_system = system
        self.last_prompt = prompt
        return self.canned

    def generate_stream(self, system: str, prompt: str):
        """Yield the canned answer in a few slices so streaming behaviour is exercised in tests."""
        self.last_system = system
        self.last_prompt = prompt
        text = self.canned or ""
        for i in range(0, len(text), 16):
            yield text[i:i + 16]


def _extract_text(resp) -> str:
    """Best-effort text from a google-genai response, never raising.

    `resp.text` raises (or is None) when the chosen candidate has no text Part, e.g. a safety
    block, or a "Thinking" model that returned only thought parts. Fall back to concatenating the
    candidate parts, then to an empty string. The caller substitutes a friendly message on ''.
    """
    try:
        text = resp.text
    except Exception:  # noqa: BLE001 - any SDK error: try the parts, then give up cleanly
        text = None
    if text and text.strip():
        return text
    try:
        for cand in (getattr(resp, "candidates", None) or []):
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            joined = "".join((getattr(p, "text", "") or "") for p in parts)
            if joined.strip():
                return joined
    except Exception:  # noqa: BLE001, S110 - best-effort fallback extraction, "" is the correct give-up
        pass
    return ""


def require_gemini_key() -> str:
    """The Gemini API key from the environment, or a ``SetupError`` naming the variables to set.
    ``answer()``/``answer_stream()`` call this up front when they will build a ``GeminiClient``, so a
    missing key fails before any embedding work or paid Jev routing call is spent on the question."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SetupError(
            "Gemini credentials not found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment "
            "or in a .env file at the repo root to enable live answers (see .env.example)."
        )
    return key


class GeminiClient:
    """Gemini adapter using the supported `google-genai` SDK.

    Reads GOOGLE_API_KEY (or GEMINI_API_KEY) at call time. Raises if absent so the autonomous
    pipeline never silently makes a billable call without credentials.
    """

    def __init__(self, cfg: dict):
        self.model = cfg.get("gemini_model") or DEFAULT_GEMINI_MODEL
        self.thinking_level = cfg.get("thinking_level")

    def _client_and_types(self):
        key = require_gemini_key()
        from google import genai
        from google.genai import types

        return genai.Client(api_key=key), types

    def _gen_config(self, types, system: str):
        """Generation config: the system prompt, plus a thinking level when the tier sets one."""
        kwargs = {"system_instruction": system}
        if self.thinking_level:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=str(self.thinking_level).upper())
        return types.GenerateContentConfig(**kwargs)

    def generate(self, system: str, prompt: str) -> str:
        client, types = self._client_and_types()
        resp = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._gen_config(types, system),
        )
        return _extract_text(resp)

    def generate_stream(self, system: str, prompt: str):
        """Yield text deltas as the model produces them (lower time-to-first-token)."""
        client, types = self._client_and_types()
        stream = client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=self._gen_config(types, system),
        )
        for chunk in stream:
            piece = _extract_text(chunk)
            if piece:
                yield piece


def _retrieval_query(question: str, history=None) -> str:
    """The text used for retrieval. For a follow-up, prepend the previous user question so a pronoun
    ("what is HER element?") still pulls the right page (the antecedent isn't in the new question)."""
    if history:
        last = history[-1]
        prev = str(last.get("question") or "").strip() if isinstance(last, dict) else ""
        if prev:
            return f"{prev} {question}"
    return question


def _history_block(history) -> str:
    """Render up to the last 6 turns as a Q/A transcript for the prompt, with each answer in full.
    The window is bounded because every turn is re-grounded on fresh retrieval, so older turns add
    mostly topic bleed and little signal as a chat grows. The retrieval query borrows only the single
    previous question (_retrieval_query), so session length never affects retrieval. Answers are not
    clipped: history is plain Q/A text with no retrieved chunk data, so 6 turns cost at most six
    model-length answers, and a clipped answer could hide a detail a follow-up depends on."""
    if not history:
        return ""
    lines = []
    for turn in history[-6:]:
        if not isinstance(turn, dict):        # the web layer validates; programmatic callers may not
            continue
        q = str(turn.get("question") or "").strip()
        a = str(turn.get("answer") or "").strip()
        if q:
            lines.append(f"Q: {q}")
        if a:
            lines.append(f"A: {a}")
    if not lines:
        return ""
    return "Earlier in this conversation:\n" + "\n".join(lines) + "\n\n"


def _source_numbers(chunks):
    """Map each distinct source URL to its 1-based citation number, in first-seen chunk order.

    This uses the same ordering rule as ``_dedupe_sources`` (the first occurrence of each url wins),
    so a bracketed [n] the model emits always points at the n-th card in the sources payload the UI
    renders. The numbering and the SSE sources list cannot drift apart. The order also survives
    ``merge_fragmented_pages``: merging replaces a page's chunks with one block at the first
    occurrence's position and keeps its url, so first-seen url order is unchanged."""
    order = {}
    for c in chunks:
        url = c.get("url")
        if url and url not in order:
            order[url] = len(order) + 1
    return order


def build_prompt(question: str, chunks, game_filter: str | None = None, history=None,
                 answer_format: str | None = None):
    """Return (system, user) prompt strings grounding the answer in the retrieved chunks.

    Each context block is prefixed with the bracketed number of its source page ([1]..[n], numbered
    by ``_source_numbers`` so they match the UI's sources list), and the prompt tells the model to
    cite each claim with those markers. When a game filter is active, a scope line names the game
    the user is focused on so the model resolves ambiguous names within it (e.g. "Jin" is the XC2
    Flesh Eater under XC2). Prior conversation turns (history) are included so follow-up questions
    resolve against them. ``answer_format`` (Jev's routed `table`/`list`/`prose` choice) inserts one
    verbatim line from ``FORMAT_LINES`` immediately before "Question:"; ``None`` or an unknown value
    adds nothing."""
    numbers = _source_numbers(chunks)
    blocks = []
    for c in chunks:
        n = numbers.get(c.get("url"))
        label = f"[{n}] " if n else ""
        blocks.append(f"{label}[{c['title']} ({c['game']})] {c['text']}\nSource: {c['url']}")
    context = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    scope = ""
    if game_filter:
        name = GAME_NAMES.get(game_filter, game_filter)
        scope = (f"The user is focused on {name} ({game_filter}); the context is filtered to that "
                 f"game, so resolve names and terms within it.\n\n")
    cite = ""
    if numbers:
        top = max(numbers.values())
        cite = (f"Cite inline: mark each claim with the bracketed number(s) [1]–[{top}] of the "
                "supporting context block(s) above, e.g. [2] or [1][3]. Never invent a number.\n\n")
    fmt = f"{FORMAT_LINES[answer_format]}\n\n" if answer_format in FORMAT_LINES else ""
    convo = _history_block(history)
    user = f"{convo}{scope}Context:\n{context}\n\n{cite}{fmt}Question: {question}"
    return SYSTEM_PROMPT, user


_BREADCRUMB = re.compile(r"^\[[^\]]*\][^:]*:\s*")


def _snippet(text: str, limit: int = 200) -> str:
    """A short, readable preview of a chunk: drop the "[GAME] Title > Heading: " breadcrumb prefix,
    collapse whitespace, truncate on a word boundary."""
    t = (text or "").strip()
    t = _BREADCRUMB.sub("", t)
    t = " ".join(t.split())
    if len(t) <= limit:
        return t
    cut = t[:limit].rsplit(" ", 1)[0].rstrip()
    return (cut or t[:limit]) + "…"


def _tier(relevance: float) -> str:
    """Bucket a 0-1 relevance into a high, mid or low tier (the UI marks low-tier source cards)."""
    if relevance >= 0.66:
        return "high"
    if relevance >= 0.33:
        return "med"
    return "low"


def _score_relevance(sources):
    """Annotate each source with a 0-1 ``relevance`` and a size ``tier`` in place.

    The list arrives already ordered best-first (the cross-encoder ``_score`` and fusion set the
    order), so ``relevance`` comes from rank position: the top source gets 1.0 and the bottom gets
    0.0. Rank drives the tier instead of the raw score because cross-encoder scores often cluster (a
    dozen near-equal pages), and min-max normalizing them would put every source in one tier. Rank
    gives a visible gradient, in clean thirds, for any set. The ``_score`` is dropped from the
    payload here, since it has already done its job of ordering."""
    n = len(sources)
    for i, s in enumerate(sources):
        s.pop("_score", None)
        rel = 1.0 if n <= 1 else (n - 1 - i) / (n - 1)
        s["relevance"] = round(rel, 4)
        s["tier"] = _tier(rel)
    return sources


def _dedupe_sources(chunks):
    """Deduped, ordered source list, one rich dict per cited page:
    ``{url, title, game, snippet, relevance, tier}``. The snippet is the first retrieved chunk's
    preview (best-ranked chunk for that page); ``relevance``/``tier`` say how
    well the page matches the question (see ``_score_relevance``)."""
    seen = set()
    sources = []
    for c in chunks:
        url = c.get("url")
        if url and url not in seen:
            seen.add(url)
            sources.append({
                "url": url,
                "title": c.get("title") or url,
                "game": c.get("game") or "",
                "snippet": _snippet(c.get("text")),
                "_score": c.get("_score"),
            })
    return _score_relevance(sources)


def _pick_route(question: str, cfg: dict, tier: str | None, history, game_filter) -> Route:
    """Route the question. An explicit known ``tier`` (CLI --tier, evals) is passed through as
    ``forced_tier``: ``route()`` then skips its own tier choice (``Route.source == "override"``) but
    still calls Jev for ``topic``/``format`` when a key is set, so a forced tier keeps the off-topic
    gate and the format hint."""
    game = GAME_NAMES.get(game_filter, game_filter) if game_filter else None
    return route(question, cfg, history=history, game=game, forced_tier=tier if tier in TIERS else None)


def _routed(question: str, cfg: dict, tier: str | None, history,
           game_filter) -> tuple[dict, str, str, Route]:
    """Route the question and apply the picked tier to cfg, returning (tiered_cfg, applied_tier,
    source, picked). ``apply_tier`` falls back to the fallback tier's entry when ``answer_tiers``
    lacks the picked tier, so the reported tier is the one actually applied. Reporting the picked
    tier instead would claim, in the result and the SSE event, a tier retrieval never used. ``picked``
    (the full ``Route``) is returned too so callers can read ``topic`` (off-topic gate) and
    ``format`` (the prompt's format hint)."""
    picked = _pick_route(question, cfg, tier, history, game_filter)
    tiered_cfg = apply_tier(cfg, picked.tier)
    applied_tier = tiered_cfg.get("answer_tier", picked.tier)
    return tiered_cfg, applied_tier, picked.source, picked


@dataclass
class Grounding:
    """The chunks, cfg, and tier a final answer is generated from, after the answerability check has
    possibly escalated retrieval to scholar depth. ``cfg``, ``tier``, and ``chunks`` are the final
    ones to use (post-escalation when ``escalated``); ``escalated`` and ``declined`` tell
    ``answer()`` and ``answer_stream()`` which extra SSE event or dict field to add.

    ``prompt_chunks`` is ``chunks`` after ``merge_fragmented_pages``, the same merged blocks the
    answerability check judged (see ``_ground``). Callers build the generation prompt from it, not
    from ``chunks``, so the check and Gemini see identical text. ``chunks`` (the raw, unmerged
    retrieval) is the source for ``_dedupe_sources``."""
    cfg: dict
    tier: str
    chunks: list
    prompt_chunks: list
    escalated: bool
    declined: bool


def _answerability_would_run(cfg: dict, picked: Route) -> bool:
    """Whether ``_ground()`` would invoke the answerability check for a route picked as ``picked``.
    All of these must hold:

    - ``router.answerability_check`` is on.
    - The tier wasn't forced (``picked.source != "override"``). A forced tier skips the check.
    - A Jev key is set (``jev_available``). Without one, an empty retrieval's trivial
      ``Verdict("not_covered", 1.0)`` would escalate and decline off a check that never ran.
    - Routing didn't fall all the way back to the fallback tier with no confidence at all
      (``picked.source == "fallback" and picked.confidence is None``).

    The last guard covers two situations that both skip the check. Either Jev is unreachable (the
    whole call failed, so paying the check's own timeout against a down provider is pointless), or
    Jev answered but with an invalid ``tier`` choice (``router._choice`` collapses a missing or
    unknown choice to ``(None, None)``, so ``confidence`` reads ``None`` although the call
    succeeded). Skipping in the second case costs a more conservative answer, not a wrong one. A
    route that picked a low-confidence tier (source ``"fallback"`` with a non-``None`` confidence)
    still runs the check. ``_ground()`` and the gate eval script (``eval/run_jev_gates_eval.py``)
    share this function, so the script can tell in advance whether a case's check1 and check2 were
    attempted without duplicating the logic."""
    routing_failed = picked.source == "fallback" and picked.confidence is None
    return (answerability_on(cfg) and picked.source != "override" and jev_available(cfg)
           and not routing_failed)


def _ground(question: str, base_cfg: dict, tiered_cfg: dict, tier: str, picked: Route, qvec,
           game_filter, k, embedder, history, http_post=None) -> Grounding:
    """Retrieve at the routed tier, then, only when ``_answerability_would_run(base_cfg, picked)``
    says so, verify that the merged, reranked chunks cover the question.

    The check must judge the same text Gemini sees. ``merge_fragmented_pages`` folds a stat page's
    fragmented factblock chunks (one-line "Introduction: X is an enemy..." scraps) into one coherent
    profile block, and that merge runs here, before ``check()``, on every retrieval. Checking the
    raw chunks would under-represent a merged block's coverage and produce false ``not_covered``
    declines. The merged result is kept on ``Grounding.prompt_chunks`` so ``answer()`` and
    ``answer_stream()`` reuse it for the generation prompt instead of merging again; ``chunks``
    (raw, unmerged) stays the source for ``_dedupe_sources``.

    On a ``not_covered`` verdict below scholar depth, this escalates once. It re-applies ``scholar``
    to ``base_cfg`` (never the already-tiered ``tiered_cfg``, so the scholar settings aren't layered
    on top of fast or thinking). Only if a real ``scholar`` entry exists (otherwise ``apply_tier``
    falls back to another tier and re-retrieves no deeper than before) it re-retrieves with the same
    ``qvec``, query text, game filter, and ``k``, re-merges, and checks again. A ``not_covered``
    verdict that starts at scholar depth, or still stands because there was nowhere deeper to
    escalate, sets ``declined``.

    ``history`` is passed to ``answerability.check()`` as well as the retrieval query. A terse
    follow-up ("and what is her element?") has no antecedent of its own, so the check needs the
    previous question exactly as routing does, or it judges coverage blind and declines follow-ups
    it shouldn't.

    ``http_post`` is for tests and evals only. It goes straight through to ``answerability.check()``
    (and from there to the shared ``router._jev_call``), so an eval script can inject a counting and
    recording wrapper without reimplementing this pipeline. Production callers never pass it and
    use the real HTTP transport.

    ``answer()`` and ``answer_stream()`` share this retrieve, merge, check, escalate, re-retrieve,
    re-merge, check pipeline; the streaming caller also emits an
    ``("tier", {"tier": g.tier, "source": "escalated"})`` event when ``escalated``.
    ``merge_fragmented_pages`` is called through this module's own name, not
    ``retrieve.merge_fragmented_pages``, so a test that monkeypatches ``rag.merge_fragmented_pages``
    (the ``spy_retrieval`` fixture patches it to identity) takes effect here."""
    retrieval_query = _retrieval_query(question, history)
    cfg = tiered_cfg
    chunks = retrieve(retrieval_query, cfg, k=k, game_filter=game_filter, embedder=embedder,
                      query_embedding=qvec)
    prompt_chunks = merge_fragmented_pages(chunks, cfg)
    escalated = declined = False
    if _answerability_would_run(base_cfg, picked):
        verdict = answerability.check(question, prompt_chunks, base_cfg, http_post=http_post,
                                      history=history)
        if answerability.is_not_covered(verdict, base_cfg) and tier != "scholar":
            scholar_cfg = apply_tier(base_cfg, "scholar")
            if scholar_cfg.get("answer_tier") == "scholar":
                cfg = scholar_cfg
                tier = "scholar"
                chunks = retrieve(retrieval_query, cfg, k=k, game_filter=game_filter, embedder=embedder,
                                  query_embedding=qvec)
                prompt_chunks = merge_fragmented_pages(chunks, cfg)
                escalated = True
                verdict = answerability.check(question, prompt_chunks, base_cfg, http_post=http_post,
                                              history=history)
            # else: no scholar tier configured. Escalating would retrieve no deeper than the current
            # tier did, so skip it; the original verdict stands and decides `declined` below.
        declined = answerability.is_not_covered(verdict, base_cfg)
    return Grounding(cfg, tier, chunks, prompt_chunks, escalated, declined)


def _route_and_embed(question: str, base_cfg: dict, tier: str | None, history, game_filter, embedder):
    """Shared prologue for ``answer()`` and ``answer_stream()``. It submits the query embed to
    ``_EXECUTOR`` before routing, so Jev's HTTP round-trip and the CPU embed run concurrently, then
    routes.

    Returns ``(tiered_cfg, applied_tier, source, picked, embed_future, off_topic)``. When
    ``off_topic`` is true, callers must not await ``embed_future``; they call
    ``embed_future.cancel()`` instead. If the embed hasn't started, cancelling skips it, so it
    doesn't tie up a thread in the small shared pool on a vector this question will never use. If it
    has started, cancel is a no-op: the embed finishes in the background, is discarded, and still
    warms a cold embedder for the next question. What to return or yield on the off-topic
    short-circuit stays with each caller, since ``answer()`` and ``answer_stream()`` shape it
    differently."""
    embed_future = _EXECUTOR.submit(_embed_query, _retrieval_query(question, history), base_cfg, embedder)
    try:
        tiered_cfg, applied_tier, source, picked = _routed(question, base_cfg, tier, history, game_filter)
    except BaseException:
        embed_future.cancel()      # nothing will await this embed now, so don't leave it queued
        raise
    off_topic = off_topic_gate_on(base_cfg) and picked.topic == "off_topic"
    return tiered_cfg, applied_tier, source, picked, embed_future, off_topic


def answer(question: str, cfg: dict | None = None, game_filter: str | None = None, k: int | None = None,
           llm=None, embedder=None, history=None, tier: str | None = None) -> dict:
    """Retrieve context, generate a grounded answer, and return {answer, sources, tier}.

    The query embedding starts on ``_EXECUTOR`` before routing, so Jev's HTTP round-trip and the
    (CPU) Qwen embed run concurrently. When the off-topic gate fires (routed ``topic == "off_topic"``
    and ``router.off_topic_gate`` is on), this returns the canned message immediately, without
    awaiting the embedding future, retrieving, or building a prompt. It cancels the future instead
    (a no-op if it already started, in which case it finishes in the background, is discarded, and
    still warms a cold embedder for the next question). Otherwise retrieval and the answerability
    check, escalation, and decline run via ``_ground`` (see its docstring). When the result is
    declined, this returns ``NOT_COVERED_MESSAGE`` with the closest sources and never constructs a
    ``GeminiClient``."""
    if not (question and question.strip()):
        return {"answer": NO_QUESTION_MESSAGE, "sources": []}
    if cfg is None:
        cfg = load_config()
    if llm is None:
        _preflight(cfg)        # fail before any embedding or paid routing work, not after
    base_cfg = cfg   # kept apart from any tier-applied cfg: the escalation re-applies scholar to THIS
    tiered_cfg, applied_tier, _source, picked, embed_future, off_topic = _route_and_embed(
        question, base_cfg, tier, history, game_filter, embedder)
    if off_topic:
        # No use for the embed here: cancel it (a no-op if it already started running) so a thread
        # in the small, shared pool isn't tied up finishing a vector this question will never use.
        embed_future.cancel()
        return {"answer": OFF_TOPIC_MESSAGE, "sources": [], "tier": None}
    qvec = embed_future.result()
    g = _ground(question, base_cfg, tiered_cfg, applied_tier, picked, qvec, game_filter, k, embedder,
               history)
    if g.declined:
        return {"answer": NOT_COVERED_MESSAGE, "sources": _dedupe_sources(g.chunks), "tier": g.tier}
    system, user = build_prompt(question, g.prompt_chunks, game_filter=game_filter, history=history,
                                answer_format=picked.format)
    if llm is None:
        llm = GeminiClient(g.cfg)
    text = llm.generate(system, user)
    if not (text and text.strip()):
        text = EMPTY_ANSWER_FALLBACK
    return {"answer": text, "sources": _dedupe_sources(g.chunks), "tier": g.tier}


def answer_stream(question: str, cfg: dict | None = None, game_filter: str | None = None, k: int | None = None,
                  llm=None, embedder=None, history=None, tier: str | None = None):
    """Stream a grounded answer as ``(kind, payload)`` events.

    The query embedding starts on ``_EXECUTOR`` before routing, same as ``answer()``. When the
    off-topic gate fires, this yields ``("text", OFF_TOPIC_MESSAGE)`` then ``("sources", [])``, with
    no ``tier`` event (so the UI shows no mode caption), and never awaits the embedding future.
    Otherwise it yields ``("tier", {"tier": str, "source": str})`` first (the routing decision). When
    the answerability check (``_ground``) escalates to scholar depth, a second
    ``("tier", {"tier": "scholar", "source": "escalated"})`` event follows, and the UI replaces the
    caption instead of appending. If the (possibly escalated) result is declined, this yields
    ``("text", NOT_COVERED_MESSAGE)``, ``("declined", True)`` (the UI relabels the sources toggle as
    the closest matches rather than what the answer is grounded in), then ``("sources", [dicts])``,
    and returns without an LLM call.
    Otherwise it yields ``("text", chunk)`` deltas as the model produces them, then
    ``("sources", [dicts])``.
    Any failure (a failing embedding future, retrieval, model, credentials) is surfaced as a final
    ``("error", message)`` event rather than raised, so the SSE connection always closes cleanly with
    something the UI can show. A ``SetupError`` (no API key, no vector store, unusable config) relays
    its own message; anything else becomes a generic one, since its text may be an
    internal detail.
    """
    if not (question and question.strip()):
        yield ("text", NO_QUESTION_MESSAGE)
        yield ("sources", [])
        return
    try:
        if cfg is None:
            cfg = load_config()
        if llm is None:
            _preflight(cfg)        # fail before any embedding or paid routing work, not after
        base_cfg = cfg   # kept apart from any tier-applied cfg: the escalation re-applies scholar to THIS
        tiered_cfg, applied_tier, source, picked, embed_future, off_topic = _route_and_embed(
            question, base_cfg, tier, history, game_filter, embedder)
        if off_topic:
            # No use for the embed here: cancel it (a no-op if it already started running) so a
            # thread in the small, shared pool isn't tied up finishing a vector this question will
            # never use.
            embed_future.cancel()
            yield ("text", OFF_TOPIC_MESSAGE)
            yield ("sources", [])
            return
        yield ("tier", {"tier": applied_tier, "source": source})
        qvec = embed_future.result()
        g = _ground(question, base_cfg, tiered_cfg, applied_tier, picked, qvec, game_filter, k, embedder,
                   history)
        if g.escalated:
            yield ("tier", {"tier": g.tier, "source": "escalated"})
        if g.declined:
            yield ("text", NOT_COVERED_MESSAGE)
            yield ("declined", True)
            yield ("sources", _dedupe_sources(g.chunks))
            return
        system, user = build_prompt(question, g.prompt_chunks, game_filter=game_filter, history=history,
                                    answer_format=picked.format)
        if llm is None:
            llm = GeminiClient(g.cfg)
        acc = ""
        for piece in llm.generate_stream(system, user):
            if piece:
                acc += piece
                yield ("text", piece)
        if not acc.strip():
            yield ("text", EMPTY_ANSWER_FALLBACK)
        yield ("sources", _dedupe_sources(g.chunks))
    except SetupError as exc:
        # A user-fixable setup problem (no API key, no vector store, bad config): its message is
        # written to be shown, so relay it instead of the generic failure text below.
        log.warning("answer_stream setup problem: %s", exc)
        yield ("error", str(exc))
    except Exception as exc:
        # Log with request context (a truncated question + the game filter) and a full traceback so a
        # field failure is triageable from logs alone. The generic user-facing message below carries
        # none of that. The question is truncated to avoid dumping an arbitrarily long payload.
        log.warning("answer_stream failed (question=%r, game_filter=%r): %s",
                    (question or "")[:200], game_filter, exc, exc_info=True)
        yield ("error", "Something went wrong while answering that. Please try again in a moment.")
