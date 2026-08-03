"""Retrieval-augmented generation: retrieve grounded context, prompt an LLM, cite sources.

The LLM is pluggable. `GeminiClient` is the default (reads credentials from the environment at call
time and never embeds a key); `MockLLM` is used in tests. Generation is grounded: the system prompt
forbids inventing facts and requires citing sources.
"""

import logging
import os
import re

from .config import load_config
from .retrieve import merge_fragmented_pages, retrieve

log = logging.getLogger(__name__)

# Current default Gemini model when a config omits `gemini_model` (never a retired id like 1.5-flash).
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
# Shown when the model returns nothing usable (e.g. a safety block) so the UI never goes blank.
EMPTY_ANSWER_FALLBACK = (
    "I could not generate an answer for that. Try rephrasing the question, narrowing to a specific "
    "game, or switching to the 'Thinking' answer style."
)
NO_QUESTION_MESSAGE = "Please enter a question to ask about the Xeno series."

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
    except Exception:  # noqa: BLE001 - any SDK error -> try the parts, then give up cleanly
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


class GeminiClient:
    """Gemini adapter using the supported `google-genai` SDK.

    Reads GOOGLE_API_KEY (or GEMINI_API_KEY) at call time. Raises if absent so the autonomous
    pipeline never silently makes a billable call without credentials.
    """

    def __init__(self, cfg: dict):
        self.model = cfg.get("gemini_model") or DEFAULT_GEMINI_MODEL

    def _client_and_types(self):
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "Gemini credentials not found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) to enable "
                "live answers."
            )
        from google import genai
        from google.genai import types

        return genai.Client(api_key=key), types

    def generate(self, system: str, prompt: str) -> str:
        client, types = self._client_and_types()
        resp = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return _extract_text(resp)

    def generate_stream(self, system: str, prompt: str):
        """Yield text deltas as the model produces them (lower time-to-first-token)."""
        client, types = self._client_and_types()
        stream = client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        for chunk in stream:
            piece = _extract_text(chunk)
            if piece:
                yield piece


def _retrieval_query(question: str, history=None) -> str:
    """The text used for retrieval. For a follow-up, prepend the previous user question so a pronoun
    ("what is HER element?") still pulls the right page (the antecedent isn't in the new question)."""
    if history:
        prev = (history[-1].get("question") or "").strip()
        if prev:
            return f"{prev} {question}"
    return question


def _history_block(history) -> str:
    """Render up to the last 6 turns as a Q/A transcript for the prompt. Answers in full, not clipped.
    A bounded sliding window (not the whole session) is what keeps a long chat from rotting the context:
    answers are re-grounded on fresh retrieval every turn, so older turns add mostly noise / topic-bleed
    and little signal. The retrieval query borrows only the single previous question (_retrieval_query),
    so retrieval itself is never polluted by session length. Within that 6-turn window the answers are
    passed whole: history is just Q/A text (no retrieved chunk data), so it's cheap (bounded by the
    model's own output length x6) and clipping risked hiding a detail a follow-up depends on."""
    if not history:
        return ""
    lines = []
    for turn in history[-6:]:
        q = (turn.get("question") or "").strip()
        a = (turn.get("answer") or "").strip()
        if q:
            lines.append(f"Q: {q}")
        if a:
            lines.append(f"A: {a}")
    if not lines:
        return ""
    return "Earlier in this conversation:\n" + "\n".join(lines) + "\n\n"


def _source_numbers(chunks):
    """Map each distinct source URL to its 1-based citation number, in first-seen chunk order.

    This is deliberately the SAME ordering rule as ``_dedupe_sources`` (first occurrence of each
    url wins), so a bracketed [n] the model emits always points at the n-th card in the sources
    payload the UI renders. The numbering and the SSE sources list can never drift apart. It is
    also stable across ``merge_fragmented_pages``: merging replaces a page's chunks with one block
    at the first occurrence's position and keeps its url, so first-seen url order is unchanged."""
    order = {}
    for c in chunks:
        url = c.get("url")
        if url and url not in order:
            order[url] = len(order) + 1
    return order


def build_prompt(question: str, chunks, game_filter: str | None = None, history=None):
    """Return (system, user) prompt strings grounding the answer in the retrieved chunks.

    Each context block is prefixed with the bracketed number of its source page ([1]..[n], numbered
    by ``_source_numbers`` so they match the UI's sources list), and the prompt instructs the model
    to cite claims with those markers, tightening "a citation on every answer" to per-claim. When a
    game filter is active, a scope line tells the model which game the user is focused on so it
    resolves ambiguous names within that game (e.g. "Jin" -> the XC2 Flesh Eater under XC2). Prior
    conversation turns (history) are included so follow-up questions resolve against them."""
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
    convo = _history_block(history)
    user = f"{convo}{scope}Context:\n{context}\n\n{cite}Question: {question}"
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
    """Bucket a 0–1 relevance into a source-bubble size tier (drives the UI bubble size)."""
    if relevance >= 0.66:
        return "high"
    if relevance >= 0.33:
        return "med"
    return "low"


def _score_relevance(sources):
    """Annotate each source with a 0–1 ``relevance`` and a size ``tier`` in place.

    The list arrives already ordered best-first (the cross-encoder ``_score`` / fusion decided the
    order), so ``relevance`` is taken from **rank position**: top → 1.0, bottom → 0.0. Rank, not the
    raw score magnitude, drives the size on purpose: real cross-encoder scores often cluster (a dozen
    near-equal pages), and min-max-normalizing those would collapse every bubble into one tier, i.e.
    the "all the same size" look. Rank guarantees a visible gradient (and clean thirds) for any set.
    The ``_score`` is dropped from the payload here; it has already done its job (ordering)."""
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
    preview (best-ranked chunk for that page); ``relevance``/``tier`` size the bubble by how
    correlated the page is to the question (see ``_score_relevance``)."""
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


def _apply_answer_style(cfg: dict) -> dict:
    """Pair the chosen generation model with its retrieval depth. ``cfg["answer_styles"]`` maps a model
    name to overrides (top_k, max_chunks_per_page, hybrid_candidates, rerank_candidates); the selected
    ``gemini_model``'s entry is merged over the base cfg so "Thinking" (flash) reads more of the wiki
    than "Faster" (flash-lite). Models absent from the map keep the base depth. Returns a new dict (or
    the original cfg unchanged). Never mutates the input."""
    style = (cfg.get("answer_styles") or {}).get(cfg.get("gemini_model"))
    return {**cfg, **style} if style else cfg


def answer(question: str, cfg: dict | None = None, game_filter: str | None = None, k: int | None = None,
           llm=None, embedder=None, history=None) -> dict:
    """Retrieve context, generate a grounded answer, and return {answer, sources}."""
    if not (question and question.strip()):
        return {"answer": NO_QUESTION_MESSAGE, "sources": []}
    if cfg is None:
        cfg = load_config()
    cfg = _apply_answer_style(cfg)
    chunks = retrieve(_retrieval_query(question, history), cfg, k=k, game_filter=game_filter,
                      embedder=embedder)
    prompt_chunks = merge_fragmented_pages(chunks, cfg)
    system, user = build_prompt(question, prompt_chunks, game_filter=game_filter, history=history)
    if llm is None:
        llm = GeminiClient(cfg)
    text = llm.generate(system, user)
    if not (text and text.strip()):
        text = EMPTY_ANSWER_FALLBACK
    return {"answer": text, "sources": _dedupe_sources(chunks)}


def answer_stream(question: str, cfg: dict | None = None, game_filter: str | None = None, k: int | None = None,
                  llm=None, embedder=None, history=None):
    """Stream a grounded answer as ``(kind, payload)`` events.

    Yields ``("text", chunk)`` deltas as the model produces them, then ``("sources", [dicts])``.
    Any failure (retrieval, model, credentials) is surfaced as a final ``("error", message)`` event
    rather than raised, so the SSE connection always closes cleanly with something the UI can show.
    """
    if not (question and question.strip()):
        yield ("text", NO_QUESTION_MESSAGE)
        yield ("sources", [])
        return
    if cfg is None:
        cfg = load_config()
    cfg = _apply_answer_style(cfg)
    try:
        chunks = retrieve(_retrieval_query(question, history), cfg, k=k, game_filter=game_filter,
                          embedder=embedder)
        prompt_chunks = merge_fragmented_pages(chunks, cfg)
        system, user = build_prompt(question, prompt_chunks, game_filter=game_filter, history=history)
        if llm is None:
            llm = GeminiClient(cfg)
        acc = ""
        for piece in llm.generate_stream(system, user):
            if piece:
                acc += piece
                yield ("text", piece)
        if not acc.strip():
            yield ("text", EMPTY_ANSWER_FALLBACK)
        yield ("sources", _dedupe_sources(chunks))
    except Exception as exc:
        # Log with request context (a truncated question + the game filter) and a full traceback so a
        # field failure is triageable from logs alone. The generic user-facing message below carries
        # none of that. The question is truncated to avoid dumping an arbitrarily long payload.
        log.warning("answer_stream failed (question=%r, game_filter=%r): %s",
                    (question or "")[:200], game_filter, exc, exc_info=True)
        yield ("error", "Something went wrong while answering that. Please try again in a moment.")
